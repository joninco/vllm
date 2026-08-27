# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 pooled sparse-attention selector backed by b12x."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.utils.b12x import get_b12x_glm_pooled_indexer

_INDEX_HEADS = 32
_INDEX_HEAD_DIM = 128
_POOL_SIZE = 4
_TOPK_TOKENS = 2048
_SELECTION_WIDTH = _TOPK_TOKENS + _POOL_SIZE - 1
_MLA_RECORD_BYTES = 528


class Glm5NextPooledIndexer(nn.Module):
    """Own the GLM selector projections, persistent state, and cache binding."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Any,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        *,
        main_layer_name: str,
        prefix: str,
    ) -> None:
        super().__init__()
        if cache_config is None:
            raise ValueError("GLM pooled selection requires a paged MLA cache")
        if topk_indices_buffer is None:
            raise ValueError("GLM pooled selection requires a top-k output buffer")
        if tuple(topk_indices_buffer.shape[1:]) != (_SELECTION_WIDTH,):
            raise ValueError("GLM pooled selection requires a 2051-column top-k buffer")

        geometry = {
            "index_topk": _TOPK_TOKENS,
            "index_n_heads": _INDEX_HEADS,
            "index_head_dim": _INDEX_HEAD_DIM,
            "index_kpool": _POOL_SIZE,
        }
        for name, expected in geometry.items():
            actual = getattr(config, name, None)
            if actual is None or int(actual) != expected:
                raise ValueError(
                    f"GLM pooled selection requires {name}={expected}, got {actual}"
                )
        if int(getattr(config, "qk_rope_head_dim", -1)) != 0:
            raise ValueError("GLM-5.3 pooled selection requires a no-RoPE indexer")

        self.topk_tokens = _TOPK_TOKENS
        self.topk_indices_buffer = topk_indices_buffer
        self.main_layer_name = main_layer_name
        self.max_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        self.max_speculative_tokens = int(vllm_config.num_speculative_tokens)
        self.block_size = int(cache_config.block_size)
        if self.block_size % _POOL_SIZE:
            raise ValueError("GLM MLA cache block size must be divisible by four")
        if self.max_tokens != int(topk_indices_buffer.shape[0]):
            raise ValueError("GLM top-k buffer does not match scheduler token capacity")

        module = get_b12x_glm_pooled_indexer()
        if module is None:
            raise RuntimeError("GLM-5.3 pooled selection requires the b12x package")
        requirements = module.cache_requirements(
            compressed_page_size=self.block_size // _POOL_SIZE,
            max_speculative_tokens=self.max_speculative_tokens,
            index_head_dim=_INDEX_HEAD_DIM,
            compress_ratio=_POOL_SIZE,
            budget=_TOPK_TOKENS,
            dtype=torch.bfloat16,
        )
        self.raw_ring_capacity = int(requirements.raw_ring_capacity)

        device = topk_indices_buffer.device
        self.index_kpool_compress_ape = nn.Parameter(
            torch.empty(
                _POOL_SIZE,
                _INDEX_HEAD_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(
                _INDEX_HEAD_DIM,
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            _INDEX_HEADS * _INDEX_HEAD_DIM,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            _INDEX_HEAD_DIM,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wk",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            _INDEX_HEADS,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.k_norm = LayerNorm(_INDEX_HEAD_DIM, eps=1e-6)

        table_width = math.ceil(self._aligned_max_seq_len / self.block_size)
        self._compressed_table_width = table_width
        self._compressed_block_table: torch.Tensor
        self.register_buffer(
            "_compressed_block_table",
            torch.full(
                (self.max_seqs, table_width),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            persistent=False,
        )
        raw_shape = (self.max_seqs, self.raw_ring_capacity, _INDEX_HEAD_DIM)
        self.register_buffer(
            "_raw_k_ring",
            torch.empty(raw_shape, dtype=torch.bfloat16, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_raw_gate_ring",
            torch.empty(raw_shape, dtype=torch.bfloat16, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_raw_logical_positions",
            torch.full(
                (self.max_seqs, self.raw_ring_capacity),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_raw_interval_start_positions",
            torch.full((self.max_seqs,), -1, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_raw_interval_start_snapshot",
            torch.empty(self.max_seqs, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_raw_state_slot_ids",
            torch.full((self.max_seqs,), -1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_sequence_lengths",
            torch.zeros(self.max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_decode_query_start_loc",
            torch.zeros(self.max_seqs + 1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_prefill_query_start_loc",
            torch.zeros(self.max_seqs + 1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_prefill_request_ids",
            torch.empty(self.max_tokens, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_num_accepted_tokens",
            torch.ones(self.max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_reset_mask",
            torch.zeros(self.max_seqs, dtype=torch.bool, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_prefix_lengths",
            torch.zeros(self.max_seqs, dtype=torch.int32, device=device),
            persistent=False,
        )

        self._selector_plan: Any | None = None
        self._selector_binding: Any | None = None
        self._selector_scratch: torch.Tensor | None = None
        self._compressed_cache: torch.Tensor | None = None

    @property
    def _aligned_max_seq_len(self) -> int:
        return math.ceil(self.max_model_len / _POOL_SIZE) * _POOL_SIZE

    @staticmethod
    def _compressed_cache_view(main_cache: torch.Tensor) -> torch.Tensor:
        if main_cache.ndim != 3 or int(main_cache.shape[-1]) != _MLA_RECORD_BYTES:
            raise ValueError("GLM MLA cache must have shape [pages, block, 528]")
        if main_cache.dtype != torch.uint8 or int(main_cache.stride(-1)) != 1:
            raise TypeError("GLM MLA cache records must be byte-addressable uint8")
        pages, block_size, _ = map(int, main_cache.shape)
        if pages <= 0 or block_size <= 0:
            raise ValueError("GLM MLA cache must contain positive page capacity")
        if tuple(map(int, main_cache.stride()[1:])) != (
            _MLA_RECORD_BYTES,
            1,
        ):
            raise ValueError("GLM MLA cache must have contiguous logical records")
        if block_size % _POOL_SIZE:
            raise ValueError("GLM MLA cache block size must be divisible by four")
        page_stride_bytes = int(main_cache.stride(0))
        semantic_page_bytes = block_size * _MLA_RECORD_BYTES
        tail_bytes = block_size // _POOL_SIZE * _INDEX_HEAD_DIM * 2
        if page_stride_bytes < semantic_page_bytes + tail_bytes:
            raise ValueError("GLM MLA cache page does not contain the selector tail")

        if page_stride_bytes % 2 or int(main_cache.storage_offset()) % 2:
            raise ValueError("GLM MLA cache storage must be BF16-aligned")
        bf16_cache = main_cache.view(torch.bfloat16)
        page_stride_elements = page_stride_bytes // 2
        tail_elements = tail_bytes // 2
        tail_offset = int(bf16_cache.storage_offset()) + semantic_page_bytes // 2
        tail_end = tail_offset + (pages - 1) * page_stride_elements + tail_elements
        storage_elements = int(main_cache.untyped_storage().nbytes()) // 2
        if tail_end > storage_elements:
            raise ValueError("GLM MLA cache storage does not include the selector tail")

        return torch.as_strided(
            bf16_cache,
            size=(pages, block_size // _POOL_SIZE, _INDEX_HEAD_DIM),
            stride=(page_stride_elements, _INDEX_HEAD_DIM, 1),
            storage_offset=tail_offset,
        )

    def bind_main_kv_cache(self, main_cache: torch.Tensor) -> None:
        module = get_b12x_glm_pooled_indexer()
        if module is None or not module.is_supported(main_cache.device):
            raise RuntimeError("b12x GLM pooled selection is unavailable")
        compressed_cache = self._compressed_cache_view(main_cache)
        bound_block_size = int(main_cache.shape[1])
        compressed_page_size = int(compressed_cache.shape[1])
        table_width = math.ceil(self._aligned_max_seq_len / bound_block_size)
        if tuple(self._compressed_block_table.shape) != (
            self.max_seqs,
            table_width,
        ):
            self._compressed_block_table = torch.full(
                (self.max_seqs, table_width),
                -1,
                dtype=torch.int32,
                device=main_cache.device,
            )
        planned_pages = max(int(compressed_cache.shape[0]), table_width)
        plan = module.plan(
            module.Caps(
                device=main_cache.device,
                max_batch=self.max_seqs,
                max_raw_state_slots=self.max_seqs,
                max_q_rows=self.max_tokens,
                max_seq_len=self._aligned_max_seq_len,
                num_compressed_cache_pages=planned_pages,
                compressed_page_size=compressed_page_size,
                max_speculative_tokens=self.max_speculative_tokens,
                index_heads=_INDEX_HEADS,
                index_head_dim=_INDEX_HEAD_DIM,
                compress_ratio=_POOL_SIZE,
                budget=_TOPK_TOKENS,
                dtype=torch.bfloat16,
            )
        )
        specs = plan.scratch_specs()
        if len(specs) != 1:
            raise RuntimeError("GLM pooled selector must expose one scratch buffer")
        scratch = torch.empty(
            specs[0].shape,
            dtype=specs[0].dtype,
            device=main_cache.device,
        )
        binding = plan.bind(
            scratch=scratch,
            compressed_k_cache=compressed_cache,
            compressed_block_table=self._compressed_block_table,
            raw_k_ring=self._raw_k_ring,
            raw_gate_ring=self._raw_gate_ring,
            raw_logical_positions=self._raw_logical_positions,
            raw_interval_start_positions=self._raw_interval_start_positions,
            raw_state_slot_ids=self._raw_state_slot_ids,
            position_embedding=self.index_kpool_compress_ape,
            selected_positions=self.topk_indices_buffer,
        )
        self._selector_plan = plan
        self._selector_binding = binding
        self._selector_scratch = scratch
        self._compressed_cache = compressed_cache
        self.block_size = bound_block_size
        self._compressed_table_width = table_width

    def unbind_main_kv_cache(self) -> None:
        self._selector_binding = None
        self._selector_plan = None
        self._selector_scratch = None
        self._compressed_cache = None

    def _metadata(self) -> Any | None:
        raw = get_forward_context().attn_metadata
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise RuntimeError("GLM pooled selection requires per-layer metadata")
        metadata = raw.get(self.main_layer_name)
        if metadata is None:
            raise RuntimeError(
                f"GLM selector metadata is missing for {self.main_layer_name}"
            )
        return metadata

    def _project_head_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Compute the selector head gate in FP32 because BF16 rounding can
        # change near-tie pool rankings at long context lengths.
        if getattr(self, "_weights_proj_fp32", None) is None:
            self._weights_proj_fp32 = self.weights_proj.weight.detach().float()
        return F.linear(hidden_states.float(), self._weights_proj_fp32)

    @staticmethod
    def _require_metadata_tensors(
        metadata: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = (
            getattr(metadata, "selector_state_slot_ids", None),
            getattr(metadata, "selector_state_is_fresh", None),
            getattr(metadata, "selector_num_accepted_tokens", None),
        )
        if any(value is None for value in values):
            raise RuntimeError("GLM pooled selector runtime metadata is incomplete")
        return cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], values)

    def _stage_metadata(self, metadata: Any, rows: int) -> None:
        num_reqs = int(metadata.num_reqs)
        if num_reqs > self.max_seqs or rows > self.max_tokens:
            raise ValueError("GLM selector batch exceeds its planned capacity")
        table_width = self._compressed_table_width
        if int(metadata.block_table.shape[1]) < table_width:
            raise ValueError("GLM selector block table is narrower than max_model_len")

        state_slots, state_is_fresh, accepted = self._require_metadata_tensors(metadata)
        active_block_table = self._compressed_block_table[:, :table_width]
        active_block_table.fill_(-1)
        active_block_table[:num_reqs].copy_(
            metadata.block_table[:num_reqs, :table_width]
        )
        self._sequence_lengths.zero_()
        self._sequence_lengths[:num_reqs].copy_(metadata.seq_lens[:num_reqs])
        self._raw_state_slot_ids.fill_(-1)
        self._raw_state_slot_ids[:num_reqs].copy_(state_slots[:num_reqs])
        self._num_accepted_tokens.fill_(1)
        self._num_accepted_tokens[:num_reqs].copy_(accepted[:num_reqs])
        self._reset_mask.zero_()
        self._reset_mask[:num_reqs].copy_(state_is_fresh[:num_reqs])
        self._prefix_lengths.zero_()
        torch.sub(
            metadata.seq_lens[:num_reqs],
            torch.diff(metadata.query_start_loc[: num_reqs + 1]),
            out=self._prefix_lengths[:num_reqs],
        )

        num_decodes = int(metadata.num_decodes)
        decode_rows = int(metadata.num_decode_tokens)
        self._decode_query_start_loc.fill_(decode_rows)
        self._decode_query_start_loc[: num_decodes + 1].copy_(
            metadata.query_start_loc[: num_decodes + 1]
        )
        prefill_rows = rows - decode_rows
        self._prefill_query_start_loc.fill_(prefill_rows)
        if num_decodes < num_reqs:
            torch.sub(
                metadata.query_start_loc[num_decodes : num_reqs + 1],
                decode_rows,
                out=self._prefill_query_start_loc[: num_reqs - num_decodes + 1],
            )

    def _stage_prefill_metadata(
        self,
        metadata: Any,
        *,
        num_decodes: int,
        decode_rows: int,
        rows: int,
    ) -> torch.Tensor:
        num_reqs = int(metadata.num_reqs)
        prefill_reqs = num_reqs - num_decodes
        prefill_rows = rows - decode_rows
        table_width = self._compressed_table_width
        state_slots, _, _ = self._require_metadata_tensors(metadata)

        active_block_table = self._compressed_block_table[:, :table_width]
        active_block_table.fill_(-1)
        active_block_table[:prefill_reqs].copy_(
            metadata.block_table[num_decodes:num_reqs, :table_width]
        )
        self._sequence_lengths.zero_()
        self._sequence_lengths[:prefill_reqs].copy_(
            metadata.seq_lens[num_decodes:num_reqs]
        )
        self._raw_state_slot_ids.fill_(-1)
        self._raw_state_slot_ids[:prefill_reqs].copy_(state_slots[num_decodes:num_reqs])
        prefill_request_ids = self._prefill_request_ids[:prefill_rows]
        prefill_request_ids.copy_(metadata.req_id_per_token[decode_rows:rows])
        prefill_request_ids.sub_(num_decodes)
        return prefill_request_ids

    @eager_break_during_capture
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb: nn.Module | None,
    ) -> torch.Tensor:
        del rotary_emb
        if q_lora is None:
            raise RuntimeError("GLM pooled selection requires q_lora_rank")
        if positions.ndim != 1 or positions.dtype != torch.int64:
            raise ValueError("GLM pooled selection requires scalar int64 positions")

        index_query = self.wq_b(q_lora)[0].view(-1, _INDEX_HEADS, _INDEX_HEAD_DIM)
        raw_key = self.wk(hidden_states)[0]
        normalized_key = self.k_norm(raw_key)
        head_weights = self._project_head_weights(hidden_states)
        gate_logits = F.linear(hidden_states, self.index_kpool_compress_gate)

        metadata = self._metadata()
        if metadata is None:
            self.topk_indices_buffer[: index_query.shape[0]].fill_(-1)
            return self.topk_indices_buffer[: index_query.shape[0]]
        binding = self._selector_binding
        if binding is None:
            raise RuntimeError("GLM pooled selector cache is not bound")

        rows = int(metadata.num_actual_tokens)
        self._stage_metadata(metadata, rows)
        module = get_b12x_glm_pooled_indexer()
        assert module is not None
        module.reset_state(
            binding,
            reset_mask=self._reset_mask,
            prefix_lengths=self._prefix_lengths,
        )

        request_ids = metadata.req_id_per_token[:rows]
        query_positions = positions[:rows]
        decode_rows = int(metadata.num_decode_tokens)
        if decode_rows:
            module.run(
                binding,
                index_query=index_query[:decode_rows].contiguous(),
                normalized_index_key=normalized_key[:decode_rows].contiguous(),
                index_gate_logits=gate_logits[:decode_rows].contiguous(),
                index_head_weights=head_weights[:decode_rows].contiguous(),
                request_ids=request_ids[:decode_rows].contiguous(),
                query_positions=query_positions[:decode_rows].contiguous(),
                sequence_lengths=self._sequence_lengths,
                query_start_loc=self._decode_query_start_loc,
                num_accepted_tokens=self._num_accepted_tokens,
            )
        if decode_rows < rows:
            prefill_request_ids = request_ids[decode_rows:rows]
            if decode_rows:
                # The decode launch must consume binding-owned metadata before
                # the same CUDA stream compacts it for the prefill launch.
                prefill_request_ids = self._stage_prefill_metadata(
                    metadata,
                    num_decodes=int(metadata.num_decodes),
                    decode_rows=decode_rows,
                    rows=rows,
                )
            module.run_prefill(
                binding,
                index_query=index_query[decode_rows:rows].contiguous(),
                normalized_index_key=normalized_key[decode_rows:rows].contiguous(),
                index_gate_logits=gate_logits[decode_rows:rows].contiguous(),
                index_head_weights=head_weights[decode_rows:rows].contiguous(),
                request_ids=prefill_request_ids.contiguous(),
                query_positions=query_positions[decode_rows:rows].contiguous(),
                sequence_lengths=self._sequence_lengths,
                query_start_loc=self._prefill_query_start_loc,
                output_row_start=decode_rows,
            )
        return self.topk_indices_buffer[:rows]

    def snapshot_speculative_interval_starts(self) -> None:
        self._raw_interval_start_snapshot.copy_(self._raw_interval_start_positions)

    def restore_speculative_interval_starts(self) -> None:
        self._raw_interval_start_positions.copy_(self._raw_interval_start_snapshot)


__all__ = ["Glm5NextPooledIndexer"]
