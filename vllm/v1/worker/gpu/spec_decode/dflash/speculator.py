# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash block-diffusion speculator for the V2 model runner.

DFlash drafts a whole block of tokens in ONE non-causal draft forward:
the block is [bonus_token, MASK, MASK, ...] and every mask slot predicts
the token at its own position. Context comes from the target model's
auxiliary hidden states: they are fc-combined, normed, projected to K/V
by every draft layer and written into the draft KV cache (the draft
never runs autoregressively over the context).
"""
import os
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


class DFlashSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DFlash speculator does not support decode context parallelism yet."
        )

        draft_hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(draft_hf_config, "dflash_config", None) or {}
        if "mask_token_id" not in dflash_config:
            raise ValueError(
                "DFlash draft config must provide dflash_config.mask_token_id."
            )
        self.mask_token_id = int(dflash_config["mask_token_id"])

        # Block = bonus token + num_speculative_steps mask slots.
        self.block_size = self.num_speculative_steps + 1
        self.max_block_tokens = self.max_num_reqs * self.block_size
        assert self.max_block_tokens <= self.max_num_tokens, (
            "max_num_batched_tokens is too small for the DFlash draft block "
            f"({self.max_block_tokens} > {self.max_num_tokens})."
        )

        self.block_offsets = torch.arange(
            self.block_size, dtype=torch.int64, device=device
        )
        self.block_qsl_gpu = (
            torch.arange(self.max_num_reqs + 1, dtype=torch.int32, device=device)
            * self.block_size
        )
        self.block_qsl_cpu = (
            torch.arange(self.max_num_reqs + 1, dtype=torch.int32) * self.block_size
        )
        self.supports_mm_inputs = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        # The draft is plain GQA even when the target is MLA (e.g. Kimi K2.6),
        # so the draft layers must not inherit an MLA-only attention backend.
        draft_vllm_config = self.vllm_config
        spec_backend = self.speculative_config.attention_backend
        if spec_backend is not None:
            draft_vllm_config = replace(
                draft_vllm_config,
                attention_config=replace(
                    draft_vllm_config.attention_config,
                    backend=spec_backend,
                ),
            )
        # DFlash drafts the block bidirectionally.
        draft_vllm_config = replace(
            draft_vllm_config,
            attention_config=replace(
                draft_vllm_config.attention_config,
                use_non_causal=True,
            ),
        )
        model = load_eagle_model(target_model, draft_vllm_config)
        # load_eagle_model resolves lm_head on the top-level target module;
        # multimodal wrappers (e.g. KimiK25ForConditionalGeneration) keep it
        # on the language model, which would leave the weightless DFlash
        # draft with a randomly initialized head. Re-share in that case.
        if not getattr(model, "has_own_lm_head", False):
            target_lm = (
                target_model.get_language_model()
                if hasattr(target_model, "get_language_model")
                else target_model
            )
            lm_head = getattr(target_lm, "lm_head", None)
            if lm_head is not None and getattr(model, "lm_head", None) is not lm_head:
                if hasattr(model, "lm_head"):
                    del model.lm_head
                model.lm_head = lm_head
                logger.info_once(
                    "DFlash draft shares the target language model's lm_head.",
                    scope="local",
                )
        self._maybe_load_mask_embedding(model)
        return model

    def _maybe_load_mask_embedding(self, model: nn.Module) -> None:
        """Load a checkpoint-provided mask embedding into the embed table.

        DFlash FP4 exports ship the trained mask embedding separately when
        the target checkpoint's embedding row for mask_token_id is zeroed.
        """
        mask_path = os.path.join(self.draft_model_config.model, "mask_embedding.pt")
        if not os.path.exists(mask_path):
            return
        data = torch.load(mask_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            token_id = int(data.get("mask_token_id", self.mask_token_id))
            embedding = data.get("embedding")
        else:
            token_id = self.mask_token_id
            embedding = data
        if embedding is None:
            return
        embedding = embedding.reshape(-1)
        embed_tokens = model.model.embed_tokens
        weight = embed_tokens.weight
        shard_indices = getattr(embed_tokens, "shard_indices", None)
        if shard_indices is not None:
            start = shard_indices.org_vocab_start_index
            end = shard_indices.org_vocab_end_index
        else:
            start, end = 0, weight.shape[0]
        if start <= token_id < end:
            row = weight.data[token_id - start]
            row.copy_(embedding.to(device=row.device, dtype=row.dtype))
        logger.info_once(
            "Loaded DFlash mask embedding for token %d from %s.",
            token_id,
            mask_path,
            scope="local",
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # The DFlash draft pass runs eagerly for now: it is a single short
        # forward (block_size tokens per request) plus an eager context-KV
        # projection whose shape varies with the scheduled token count.
        logger.info_once(
            "DFlash V2 speculator runs the draft pass eagerly "
            "(CUDA graphs for the draft are not implemented yet).",
            scope="local",
        )

    def capture(self, attn_states) -> None:
        return

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_ctx = input_batch.num_tokens
        idx_mapping = input_batch.idx_mapping

        assert aux_hidden_states, (
            "DFlash requires auxiliary hidden states from the target model."
        )
        context_states = self.model.combine_hidden_states(
            torch.cat(aux_hidden_states, dim=-1)[:num_ctx]
        )

        use_cache = not is_profile and bool(slot_mappings)

        # 1) Project this step's target hidden states into draft context K/V
        # and write them at the same slots the target step used. Rejected
        # positions are written too; they fall outside the attention window
        # below and get overwritten once those positions are re-processed.
        context_slot_mapping = None
        if use_cache:
            context_slot_mapping = {
                name: slot_mappings[name][:num_ctx]
                for name in self.draft_attn_layer_names
            }
        self.model.precompute_and_store_context_kv(
            context_states,
            input_batch.positions[:num_ctx],
            context_slot_mapping,
        )

        # 2) Build the draft block: [bonus, MASK * num_speculative_steps].
        block = self.block_size
        num_block_tokens = num_reqs * block
        # Position of the bonus token = first not-yet-cached position.
        eff_seq_lens = (input_batch.seq_lens[:num_reqs] - num_rejected).to(torch.int64)
        # last_sampled is [max_num_reqs, 1]; flatten before indexing.
        last_sampled_flat = last_sampled.reshape(-1)[idx_mapping.to(torch.int64)]
        bonus_tokens = torch.where(
            num_sampled > 0,
            last_sampled_flat,
            next_prefill_tokens.reshape(-1)[idx_mapping.to(torch.int64)].to(
                last_sampled_flat.dtype
            ),
        )

        input_ids = self.input_buffers.input_ids[:num_block_tokens]
        input_ids.fill_(self.mask_token_id)
        input_ids[:: block] = bonus_tokens.to(input_ids.dtype)
        positions = self.input_buffers.positions[:num_block_tokens]
        positions.copy_(
            (eff_seq_lens.unsqueeze(1) + self.block_offsets).reshape(-1)
        )
        block_seq_lens = self.input_buffers.seq_lens[:num_reqs]
        block_seq_lens.copy_((eff_seq_lens + block).to(block_seq_lens.dtype))

        # 3) Per-group slot mappings + non-causal attention metadata for the
        # block tokens.
        block_attn_metadata = None
        block_slot_mappings_by_layer = None
        if use_cache:
            block_qsl_gpu = self.block_qsl_gpu[: num_reqs + 1]
            block_slot_mappings = self.block_tables.compute_slot_mappings(
                idx_mapping,
                block_qsl_gpu,
                positions,
                num_block_tokens,
            )
            block_slot_mappings_by_layer = build_slot_mappings_by_layer(
                block_slot_mappings, self.kv_cache_config
            )
            max_seq_len = int(
                input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
            ) + block
            block_attn_metadata = build_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=num_reqs,
                num_tokens=num_block_tokens,
                query_start_loc_gpu=block_qsl_gpu,
                query_start_loc_cpu=self.block_qsl_cpu[: num_reqs + 1],
                max_query_len=block,
                seq_lens=block_seq_lens,
                max_seq_len=min(max_seq_len, self.max_model_len),
                block_tables=[
                    x[:num_reqs] for x in self.block_tables.input_block_tables
                ],
                slot_mappings=block_slot_mappings,
                kv_cache_config=self.kv_cache_config,
                causal=False,
            )

        # 4) One non-causal draft pass over all blocks.
        with set_forward_context(
            block_attn_metadata,
            self.vllm_config,
            num_tokens=num_block_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=block_slot_mappings_by_layer,
            batch_descriptor=BatchDescriptor(num_tokens=num_block_tokens),
        ):
            hidden = self.model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=None,
            )

        # 5) Greedy-sample the mask slots: slot i predicts position i in place.
        mask_hidden = hidden.view(num_reqs, block, -1)[:, 1:, :].reshape(
            num_reqs * self.num_speculative_steps, -1
        )
        logits = self.model.compute_logits(mask_hidden)
        draft = logits.argmax(dim=-1).view(num_reqs, self.num_speculative_steps)
        self.draft_tokens[:num_reqs].copy_(draft)

        if os.environ.get("VLLM_DFLASH_V2_DEBUG") == "1" and not dummy_run:
            logger.info(
                "DFLASH-V2 dbg: num_reqs=%d num_ctx=%d eff_seq=%s bonus=%s "
                "blk_ids[:10]=%s blk_pos[:10]=%s blk_seq=%s drafts=%s "
                "ctx_pos[:5]=%s ctx_pos[-5:]=%s rej=%s sampled=%s",
                num_reqs,
                num_ctx,
                eff_seq_lens[:2].tolist(),
                bonus_tokens[:2].tolist(),
                input_ids[:10].tolist(),
                positions[:10].tolist(),
                block_seq_lens[:2].tolist(),
                draft[:1].tolist(),
                input_batch.positions[:5].tolist(),
                input_batch.positions[max(0, num_ctx - 5) : num_ctx].tolist(),
                num_rejected[:2].tolist(),
                num_sampled[:2].tolist(),
            )
        return self.draft_tokens[:num_reqs]
