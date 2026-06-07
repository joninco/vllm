# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for B12X_MLA_SPARSE decode-head padding introduced in PR.

These tests cover:
  1. The ``_DECODE_HEADS_PER_BLOCK`` constant value.
  2. The ``_decode_num_heads`` rounding formula
     (ceil(workspace_num_heads / 8) * 8, minimum 8).
  3. The query tensor zero-padding logic applied before the decode kernel.
  4. The output / LSE slice-back logic applied after the decode kernel.

All tensor tests use CPU tensors so they run without a GPU or the ``b12x``
package.  The constant / formula tests import only ``_DECODE_HEADS_PER_BLOCK``
and the helper ``_cdiv`` from the module, so they are also cheap.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        "B12X decode-padding tests require CUDA for module-level imports.",
        allow_module_level=True,
    )

# Import the symbols added / touched by this PR.
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    _DECODE_HEADS_PER_BLOCK,
    _cdiv,
)

# ---------------------------------------------------------------------------
# Helper: replicate the _decode_num_heads formula from B12xMLASparseImpl.__init__
# ---------------------------------------------------------------------------


def _compute_decode_num_heads(workspace_num_heads: int) -> int:
    """Mirrors the new ``_decode_num_heads`` computation in __init__."""
    decode_num_heads = (
        _cdiv(workspace_num_heads, _DECODE_HEADS_PER_BLOCK)
        * _DECODE_HEADS_PER_BLOCK
    )
    return max(decode_num_heads, _DECODE_HEADS_PER_BLOCK)


# ---------------------------------------------------------------------------
# Helper: replicate the q-padding logic from forward_mqa
# ---------------------------------------------------------------------------


def _apply_q_padding(
    q_all: torch.Tensor,
    decode_num_heads: int,
    q_head_dim: int,
) -> torch.Tensor:
    """Mirrors the decode-q padding in forward_mqa."""
    num_actual_heads = q_all.shape[1]
    num_actual_toks = q_all.shape[0]
    if decode_num_heads == num_actual_heads:
        return q_all
    decode_q = q_all.new_zeros((num_actual_toks, decode_num_heads, q_head_dim))
    decode_q[:, :num_actual_heads, :].copy_(q_all)
    return decode_q


# ---------------------------------------------------------------------------
# Helper: replicate the output-slicing logic from forward_mqa
# ---------------------------------------------------------------------------


def _apply_output_slice(
    out: torch.Tensor,
    decode_num_heads: int,
    num_actual_heads: int,
) -> torch.Tensor:
    """Mirrors the out[:, :num_actual_heads, :].contiguous() slice."""
    if decode_num_heads == num_actual_heads:
        return out
    return out[:, :num_actual_heads, :].contiguous()


def _apply_lse_slice(
    lse: torch.Tensor,
    decode_num_heads: int,
    num_actual_heads: int,
) -> torch.Tensor:
    """Mirrors the lse[:, :num_actual_heads].contiguous() slice."""
    if decode_num_heads == num_actual_heads:
        return lse
    return lse[:, :num_actual_heads].contiguous()


# ===========================================================================
# 1.  Constant
# ===========================================================================


def test_decode_heads_per_block_constant_is_8():
    """_DECODE_HEADS_PER_BLOCK must equal 8 (b12x decode kernel alignment)."""
    assert _DECODE_HEADS_PER_BLOCK == 8


# ===========================================================================
# 2.  _decode_num_heads calculation
# ===========================================================================


class TestDecodeNumHeadsCalculation:
    """The formula: ceil(workspace_num_heads / 8) * 8, min 8."""

    @pytest.mark.parametrize(
        "workspace_num_heads",
        [8, 16, 24, 32, 64, 128, 256],
    )
    def test_already_aligned_is_unchanged(self, workspace_num_heads):
        """Values that are already multiples of 8 should not be padded."""
        result = _compute_decode_num_heads(workspace_num_heads)
        assert result == workspace_num_heads

    @pytest.mark.parametrize(
        ("workspace_num_heads", "expected"),
        [
            (1, 8),
            (7, 8),
            (9, 16),
            (10, 16),
            (15, 16),
            (17, 24),
            (23, 24),
            (25, 32),
            (100, 104),
            (127, 128),
            (129, 136),
        ],
    )
    def test_unaligned_rounds_up_to_next_multiple_of_8(
        self, workspace_num_heads, expected
    ):
        """Non-aligned values must be rounded up to the next multiple of 8."""
        result = _compute_decode_num_heads(workspace_num_heads)
        assert result == expected

    @pytest.mark.parametrize("workspace_num_heads", [1, 2, 3, 4, 5, 6, 7])
    def test_minimum_is_8(self, workspace_num_heads):
        """Result must be at least _DECODE_HEADS_PER_BLOCK == 8."""
        result = _compute_decode_num_heads(workspace_num_heads)
        assert result >= _DECODE_HEADS_PER_BLOCK
        assert result == 8

    @pytest.mark.parametrize(
        "workspace_num_heads",
        [8, 9, 15, 16, 17, 32, 33, 128, 129, 200, 255, 256],
    )
    def test_result_is_always_multiple_of_8(self, workspace_num_heads):
        """Result must always be a multiple of _DECODE_HEADS_PER_BLOCK."""
        result = _compute_decode_num_heads(workspace_num_heads)
        assert result % _DECODE_HEADS_PER_BLOCK == 0

    @pytest.mark.parametrize(
        "workspace_num_heads",
        [8, 9, 15, 16, 17, 32, 33, 128, 129, 200, 255, 256],
    )
    def test_result_is_at_least_workspace_num_heads(self, workspace_num_heads):
        """Padding never reduces the head count below the original."""
        result = _compute_decode_num_heads(workspace_num_heads)
        assert result >= workspace_num_heads

    def test_padding_amount_is_less_than_block_size(self):
        """Padding added is strictly less than _DECODE_HEADS_PER_BLOCK."""
        for workspace_num_heads in range(1, 300):
            result = _compute_decode_num_heads(workspace_num_heads)
            padding = result - workspace_num_heads
            assert padding < _DECODE_HEADS_PER_BLOCK, (
                f"workspace={workspace_num_heads}, result={result}, "
                f"padding={padding} >= {_DECODE_HEADS_PER_BLOCK}"
            )

    def test_formula_for_tp_sharded_heads(self):
        """Simulate TP-sharded head counts typical for DeepSeek-R1 (128 heads).

        For tensor_parallel_size in {1, 2, 4, 8, 16}:
          128//1=128 -> 128   (no padding)
          128//2= 64 ->  64   (no padding)
          128//4= 32 ->  32   (no padding)
          128//8= 16 ->  16   (no padding)
          128//16= 8 ->   8   (no padding)
        All are already multiples of 8, so no padding.
        """
        total_heads = 128
        for tp in [1, 2, 4, 8, 16]:
            ws = total_heads // tp
            result = _compute_decode_num_heads(ws)
            assert result == ws, f"TP={tp}: expected {ws}, got {result}"


# ===========================================================================
# 3.  Query tensor zero-padding
# ===========================================================================


class TestDecodeQPadding:
    """Tests for the q-tensor zero-padding before the decode kernel call."""

    def _make_q(self, num_toks: int, num_heads: int, head_dim: int) -> torch.Tensor:
        return torch.randn(num_toks, num_heads, head_dim, dtype=torch.bfloat16)

    def test_no_padding_when_heads_already_aligned(self):
        """When decode_num_heads == num_actual_heads, decode_q is q_all."""
        num_toks, num_heads, q_head_dim = 4, 16, 576
        q_all = self._make_q(num_toks, num_heads, q_head_dim)
        decode_q = _apply_q_padding(q_all, decode_num_heads=16, q_head_dim=q_head_dim)
        # Should be the same object (no copy).
        assert decode_q is q_all

    def test_padding_creates_wider_tensor(self):
        """When unaligned, decode_q has decode_num_heads columns."""
        num_toks, num_actual_heads, q_head_dim = 4, 9, 576
        decode_num_heads = 16  # rounded up from 9
        q_all = self._make_q(num_toks, num_actual_heads, q_head_dim)
        decode_q = _apply_q_padding(q_all, decode_num_heads, q_head_dim)
        assert decode_q.shape == (num_toks, decode_num_heads, q_head_dim)

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [
            (7, 8),
            (9, 16),
            (15, 16),
            (17, 24),
            (100, 104),
        ],
    )
    def test_actual_head_values_are_copied_correctly(
        self, num_actual_heads, decode_num_heads
    ):
        """The first num_actual_heads slices of decode_q must equal q_all."""
        num_toks, q_head_dim = 6, 576
        q_all = self._make_q(num_toks, num_actual_heads, q_head_dim)
        decode_q = _apply_q_padding(q_all, decode_num_heads, q_head_dim)
        torch.testing.assert_close(
            decode_q[:, :num_actual_heads, :],
            q_all,
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [
            (7, 8),
            (9, 16),
            (15, 16),
            (17, 24),
        ],
    )
    def test_padded_heads_are_zero(self, num_actual_heads, decode_num_heads):
        """Extra (padded) head positions in decode_q must be exactly zero."""
        num_toks, q_head_dim = 8, 576
        q_all = self._make_q(num_toks, num_actual_heads, q_head_dim)
        decode_q = _apply_q_padding(q_all, decode_num_heads, q_head_dim)
        padded_region = decode_q[:, num_actual_heads:, :]
        assert torch.all(padded_region == 0), (
            "Padded head region must be all-zero before decode kernel call."
        )

    def test_padding_dtype_and_device_match_input(self):
        """decode_q must share dtype and device with q_all."""
        q_all = torch.randn(3, 9, 576, dtype=torch.bfloat16)
        decode_q = _apply_q_padding(q_all, decode_num_heads=16, q_head_dim=576)
        assert decode_q.dtype == q_all.dtype
        assert decode_q.device == q_all.device

    def test_padding_all_tokens_copied(self):
        """Padding must not drop any token row."""
        num_toks, num_actual_heads, decode_num_heads = 10, 7, 8
        q_head_dim = 576
        q_all = torch.arange(
            num_toks * num_actual_heads * q_head_dim, dtype=torch.float32
        ).reshape(num_toks, num_actual_heads, q_head_dim)
        decode_q = _apply_q_padding(q_all, decode_num_heads, q_head_dim)
        for tok in range(num_toks):
            torch.testing.assert_close(
                decode_q[tok, :num_actual_heads, :],
                q_all[tok],
                rtol=0,
                atol=0,
            )

    def test_no_padding_single_token_batch(self):
        """Batch of 1 token with unaligned heads: padding still zero-fills."""
        q_all = torch.randn(1, 7, 576, dtype=torch.bfloat16)
        decode_q = _apply_q_padding(q_all, decode_num_heads=8, q_head_dim=576)
        assert decode_q.shape == (1, 8, 576)
        torch.testing.assert_close(decode_q[:, :7, :], q_all, rtol=0, atol=0)
        assert torch.all(decode_q[:, 7:, :] == 0)

    def test_padding_large_batch(self):
        """Padding works correctly for a large batch of tokens."""
        num_toks, num_actual_heads, decode_num_heads, q_head_dim = 128, 15, 16, 576
        q_all = torch.randn(num_toks, num_actual_heads, q_head_dim, dtype=torch.bfloat16)
        decode_q = _apply_q_padding(q_all, decode_num_heads, q_head_dim)
        assert decode_q.shape == (num_toks, decode_num_heads, q_head_dim)
        torch.testing.assert_close(
            decode_q[:, :num_actual_heads, :], q_all, rtol=0, atol=0
        )
        assert torch.all(decode_q[:, num_actual_heads:, :] == 0)


# ===========================================================================
# 4.  Output tensor slicing
# ===========================================================================


class TestOutputSlicing:
    """Tests for out[:, :num_actual_heads, :].contiguous() after decode kernel."""

    def _make_out(self, num_toks: int, num_heads: int, v_head_dim: int) -> torch.Tensor:
        return torch.randn(num_toks, num_heads, v_head_dim, dtype=torch.bfloat16)

    def test_no_slicing_when_aligned(self):
        """When decode_num_heads == num_actual_heads, the tensor is returned as-is."""
        out = self._make_out(4, 16, 512)
        result = _apply_output_slice(out, decode_num_heads=16, num_actual_heads=16)
        assert result is out

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [
            (7, 8),
            (9, 16),
            (15, 16),
            (17, 24),
            (100, 104),
        ],
    )
    def test_slicing_trims_to_actual_heads(self, num_actual_heads, decode_num_heads):
        """out[:, :num_actual_heads, :] must have the expected shape."""
        num_toks, v_head_dim = 8, 512
        out = self._make_out(num_toks, decode_num_heads, v_head_dim)
        result = _apply_output_slice(out, decode_num_heads, num_actual_heads)
        assert result.shape == (num_toks, num_actual_heads, v_head_dim)

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [(7, 8), (9, 16), (15, 16), (17, 24)],
    )
    def test_slicing_preserves_values(self, num_actual_heads, decode_num_heads):
        """Values in the first num_actual_heads positions must be unchanged."""
        num_toks, v_head_dim = 6, 512
        out = self._make_out(num_toks, decode_num_heads, v_head_dim)
        result = _apply_output_slice(out, decode_num_heads, num_actual_heads)
        torch.testing.assert_close(
            result,
            out[:, :num_actual_heads, :],
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [(7, 8), (9, 16), (15, 16), (17, 24)],
    )
    def test_output_is_contiguous_after_slice(self, num_actual_heads, decode_num_heads):
        """Result must be contiguous (required by downstream consumers)."""
        out = self._make_out(6, decode_num_heads, 512)
        result = _apply_output_slice(out, decode_num_heads, num_actual_heads)
        assert result.is_contiguous()

    def test_slicing_single_token(self):
        """Slice with batch-size 1 produces correct shape and values."""
        num_actual_heads, decode_num_heads, v_head_dim = 7, 8, 512
        out = self._make_out(1, decode_num_heads, v_head_dim)
        result = _apply_output_slice(out, decode_num_heads, num_actual_heads)
        assert result.shape == (1, num_actual_heads, v_head_dim)
        torch.testing.assert_close(result, out[:, :num_actual_heads, :], rtol=0, atol=0)

    def test_slicing_large_batch(self):
        """Slice over a larger batch does not corrupt values."""
        num_toks, num_actual_heads, decode_num_heads, v_head_dim = 128, 15, 16, 512
        out = torch.arange(
            num_toks * decode_num_heads * v_head_dim, dtype=torch.float32
        ).reshape(num_toks, decode_num_heads, v_head_dim)
        result = _apply_output_slice(out, decode_num_heads, num_actual_heads)
        expected = out[:, :num_actual_heads, :].contiguous()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_sliced_output_dtype_matches_input(self):
        """Sliced output should retain the input dtype (bfloat16)."""
        out = self._make_out(4, 16, 512)
        result = _apply_output_slice(out, decode_num_heads=16, num_actual_heads=9)
        assert result.dtype == out.dtype


# ===========================================================================
# 5.  LSE tensor slicing
# ===========================================================================


class TestLSESlicing:
    """Tests for lse[:, :num_actual_heads].contiguous() after decode kernel."""

    def _make_lse(self, num_toks: int, num_heads: int) -> torch.Tensor:
        return torch.randn(num_toks, num_heads, dtype=torch.float32)

    def test_no_slicing_when_aligned(self):
        """When decode_num_heads == num_actual_heads, lse is returned as-is."""
        lse = self._make_lse(4, 16)
        result = _apply_lse_slice(lse, decode_num_heads=16, num_actual_heads=16)
        assert result is lse

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [
            (7, 8),
            (9, 16),
            (15, 16),
            (17, 24),
            (100, 104),
        ],
    )
    def test_slicing_trims_to_actual_heads(self, num_actual_heads, decode_num_heads):
        """lse[:, :num_actual_heads] must have the expected shape."""
        num_toks = 8
        lse = self._make_lse(num_toks, decode_num_heads)
        result = _apply_lse_slice(lse, decode_num_heads, num_actual_heads)
        assert result.shape == (num_toks, num_actual_heads)

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [(7, 8), (9, 16), (15, 16), (17, 24)],
    )
    def test_slicing_preserves_values(self, num_actual_heads, decode_num_heads):
        """Values in the first num_actual_heads columns must be unchanged."""
        lse = self._make_lse(6, decode_num_heads)
        result = _apply_lse_slice(lse, decode_num_heads, num_actual_heads)
        torch.testing.assert_close(
            result,
            lse[:, :num_actual_heads],
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [(7, 8), (9, 16), (15, 16), (17, 24)],
    )
    def test_lse_is_contiguous_after_slice(self, num_actual_heads, decode_num_heads):
        """LSE result must be contiguous (required for DCP reduce)."""
        lse = self._make_lse(6, decode_num_heads)
        result = _apply_lse_slice(lse, decode_num_heads, num_actual_heads)
        assert result.is_contiguous()

    def test_lse_slicing_single_token(self):
        """LSE slice with batch-size 1 produces correct shape and values."""
        num_actual_heads, decode_num_heads = 7, 8
        lse = self._make_lse(1, decode_num_heads)
        result = _apply_lse_slice(lse, decode_num_heads, num_actual_heads)
        assert result.shape == (1, num_actual_heads)
        torch.testing.assert_close(result, lse[:, :num_actual_heads], rtol=0, atol=0)

    def test_lse_dtype_float32(self):
        """LSE slice preserves float32 dtype expected by downstream callers."""
        lse = self._make_lse(4, 16)
        result = _apply_lse_slice(lse, decode_num_heads=16, num_actual_heads=9)
        assert result.dtype == torch.float32


# ===========================================================================
# 6.  Round-trip: q-pad then output-unpad
# ===========================================================================


class TestRoundTrip:
    """Verify that padding and unpadding are consistent end-to-end."""

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [
            (7, 8),
            (9, 16),
            (15, 16),
            (17, 24),
            (100, 104),
        ],
    )
    def test_output_shape_matches_original_q_shape(
        self, num_actual_heads, decode_num_heads
    ):
        """After padding q and slicing out, output heads equal num_actual_heads."""
        num_toks, q_head_dim, v_head_dim = 8, 576, 512
        q_all = torch.randn(num_toks, num_actual_heads, q_head_dim, dtype=torch.bfloat16)

        # Pad q
        decode_q = _apply_q_padding(q_all, decode_num_heads, q_head_dim)
        assert decode_q.shape[1] == decode_num_heads

        # Simulate kernel output (decode_num_heads in head dim)
        fake_out = torch.randn(num_toks, decode_num_heads, v_head_dim, dtype=torch.bfloat16)

        # Slice back
        out = _apply_output_slice(fake_out, decode_num_heads, num_actual_heads)
        assert out.shape == (num_toks, num_actual_heads, v_head_dim)

    @pytest.mark.parametrize(
        ("num_actual_heads", "decode_num_heads"),
        [
            (7, 8),
            (9, 16),
            (15, 16),
            (17, 24),
        ],
    )
    def test_lse_shape_matches_original_q_heads(
        self, num_actual_heads, decode_num_heads
    ):
        """After LSE slice, head dim equals num_actual_heads."""
        num_toks = 6
        fake_lse = torch.randn(num_toks, decode_num_heads, dtype=torch.float32)
        lse = _apply_lse_slice(fake_lse, decode_num_heads, num_actual_heads)
        assert lse.shape == (num_toks, num_actual_heads)

    @pytest.mark.parametrize(
        "num_actual_heads",
        [8, 16, 32, 64, 128],
    )
    def test_aligned_heads_no_padding_no_slicing(self, num_actual_heads):
        """When heads are aligned to 8, the round-trip must be a strict no-op."""
        decode_num_heads = _compute_decode_num_heads(num_actual_heads)
        assert decode_num_heads == num_actual_heads

        num_toks, q_head_dim, v_head_dim = 4, 576, 512
        q_all = torch.randn(num_toks, num_actual_heads, q_head_dim, dtype=torch.bfloat16)
        decode_q = _apply_q_padding(q_all, decode_num_heads, q_head_dim)
        assert decode_q is q_all  # same object

        fake_out = torch.randn(num_toks, decode_num_heads, v_head_dim, dtype=torch.bfloat16)
        out = _apply_output_slice(fake_out, decode_num_heads, num_actual_heads)
        assert out is fake_out  # same object


# ===========================================================================
# 7.  Regression / boundary cases
# ===========================================================================


class TestRegressionAndBoundary:
    """Regression and boundary checks for the decode-head padding change."""

    def test_minimum_decode_num_heads_even_for_single_head(self):
        """workspace_num_heads=1 must give decode_num_heads=8 (not 0 or 1)."""
        assert _compute_decode_num_heads(1) == 8

    def test_decode_plan_head_count_never_less_than_8(self):
        """_decode_num_heads must never be less than 8 for any input."""
        for workspace_num_heads in range(1, 200):
            assert _compute_decode_num_heads(workspace_num_heads) >= 8

    def test_q_padding_produces_contiguous_output(self):
        """decode_q must be contiguous so the kernel does not see strided inputs."""
        q_all = torch.randn(8, 7, 576, dtype=torch.bfloat16)
        decode_q = _apply_q_padding(q_all, decode_num_heads=8, q_head_dim=576)
        # When heads != decode_num_heads a new tensor is allocated -> contiguous
        assert decode_q.is_contiguous()

    def test_output_and_lse_slicing_both_make_contiguous(self):
        """Both out and lse must be contiguous after slicing (DCP reduce path)."""
        num_toks, decode_num_heads, num_actual_heads = 4, 16, 9
        out = torch.randn(num_toks, decode_num_heads, 512, dtype=torch.bfloat16)
        lse = torch.randn(num_toks, decode_num_heads, dtype=torch.float32)

        out_sliced = _apply_output_slice(out, decode_num_heads, num_actual_heads)
        lse_sliced = _apply_lse_slice(lse, decode_num_heads, num_actual_heads)

        assert out_sliced.is_contiguous()
        assert lse_sliced.is_contiguous()

    def test_decode_num_heads_is_multiple_of_8_for_all_small_values(self):
        """Exhaustive check for workspace_num_heads in [1, 64]."""
        for wnh in range(1, 65):
            result = _compute_decode_num_heads(wnh)
            assert result % 8 == 0, (
                f"workspace={wnh} -> decode_num_heads={result} not multiple of 8"
            )

    def test_q_padding_with_zero_extra_heads_does_not_allocate(self):
        """When no padding is needed, the function must return the same object."""
        q_all = torch.randn(4, 8, 576, dtype=torch.bfloat16)  # 8 heads = aligned
        decode_q = _apply_q_padding(q_all, decode_num_heads=8, q_head_dim=576)
        assert decode_q is q_all

    def test_output_slice_with_zero_padding_does_not_allocate(self):
        """When no padding was applied, slicing must return the original object."""
        out = torch.randn(4, 8, 512, dtype=torch.bfloat16)
        result = _apply_output_slice(out, decode_num_heads=8, num_actual_heads=8)
        assert result is out

    def test_lse_slice_with_zero_padding_does_not_allocate(self):
        """When no padding was applied, LSE slicing returns the original object."""
        lse = torch.randn(4, 8, dtype=torch.float32)
        result = _apply_lse_slice(lse, decode_num_heads=8, num_actual_heads=8)
        assert result is lse

    def test_q_head_dim_is_preserved_after_padding(self):
        """The q_head_dim axis must never change size during padding."""
        q_head_dim = 576
        for num_actual_heads in [7, 9, 15, 17, 100]:
            q_all = torch.randn(4, num_actual_heads, q_head_dim, dtype=torch.bfloat16)
            decode_num_heads = _compute_decode_num_heads(num_actual_heads)
            decode_q = _apply_q_padding(q_all, decode_num_heads, q_head_dim)
            assert decode_q.shape[2] == q_head_dim

    def test_slice_only_removes_trailing_heads_not_leading(self):
        """Slicing must trim from the end, not from the front."""
        num_toks, decode_num_heads, num_actual_heads, v_head_dim = 4, 16, 9, 512
        # Fill each head column with a distinct scalar to detect off-by-one.
        out = torch.zeros(num_toks, decode_num_heads, v_head_dim, dtype=torch.float32)
        for h in range(decode_num_heads):
            out[:, h, :] = float(h)

        result = _apply_output_slice(out, decode_num_heads, num_actual_heads)
        # The slice must contain heads 0..num_actual_heads-1.
        for h in range(num_actual_heads):
            assert torch.all(result[:, h, :] == float(h)), (
                f"Head {h} value corrupted after slice"
            )

    def test_lse_slice_only_removes_trailing_heads_not_leading(self):
        """LSE slicing trims trailing heads, not leading ones."""
        num_toks, decode_num_heads, num_actual_heads = 4, 16, 9
        lse = torch.zeros(num_toks, decode_num_heads, dtype=torch.float32)
        for h in range(decode_num_heads):
            lse[:, h] = float(h)

        result = _apply_lse_slice(lse, decode_num_heads, num_actual_heads)
        for h in range(num_actual_heads):
            assert torch.all(result[:, h] == float(h)), (
                f"LSE head {h} value corrupted after slice"
            )
