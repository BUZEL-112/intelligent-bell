"""Rigorous numerical verification suite for SpecCache Phase 1.

Verifies:
1. Reference unit tests: rotate_half, quantize_kv_int8, group homomorphism.
2. Production Triton kernel vs. Eager PyTorch reference across:
   - Sequence lengths T: [1, 17, 128, 511, 2048, 4096] (includes prime/unaligned lengths)
   - KV Heads H: [4, 8, 32] (covering GQA like Llama-3-8B / Qwen-2.5 and MHA)
   - Head Dimension D: [64, 128]
   - Position shifts Delta p: negative (-512), zero (0), and positive (+1024)
   - Numerical tolerance: rtol=1e-2, atol=1e-2
3. End-to-end Rotational Equivalence Test:
   Proves R(m + Delta p) = R(Delta p) * R(m) under INT8 quantization noise:
   - Raw FP16 keys K rotated at position m = 600, quantized to INT8.
   - Dequantized and shifted by Delta p = -350 using Triton kernel.
   - Compared against raw K rotated directly at target position 250.
4. Input validation and fail-fast exception handling.
"""

from typing import Tuple
import pytest
import torch

from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    dequant_and_rope_key_compiled,
    dequant_and_rope_key_ref,
    dequant_value_compiled,
    dequant_value_ref,
    pytorch_compiled_dequant_rope_shift,
    pytorch_eager_dequant_rope_shift,
    quantize_kv_int8,
    quantize_symmetric_int8,
    rotate_half,
    validate_key_inputs,
    validate_value_inputs,
)

try:
    from speccache.kernels.triton_rope_kv import (
        TRITON_AVAILABLE,
        dequant_and_rope_key,
        dequant_kv_cache,
        dequant_value,
        fused_k_dequant_rope_shift_kernel,
        triton_fused_k_dequant_rope_shift,
    )
except ImportError:
    TRITON_AVAILABLE = False


def _create_mock_quantized_key(
    t: int,
    h: int,
    d: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper to generate reproducible INT8 quantized keys and scales."""
    torch.manual_seed(42)
    raw_k = torch.randn((t, h, d), dtype=torch.float32, device=device)
    k_quant, k_scale = quantize_kv_int8(raw_k)
    return k_quant, k_scale


def _create_mock_quantized_value(
    t: int,
    h: int,
    d: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper to generate reproducible INT8 quantized values and scales."""
    torch.manual_seed(1337)
    raw_v = torch.randn((t, h, d), dtype=torch.float32, device=device)
    v_quant, v_scale = quantize_kv_int8(raw_v)
    return v_quant, v_scale


# ==============================================================================
# Unit & Mathematical Invariant Tests (CPU & GPU)
# ==============================================================================

def test_rotate_half():
    """Verify rotate_half maps [x1, x2] to [-x2, x1]."""
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # shape [1, 1, 4]
    rotated = rotate_half(x)
    expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])
    assert torch.equal(rotated, expected)


def test_quantize_kv_int8_absmax():
    """Verify dynamic per-token-head absmax scale calculation and int8 clamping."""
    x = torch.tensor([[[-127.0, 0.0, 63.5, 127.0]]], dtype=torch.float32)
    q, s = quantize_kv_int8(x)
    assert q.dtype == torch.int8
    assert s.dtype == torch.float16
    assert s.shape == (1, 1, 1)
    assert torch.isclose(s.squeeze(), torch.tensor(1.0, dtype=torch.float16), atol=1e-4)
    assert torch.equal(q, torch.tensor([[[-127, 0, 64, 127]]], dtype=torch.int8))


def test_reference_zero_shift_identity():
    """Rotating by Delta p = 0 must mathematically equal pure INT8 dequantization."""
    device = torch.device("cpu")
    t, h, d = 16, 4, 64
    k_quant, k_scale = _create_mock_quantized_key(t, h, d, device)

    delta = torch.tensor([0], device=device)
    cos_delta, sin_delta = compute_rope_cos_sin_delta(delta, d, device=device)

    assert torch.allclose(cos_delta, torch.ones_like(cos_delta))
    assert torch.allclose(sin_delta, torch.zeros_like(sin_delta))

    k_rotated = pytorch_eager_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta)
    k_direct = (k_quant.to(torch.float32) * k_scale.to(torch.float32)).to(torch.float16)

    assert torch.allclose(k_rotated, k_direct, atol=1e-4, rtol=1e-4)


def test_reference_rotational_composition():
    """Verify group homomorphism: R(delta1 + delta2) == R(delta2) * R(delta1)."""
    device = torch.device("cpu")
    t, h, d = 8, 2, 64
    d_half = d // 2

    k_quant, k_scale = _create_mock_quantized_key(t, h, d, device)

    delta1 = torch.tensor([15], device=device)
    delta2 = torch.tensor([27], device=device)
    delta_sum = torch.tensor([42], device=device)

    cos1, sin1 = compute_rope_cos_sin_delta(delta1, d, device=device)
    cos2, sin2 = compute_rope_cos_sin_delta(delta2, d, device=device)
    cos_sum, sin_sum = compute_rope_cos_sin_delta(delta_sum, d, device=device)

    # 1. Single direct rotation by (delta1 + delta2)
    k_direct = pytorch_eager_dequant_rope_shift(k_quant, k_scale, cos_sum, sin_sum)

    # 2. Composition: rotate by delta1, then rotate result by delta2
    k_step1 = pytorch_eager_dequant_rope_shift(k_quant, k_scale, cos1, sin1)
    cos2_full = torch.cat([cos2, cos2], dim=-1)[:, None, :].to(torch.float32)
    sin2_full = torch.cat([sin2, sin2], dim=-1)[:, None, :].to(torch.float32)
    k1_f32 = k_step1.to(torch.float32)
    k_composed = (k1_f32 * cos2_full + rotate_half(k1_f32) * sin2_full).to(torch.float16)

    assert torch.allclose(k_direct, k_composed, atol=1e-3, rtol=1e-3)


# ==============================================================================
# Triton GPU Kernel Correctness Suite (Section C)
# ==============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for Triton tests")
@pytest.mark.parametrize("t", [1, 17, 128, 511, 2048, 4096])
@pytest.mark.parametrize("h", [4, 8, 32])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("delta_p", [-512, 0, 1024])
def test_triton_vs_ref_correctness(t: int, h: int, d: int, delta_p: int):
    """Verify Triton kernel matches PyTorch reference across shapes and shifts.

    Stress tests:
    - Prime / unaligned token counts (T=17, T=511)
    - GQA and MHA KV head configurations (H=4, 8, 32)
    - Head dimensions (D=64, 128)
    - Negative, zero, and positive relative position shifts
    - Numerical precision within rtol=1e-2, atol=1e-2
    """
    device = torch.device("cuda")
    k_quant, k_scale = _create_mock_quantized_key(t, h, d, device)

    delta = torch.tensor([delta_p], device=device)
    cos_delta, sin_delta = compute_rope_cos_sin_delta(delta, d, device=device)

    # Reference execution
    ref_out = pytorch_eager_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta)

    # Triton 2D-tiled execution
    triton_out = triton_fused_k_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta)

    torch.testing.assert_close(triton_out, ref_out, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for Triton tests")
def test_end_to_end_rotational_equivalence():
    """End-to-end proof of RoPE relative rotational identity under INT8 quantization.

    Scenario:
    1. Take raw FP16 keys K.
    2. Rotate by initial position m = 600.
    3. Quantize to INT8.
    4. Use Triton kernel to dequantize and relative shift by Delta p = -350.
    5. Compare against raw K rotated directly at target position m + Delta p = 250
       (quantized to INT8 and dequantized to match quantization stage).
    """
    device = torch.device("cuda")
    torch.manual_seed(999)

    t, h, d = 256, 8, 128
    # 1. Raw FP16 keys K
    raw_k = torch.randn((t, h, d), dtype=torch.float16, device=device)

    # 2. Rotate by m = 600
    cos_600, sin_600 = compute_rope_cos_sin_delta(torch.tensor([600], device=device), d, device=device)
    cos_600_full = torch.cat([cos_600, cos_600], dim=-1)[:, None, :].to(torch.float32)
    sin_600_full = torch.cat([sin_600, sin_600], dim=-1)[:, None, :].to(torch.float32)

    raw_k_f32 = raw_k.to(torch.float32)
    k_rot_600 = (raw_k_f32 * cos_600_full + rotate_half(raw_k_f32) * sin_600_full).to(torch.float16)

    # 3. Quantize to INT8
    k_quant_600, k_scale_600 = quantize_kv_int8(k_rot_600)

    # 4. Dequantize and relative shift by Delta p = -350 using Triton kernel
    cos_neg350, sin_neg350 = compute_rope_cos_sin_delta(
        torch.tensor([-350], device=device), d, device=device
    )
    triton_shifted = triton_fused_k_dequant_rope_shift(
        k_quant_600, k_scale_600, cos_neg350, sin_neg350
    )

    # 5. Direct rotation at target position 250 (600 - 350 = 250)
    cos_250, sin_250 = compute_rope_cos_sin_delta(torch.tensor([250], device=device), d, device=device)
    cos_250_full = torch.cat([cos_250, cos_250], dim=-1)[:, None, :].to(torch.float32)
    sin_250_full = torch.cat([sin_250, sin_250], dim=-1)[:, None, :].to(torch.float32)

    k_rot_250 = (raw_k_f32 * cos_250_full + rotate_half(raw_k_f32) * sin_250_full).to(torch.float16)
    k_quant_250, k_scale_250 = quantize_kv_int8(k_rot_250)
    direct_target = (k_quant_250.to(torch.float32) * k_scale_250.to(torch.float32)).to(torch.float16)

    # Both pathways undergo one round of INT8 quantization and float16 conversion
    torch.testing.assert_close(triton_shifted, direct_target, rtol=1e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for Triton tests")
def test_triton_preallocated_buffer():
    """Verify write directly into pre-allocated output buffer."""
    device = torch.device("cuda")
    t, h, d = 64, 8, 128
    k_quant, k_scale = _create_mock_quantized_key(t, h, d, device)

    cos_delta, sin_delta = compute_rope_cos_sin_delta(torch.tensor([12], device=device), d, device=device)
    preallocated = torch.empty((t, h, d), dtype=torch.float16, device=device)

    ret = triton_fused_k_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta, out=preallocated)
    assert ret.data_ptr() == preallocated.data_ptr()

    ref = pytorch_eager_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta)
    torch.testing.assert_close(ret, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for Triton tests")
def test_triton_value_dequant_correctness():
    """Verify Value state dequantization."""
    device = torch.device("cuda")
    t, h, d = 64, 8, 128
    v_quant, v_scale = _create_mock_quantized_value(t, h, d, device)

    ref = dequant_value_ref(v_quant, v_scale)
    triton_out = dequant_value(v_quant, v_scale)

    torch.testing.assert_close(triton_out, ref, rtol=1e-2, atol=1e-2)


# ==============================================================================
# Fail-Fast Input Validation Tests
# ==============================================================================

def test_validation_odd_head_dim():
    """Head dimension must be even for RoPE."""
    with pytest.raises(ValueError, match="head_dim must be even"):
        k = torch.zeros((4, 2, 63), dtype=torch.int8)
        s = torch.ones((4, 2, 1), dtype=torch.float16)
        c = torch.ones((1, 31), dtype=torch.float16)
        validate_key_inputs(k, s, c, c)


def test_validation_invalid_quant_dtype():
    """Non-int8 keys must raise TypeError."""
    with pytest.raises(TypeError, match="must be torch.int8"):
        k = torch.zeros((4, 2, 64), dtype=torch.float16)
        s = torch.ones((4, 2, 1), dtype=torch.float16)
        c = torch.ones((1, 32), dtype=torch.float16)
        validate_key_inputs(k, s, c, c)
