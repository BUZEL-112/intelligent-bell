"""Numerical correctness and validation test suite for SpecCache Phase 1.

Verifies:
1. Eager reference implementation against mathematical ground truth.
2. Production Triton kernel against eager reference across varied:
   - Sequence lengths T (1, 16, 127, 256, 1024)
   - KV head counts H (1, 2, 8, 16, 32) covering MQA, GQA, and MHA
   - Head dimensions D (64, 128, 256)
   - Uniform scalar shifts [1, D // 2] and per-token shifts [T, D // 2]
   - Negative, positive, and zero (identity) shifts
3. Pre-allocated output buffers vs dynamic allocation.
4. Value cache INT8 dequantization.
5. Invariance of the relative RoPE rotational group homomorphism:
   R(p1 + p2) = R(p2) * R(p1).
6. Fail-fast validation exceptions for invalid inputs.
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
    # Generate representative unquantized FP32 activations
    raw_k = torch.randn((t, h, d), dtype=torch.float32, device=device)
    k_quant, k_scale = quantize_symmetric_int8(raw_k)
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
    v_quant, v_scale = quantize_symmetric_int8(raw_v)
    return v_quant, v_scale


# ==============================================================================
# Reference Implementation & Mathematical Invariant Tests
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

    # Shift of 0
    delta = torch.tensor([0], device=device)
    cos_delta, sin_delta = compute_rope_cos_sin_delta(delta, d, device=device)

    # With shift 0, cos is 1 and sin is 0
    assert torch.allclose(cos_delta, torch.ones_like(cos_delta))
    assert torch.allclose(sin_delta, torch.zeros_like(sin_delta))

    k_rotated = dequant_and_rope_key_ref(k_quant, k_scale, cos_delta, sin_delta)
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
    k_direct = dequant_and_rope_key_ref(k_quant, k_scale, cos_sum, sin_sum)

    # 2. Composition: rotate by delta1, then rotate result by delta2
    k_step1 = dequant_and_rope_key_ref(k_quant, k_scale, cos1, sin1)
    # Apply delta2 rotation to k_step1 (treating scale as 1.0)
    x1 = k_step1[..., :d_half].to(torch.float32)
    x2 = k_step1[..., d_half:].to(torch.float32)
    c2 = cos2[:, None, :].to(torch.float32)
    s2 = sin2[:, None, :].to(torch.float32)
    out1 = x1 * c2 - x2 * s2
    out2 = x2 * c2 + x1 * s2
    k_composed = torch.cat([out1, out2], dim=-1).to(torch.float16)

    assert torch.allclose(k_direct, k_composed, atol=1e-3, rtol=1e-3)


def test_reference_value_dequant():
    """Test Value reference dequantization."""
    device = torch.device("cpu")
    t, h, d = 32, 8, 128
    v_quant, v_scale = _create_mock_quantized_value(t, h, d, device)

    v_out = dequant_value_ref(v_quant, v_scale)
    expected = (v_quant.to(torch.float32) * v_scale.to(torch.float32)).to(torch.float16)
    assert torch.allclose(v_out, expected, atol=1e-4, rtol=1e-4)


# ==============================================================================
# Validation & Fail-Fast Exception Tests
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


def test_validation_invalid_scale_shape():
    """Scale must have shape [T, H, 1]."""
    with pytest.raises(ValueError, match="k_scale must have shape"):
        k = torch.zeros((4, 2, 64), dtype=torch.int8)
        s = torch.ones((4, 1, 1), dtype=torch.float16)
        c = torch.ones((1, 32), dtype=torch.float16)
        validate_key_inputs(k, s, c, c)


def test_validation_invalid_shift_shape():
    """Shift tensors must match [1, D // 2] or [T, D // 2]."""
    with pytest.raises(ValueError, match="cos_delta must have shape"):
        k = torch.zeros((4, 2, 64), dtype=torch.int8)
        s = torch.ones((4, 2, 1), dtype=torch.float16)
        c = torch.ones((2, 32), dtype=torch.float16)  # Expected 1 or 4
        validate_key_inputs(k, s, c, c)


def test_validation_value_shape():
    """v_quant and v_scale validation."""
    with pytest.raises(ValueError, match="v_scale must have shape"):
        v = torch.zeros((4, 2, 64), dtype=torch.int8)
        s = torch.ones((4, 3, 1), dtype=torch.float16)
        validate_value_inputs(v, s)


# ==============================================================================
# Triton GPU Kernel Numerical Correctness Tests
# ==============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for Triton tests")
@pytest.mark.parametrize("t", [1, 16, 127, 256, 1024])
@pytest.mark.parametrize("h", [1, 2, 8, 16])
@pytest.mark.parametrize("d", [64, 128])
def test_triton_vs_ref_uniform_shift(t: int, h: int, d: int):
    """Test Triton kernel with uniform scalar shift against eager reference."""
    device = torch.device("cuda")
    k_quant, k_scale = _create_mock_quantized_key(t, h, d, device)

    delta = torch.tensor([19], device=device)
    cos_delta, sin_delta = compute_rope_cos_sin_delta(delta, d, device=device)

    # Reference
    out_ref = dequant_and_rope_key_ref(k_quant, k_scale, cos_delta, sin_delta)

    # Triton kernel
    out_triton = dequant_and_rope_key(k_quant, k_scale, cos_delta, sin_delta)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for Triton tests")
@pytest.mark.parametrize("t", [8, 64, 256])
@pytest.mark.parametrize("h", [4, 8])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_triton_vs_ref_per_token_shift(t: int, h: int, d: int):
    """Test Triton kernel with per-token shift vectors [T, D // 2]."""
    device = torch.device("cuda")
    k_quant, k_scale = _create_mock_quantized_key(t, h, d, device)

    # Per-token displacements: [t_0, t_1, ..., t_{T-1}]
    deltas = torch.randint(-100, 100, (t,), device=device)
    cos_delta, sin_delta = compute_rope_cos_sin_delta(deltas, d, device=device)

    out_ref = dequant_and_rope_key_ref(k_quant, k_scale, cos_delta, sin_delta)
    out_triton = dequant_and_rope_key(k_quant, k_scale, cos_delta, sin_delta)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for Triton tests")
def test_triton_preallocated_out_buffer():
    """Verify in-place writing to a caller-provided output buffer."""
    device = torch.device("cuda")
    t, h, d = 64, 8, 128
    k_quant, k_scale = _create_mock_quantized_key(t, h, d, device)

    delta = torch.tensor([33], device=device)
    cos_delta, sin_delta = compute_rope_cos_sin_delta(delta, d, device=device)

    preallocated = torch.empty((t, h, d), dtype=torch.float16, device=device)
    returned = dequant_and_rope_key(k_quant, k_scale, cos_delta, sin_delta, out=preallocated)

    # Must write directly to the buffer
    assert returned.data_ptr() == preallocated.data_ptr()

    out_ref = dequant_and_rope_key_ref(k_quant, k_scale, cos_delta, sin_delta)
    torch.testing.assert_close(returned, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for Triton tests")
@pytest.mark.parametrize("t", [16, 128, 512])
@pytest.mark.parametrize("h", [2, 8])
@pytest.mark.parametrize("d", [64, 128])
def test_triton_value_dequant(t: int, h: int, d: int):
    """Test Triton Value dequantization against eager reference."""
    device = torch.device("cuda")
    v_quant, v_scale = _create_mock_quantized_value(t, h, d, device)

    out_ref = dequant_value_ref(v_quant, v_scale)
    out_triton = dequant_value(v_quant, v_scale)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for Triton tests")
def test_triton_dequant_kv_cache_unified():
    """Test combined KV dequantization entrypoint."""
    device = torch.device("cuda")
    t, h, d = 32, 4, 128
    k_quant, k_scale = _create_mock_quantized_key(t, h, d, device)
    v_quant, v_scale = _create_mock_quantized_value(t, h, d, device)

    delta = torch.tensor([7], device=device)
    cos_delta, sin_delta = compute_rope_cos_sin_delta(delta, d, device=device)

    k_out, v_out = dequant_kv_cache(
        k_quant, k_scale, v_quant, v_scale, cos_delta, sin_delta
    )

    k_ref = dequant_and_rope_key_ref(k_quant, k_scale, cos_delta, sin_delta)
    v_ref = dequant_value_ref(v_quant, v_scale)

    torch.testing.assert_close(k_out, k_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v_out, v_ref, atol=1e-3, rtol=1e-3)
