"""Reference PyTorch and torch.compile implementations for SpecCache Phase 1.

This module provides high-precision eager PyTorch baselines and compiled
variants for:
1. Per-token-head INT8 dequantization fused with relative RoPE phase-shifting
2. Per-token-head INT8 dequantization for Value states
3. Helper utilities for RoPE frequency tables and symmetric quantization
"""

from typing import Optional, Tuple
import torch


def validate_key_inputs(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    cos_delta: torch.Tensor,
    sin_delta: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> None:
    """Validate tensor shapes, dtypes, and device compatibility for key operations.

    Fails early with informative exceptions if tensor invariants are violated.
    """
    if k_quant.ndim != 3:
        raise ValueError(
            f"k_quant must be 3-dimensional [num_tokens, num_kv_heads, head_dim], "
            f"got ndim={k_quant.ndim} with shape {k_quant.shape}"
        )

    t, h, d = k_quant.shape

    if d % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE half-pairing, got head_dim={d}")

    if k_quant.dtype != torch.int8:
        raise TypeError(f"k_quant must be torch.int8, got {k_quant.dtype}")

    if k_scale.ndim != 3 or k_scale.shape != (t, h, 1):
        raise ValueError(
            f"k_scale must have shape [{t}, {h}, 1], got {k_scale.shape}"
        )

    if k_scale.dtype != torch.float16:
        raise TypeError(f"k_scale must be torch.float16, got {k_scale.dtype}")

    d_half = d // 2
    valid_shift_shapes = {(1, d_half), (t, d_half)}

    if cos_delta.shape not in valid_shift_shapes:
        raise ValueError(
            f"cos_delta must have shape [1, {d_half}] or [{t}, {d_half}], "
            f"got {cos_delta.shape}"
        )

    if sin_delta.shape != cos_delta.shape:
        raise ValueError(
            f"sin_delta shape {sin_delta.shape} must match cos_delta shape {cos_delta.shape}"
        )

    if cos_delta.dtype != torch.float16 or sin_delta.dtype != torch.float16:
        raise TypeError(
            f"cos_delta and sin_delta must be torch.float16, got "
            f"{cos_delta.dtype} and {sin_delta.dtype}"
        )

    device = k_quant.device
    if (
        k_scale.device != device
        or cos_delta.device != device
        or sin_delta.device != device
    ):
        raise ValueError(
            f"All tensors must reside on the same device ({device}), but got devices: "
            f"k_scale={k_scale.device}, cos_delta={cos_delta.device}, sin_delta={sin_delta.device}"
        )

    if out is not None:
        if out.shape != (t, h, d):
            raise ValueError(
                f"Pre-allocated out tensor must have shape [{t}, {h}, {d}], got {out.shape}"
            )
        if out.dtype != torch.float16:
            raise TypeError(f"out tensor must be torch.float16, got {out.dtype}")
        if out.device != device:
            raise ValueError(f"out tensor device ({out.device}) does not match input device ({device})")


def validate_value_inputs(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> None:
    """Validate tensor shapes, dtypes, and devices for value dequantization."""
    if v_quant.ndim != 3:
        raise ValueError(
            f"v_quant must be 3-dimensional [num_tokens, num_kv_heads, head_dim], "
            f"got ndim={v_quant.ndim} with shape {v_quant.shape}"
        )

    t, h, d = v_quant.shape

    if v_quant.dtype != torch.int8:
        raise TypeError(f"v_quant must be torch.int8, got {v_quant.dtype}")

    if v_scale.ndim != 3 or v_scale.shape != (t, h, 1):
        raise ValueError(
            f"v_scale must have shape [{t}, {h}, 1], got {v_scale.shape}"
        )

    if v_scale.dtype != torch.float16:
        raise TypeError(f"v_scale must be torch.float16, got {v_scale.dtype}")

    if v_scale.device != v_quant.device:
        raise ValueError(
            f"v_scale device ({v_scale.device}) does not match v_quant device ({v_quant.device})"
        )

    if out is not None:
        if out.shape != (t, h, d):
            raise ValueError(
                f"out tensor must have shape [{t}, {h}, {d}], got {out.shape}"
            )
        if out.dtype != torch.float16:
            raise TypeError(f"out tensor must be torch.float16, got {out.dtype}")
        if out.device != v_quant.device:
            raise ValueError(f"out tensor device ({out.device}) does not match input device ({v_quant.device})")


def compute_rope_cos_sin_delta(
    delta_positions: torch.Tensor,
    head_dim: int,
    theta_base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute relative RoPE cos and sin tables for given position displacements.

    According to the RoPE formulation, the rotary frequency for pair index i is:
    theta_i = theta_base^(-2 * i / head_dim) for i in [0, head_dim // 2)

    For displacement Delta p, the rotation angle is:
    angle = Delta p * theta_i

    Args:
        delta_positions: 1D tensor [num_tokens] or scalar/1-element tensor [1].
        head_dim: Full dimension D of each head (must be even).
        theta_base: Base frequency parameter (default 10000.0).
        device: Target torch device.
        dtype: Output floating-point dtype (default torch.float16).

    Returns:
        cos_delta: Tensor of shape [T, D // 2] or [1, D // 2].
        sin_delta: Tensor of shape [T, D // 2] or [1, D // 2].
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    d_half = head_dim // 2
    i_indices = torch.arange(0, d_half, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta_base ** (2.0 * i_indices / head_dim))

    if delta_positions.ndim == 0:
        delta_pos = delta_positions.view(1).to(dtype=torch.float32, device=device)
    else:
        delta_pos = delta_positions.view(-1).to(dtype=torch.float32, device=device)

    # Outer product: [T, 1] * [1, d_half] -> [T, d_half]
    angles = delta_pos[:, None] * inv_freq[None, :]

    cos_delta = torch.cos(angles).to(dtype=dtype)
    sin_delta = torch.sin(angles).to(dtype=dtype)
    return cos_delta, sin_delta


def quantize_symmetric_int8(
    tensor: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token-head INT8 quantization for KV cache states.

    Maps floating-point values in [-max_val, max_val] to integer range [-127, 127].
    The quantization scale is stored per token-head slice with shape [T, H, 1].

    Args:
        tensor: Floating-point tensor of shape [T, H, D].
        eps: Small epsilon preventing division by zero for all-zero slices.

    Returns:
        quant: torch.int8 tensor of shape [T, H, D].
        scale: torch.float16 tensor of shape [T, H, 1].
    """
    if tensor.ndim != 3:
        raise ValueError(f"Input tensor must have shape [T, H, D], got {tensor.shape}")

    abs_max = torch.amax(torch.abs(tensor.to(torch.float32)), dim=-1, keepdim=True)
    scale = torch.clamp(abs_max / 127.0, min=eps).to(torch.float16)

    # Symmetric clamp to [-127, 127]
    scaled = tensor.to(torch.float32) / scale.to(torch.float32)
    quant = torch.clamp(torch.round(scaled), -127.0, 127.0).to(torch.int8)
    return quant, scale


def dequant_and_rope_key_ref(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    cos_delta: torch.Tensor,
    sin_delta: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eager PyTorch reference for fused INT8 dequantization and relative RoPE.

    Implements the HuggingFace Llama/Qwen half-rotated RoPE pairing convention:
    Pair index i in [0, D // 2) with index i + D // 2:
        x1 = q_x1 * scale
        x2 = q_x2 * scale
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

    Accumulation is performed in FP32 to ensure strict numerical ground truth,
    then cast to FP16.

    Args:
        k_quant: INT8 quantized keys [T, H, D].
        k_scale: FP16 scales [T, H, 1].
        cos_delta: FP16 cosine factors [1, D // 2] or [T, D // 2].
        sin_delta: FP16 sine factors [1, D // 2] or [T, D // 2].
        out: Optional pre-allocated output buffer [T, H, D] in FP16.

    Returns:
        k_out: Rotated and dequantized FP16 tensor [T, H, D].
    """
    validate_key_inputs(k_quant, k_scale, cos_delta, sin_delta, out)

    t, h, d = k_quant.shape
    d_half = d // 2

    # Cast INT8 to FP32 and multiply by scale in FP32 for precision
    q_fp32 = k_quant.to(torch.float32)
    scale_fp32 = k_scale.to(torch.float32)
    k_dequant = q_fp32 * scale_fp32

    # Split into first half [0, D/2) and second half [D/2, D)
    x1 = k_dequant[..., :d_half]
    x2 = k_dequant[..., d_half:]

    # Prepare broadcastable cos and sin:
    # If shape is [1, D/2], unsqueeze to [1, 1, D/2]
    # If shape is [T, D/2], unsqueeze to [T, 1, D/2]
    cos_bcast = cos_delta[:, None, :].to(torch.float32)
    sin_bcast = sin_delta[:, None, :].to(torch.float32)

    # Relative rotary phase-shifting
    out1 = x1 * cos_bcast - x2 * sin_bcast
    out2 = x2 * cos_bcast + x1 * sin_bcast

    result = torch.cat([out1, out2], dim=-1).to(torch.float16)

    if out is not None:
        out.copy_(result)
        return out
    return result


def dequant_value_ref(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eager PyTorch reference for Value cache INT8 dequantization.

    Value states do not undergo rotary positional embedding.

    Args:
        v_quant: INT8 quantized values [T, H, D].
        v_scale: FP16 scales [T, H, 1].
        out: Optional pre-allocated output buffer [T, H, D] in FP16.

    Returns:
        v_out: Dequantized FP16 tensor [T, H, D].
    """
    validate_value_inputs(v_quant, v_scale, out)

    # Dequantize with FP32 intermediate product
    result = (v_quant.to(torch.float32) * v_scale.to(torch.float32)).to(torch.float16)

    if out is not None:
        out.copy_(result)
        return out
    return result


# Compiled baseline wrappers for torch.compile
try:
    _compiled_key_ref = torch.compile(dequant_and_rope_key_ref, mode="reduce-overhead")
    _compiled_val_ref = torch.compile(dequant_value_ref, mode="reduce-overhead")
except Exception:
    # Fallback to eager function when torch.compile is unavailable or unsupported
    _compiled_key_ref = dequant_and_rope_key_ref
    _compiled_val_ref = dequant_value_ref


def dequant_and_rope_key_compiled(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    cos_delta: torch.Tensor,
    sin_delta: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compiled baseline wrapper using torch.compile."""
    return _compiled_key_ref(k_quant, k_scale, cos_delta, sin_delta, out)


def dequant_value_compiled(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compiled baseline wrapper using torch.compile."""
    return _compiled_val_ref(v_quant, v_scale, out)
