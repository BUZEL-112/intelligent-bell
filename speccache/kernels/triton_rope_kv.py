"""Production-grade, autotuned OpenAI Triton kernels for SpecCache Phase 1.

Fuses:
1. Per-token-head INT8 KV-cache dequantization with in-place relative
   Rotary Position Embedding (RoPE) phase-shifting for Key states.
2. High-throughput INT8 dequantization for Value states.

Hardware-Aware Systems Design Principles:
- 2D Grid Tiling (BLOCK_T, BLOCK_H): Tiles over both token chunks and KV heads
  in a 2D grid (cdiv(T, BLOCK_T), cdiv(H, BLOCK_H)). This ensures small token
  chunks (e.g., T=128 with H=8) launch sufficient threadblocks to saturate SMs
  on NVIDIA Ada Lovelace, Ampere, and Hopper architectures.
- SRAM Register Reuse: RoPE factors cos and sin depend solely on token position
  and head dimension, remaining invariant across head indices. By tiling over
  BLOCK_H heads in each block, cos and sin factors are loaded once into registers
  and reused across all BLOCK_H heads in SRAM, eliminating redundant HBM reads.
- Robust Boundary Masking: 2D/3D masking guards against arbitrary non-power-of-two
  sequence lengths (e.g., T=17, T=511) to guarantee memory safety.
- Numerical Stability: INT8 keys and FP16 scales are cast to FP32 in registers
  for dequantization and rotary matrix multiplication, then cast to FP16 before
  storing to k_out.
"""

from typing import Optional, Tuple
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from speccache.kernels.reference import (
    validate_key_inputs,
    validate_value_inputs,
)


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            # Balanced 2D tiles for small to medium token counts
            triton.Config({"BLOCK_T": 1, "BLOCK_H": 1}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_T": 1, "BLOCK_H": 2}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_T": 1, "BLOCK_H": 4}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_T": 4, "BLOCK_H": 1}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_T": 4, "BLOCK_H": 2}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_T": 4, "BLOCK_H": 4}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_T": 8, "BLOCK_H": 1}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_T": 8, "BLOCK_H": 2}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_T": 8, "BLOCK_H": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_T": 8, "BLOCK_H": 8}, num_warps=8, num_stages=3),
            # Large token tiles for long-context prefill / prompt caching
            triton.Config({"BLOCK_T": 16, "BLOCK_H": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_T": 16, "BLOCK_H": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_T": 16, "BLOCK_H": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_T": 32, "BLOCK_H": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_T": 32, "BLOCK_H": 2}, num_warps=8, num_stages=4),
        ],
        key=["num_tokens", "num_heads", "head_dim"],
    )
    @triton.jit
    def fused_k_dequant_rope_shift_kernel(
        k_quant_ptr,
        k_scale_ptr,
        cos_delta_ptr,
        sin_delta_ptr,
        k_out_ptr,
        num_tokens,
        num_heads,
        head_dim,
        d_half,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_st,
        stride_sh,
        stride_ct,
        stride_cd,
        stride_sit,
        stride_sid,
        stride_ot,
        stride_oh,
        stride_od,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused 2D-Tiled INT8 Dequantization + Relative RoPE Phase-Shift Kernel.

        Grid layout:
            program_id(0): Token chunk index (cdiv(num_tokens, BLOCK_T))
            program_id(1): Head chunk index (cdiv(num_heads, BLOCK_H))

        Each thread block processes a 2D tile (BLOCK_T, BLOCK_H, BLOCK_D) where
        BLOCK_D >= d_half = head_dim // 2.
        """
        pid_t = tl.program_id(axis=0)
        pid_h = tl.program_id(axis=1)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)

        mask_t = offs_t < num_tokens
        mask_h = offs_h < num_heads
        mask_d = offs_d < d_half

        # 2D mask for token-dim factors
        mask_td = mask_t[:, None] & mask_d[None, :]

        # Load relative RoPE cos and sin factors for BLOCK_T tokens
        # Reused in SRAM across all BLOCK_H heads
        cos_ptrs = cos_delta_ptr + (offs_t[:, None] * stride_ct) + (offs_d[None, :] * stride_cd)
        sin_ptrs = sin_delta_ptr + (offs_t[:, None] * stride_sit) + (offs_d[None, :] * stride_sid)

        cos_val = tl.load(cos_ptrs, mask=mask_td, other=0.0).to(tl.float32)
        sin_val = tl.load(sin_ptrs, mask=mask_td, other=0.0).to(tl.float32)

        # Broadcast across head dimension BLOCK_H: shape [BLOCK_T, 1, BLOCK_D]
        cos_3d = cos_val[:, None, :]
        sin_3d = sin_val[:, None, :]

        # 3D mask for loading keys: shape [BLOCK_T, BLOCK_H, BLOCK_D]
        mask_3d = mask_t[:, None, None] & mask_h[None, :, None] & mask_d[None, None, :]

        # Pointer arithmetic for first half [0, D // 2)
        k_base = (
            offs_t[:, None, None] * stride_kt
            + offs_h[None, :, None] * stride_kh
            + offs_d[None, None, :] * stride_kd
        )
        k1_ptrs = k_quant_ptr + k_base
        # Pointer arithmetic for second half [D // 2, D)
        k2_ptrs = k1_ptrs + d_half * stride_kd

        # Load quantized INT8 key halves
        q1 = tl.load(k1_ptrs, mask=mask_3d, other=0)
        q2 = tl.load(k2_ptrs, mask=mask_3d, other=0)

        # Load scales: k_scale shape [T, H, 1]
        scale_mask = mask_t[:, None, None] & mask_h[None, :, None]
        scale_ptrs = (
            k_scale_ptr
            + offs_t[:, None, None] * stride_st
            + offs_h[None, :, None] * stride_sh
        )
        scale = tl.load(scale_ptrs, mask=scale_mask, other=0.0).to(tl.float32)

        # Dequantize with FP32 precision
        x1 = q1.to(tl.float32) * scale
        x2 = q2.to(tl.float32) * scale

        # Relative RoPE phase-shifting
        # out1 = x1 * cos - x2 * sin
        # out2 = x2 * cos + x1 * sin
        out1 = x1 * cos_3d - x2 * sin_3d
        out2 = x2 * cos_3d + x1 * sin_3d

        # Store to k_out in FP16
        out_base = (
            offs_t[:, None, None] * stride_ot
            + offs_h[None, :, None] * stride_oh
            + offs_d[None, None, :] * stride_od
        )
        out1_ptrs = k_out_ptr + out_base
        out2_ptrs = out1_ptrs + d_half * stride_od

        tl.store(out1_ptrs, out1.to(tl.float16), mask=mask_3d)
        tl.store(out2_ptrs, out2.to(tl.float16), mask=mask_3d)


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_T": 1, "BLOCK_H": 1}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_T": 4, "BLOCK_H": 2}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_T": 8, "BLOCK_H": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_T": 16, "BLOCK_H": 4}, num_warps=8, num_stages=3),
        ],
        key=["num_tokens", "num_heads", "head_dim"],
    )
    @triton.jit
    def _dequant_value_kernel_2d(
        v_quant_ptr,
        v_scale_ptr,
        v_out_ptr,
        num_tokens,
        num_heads,
        head_dim,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_st,
        stride_sh,
        stride_ot,
        stride_oh,
        stride_od,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused INT8 Dequantization Kernel for Value states."""
        pid_t = tl.program_id(axis=0)
        pid_h = tl.program_id(axis=1)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)

        mask_3d = (
            (offs_t[:, None, None] < num_tokens)
            & (offs_h[None, :, None] < num_heads)
            & (offs_d[None, None, :] < head_dim)
        )

        v_ptrs = (
            v_quant_ptr
            + offs_t[:, None, None] * stride_vt
            + offs_h[None, :, None] * stride_vh
            + offs_d[None, None, :] * stride_vd
        )
        q = tl.load(v_ptrs, mask=mask_3d, other=0)

        scale_mask = (offs_t[:, None, None] < num_tokens) & (offs_h[None, :, None] < num_heads)
        scale_ptrs = (
            v_scale_ptr
            + offs_t[:, None, None] * stride_st
            + offs_h[None, :, None] * stride_sh
        )
        scale = tl.load(scale_ptrs, mask=scale_mask, other=0.0).to(tl.float32)

        out = q.to(tl.float32) * scale

        out_ptrs = (
            v_out_ptr
            + offs_t[:, None, None] * stride_ot
            + offs_h[None, :, None] * stride_oh
            + offs_d[None, None, :] * stride_od
        )
        tl.store(out_ptrs, out.to(tl.float16), mask=mask_3d)


def triton_fused_k_dequant_rope_shift(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    cos_delta: torch.Tensor,
    sin_delta: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute production 2D-tiled Triton kernel fusing INT8 dequant with relative RoPE.

    Validates tensor shapes, dtypes, contiguous layouts, and CUDA device compatibility.

    Args:
        k_quant: INT8 quantized keys of shape [num_tokens, num_kv_heads, head_dim].
        k_scale: FP16 scale of shape [num_tokens, num_kv_heads, 1].
        cos_delta: FP16 cosine factors of shape [1, head_dim // 2] or [num_tokens, head_dim // 2].
        sin_delta: FP16 sine factors of shape [1, head_dim // 2] or [num_tokens, head_dim // 2].
        out: Optional pre-allocated FP16 buffer of shape [num_tokens, num_kv_heads, head_dim].

    Returns:
        k_out: FP16 tensor of shape [num_tokens, num_kv_heads, head_dim].
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("OpenAI Triton is not installed or available in this environment.")

    validate_key_inputs(k_quant, k_scale, cos_delta, sin_delta, out)

    if not k_quant.is_cuda:
        raise RuntimeError(
            f"SpecCache Triton kernels require an NVIDIA CUDA device, got device: {k_quant.device}."
        )

    t, h, d = k_quant.shape
    d_half = d // 2

    if out is None:
        out = torch.empty((t, h, d), dtype=torch.float16, device=k_quant.device)

    # Ensure tensors are contiguous for optimal coalesced memory access
    k_quant_c = k_quant.contiguous()
    k_scale_c = k_scale.contiguous()
    cos_delta_c = cos_delta.contiguous()
    sin_delta_c = sin_delta.contiguous()

    # Determine token stride for cos and sin:
    # If shape is [1, D // 2], stride_ct is 0 for kernel-level token broadcasting.
    stride_ct = 0 if cos_delta_c.shape[0] == 1 else cos_delta_c.stride(0)
    stride_cd = cos_delta_c.stride(1)

    stride_sit = 0 if sin_delta_c.shape[0] == 1 else sin_delta_c.stride(0)
    stride_sid = sin_delta_c.stride(1)

    block_d = triton.next_power_of_2(d_half)

    # 2D Grid launch: (cdiv(T, BLOCK_T), cdiv(H, BLOCK_H))
    grid = lambda META: (
        triton.cdiv(t, META["BLOCK_T"]),
        triton.cdiv(h, META["BLOCK_H"]),
    )

    fused_k_dequant_rope_shift_kernel[grid](
        k_quant_c,
        k_scale_c,
        cos_delta_c,
        sin_delta_c,
        out,
        t,
        h,
        d,
        d_half,
        k_quant_c.stride(0),
        k_quant_c.stride(1),
        k_quant_c.stride(2),
        k_scale_c.stride(0),
        k_scale_c.stride(1),
        stride_ct,
        stride_cd,
        stride_sit,
        stride_sid,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=block_d,
    )

    return out


# Alias for backward compatibility
dequant_and_rope_key = triton_fused_k_dequant_rope_shift


def dequant_value(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute production Triton kernel for INT8 Value cache dequantization.

    Args:
        v_quant: INT8 quantized values of shape [num_tokens, num_kv_heads, head_dim].
        v_scale: FP16 scale of shape [num_tokens, num_kv_heads, 1].
        out: Optional pre-allocated FP16 buffer of shape [num_tokens, num_kv_heads, head_dim].

    Returns:
        v_out: FP16 tensor of shape [num_tokens, num_kv_heads, head_dim].
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("OpenAI Triton is not installed or available in this environment.")

    validate_value_inputs(v_quant, v_scale, out)

    if not v_quant.is_cuda:
        raise RuntimeError(
            f"SpecCache Triton kernels require an NVIDIA CUDA device, got device: {v_quant.device}."
        )

    t, h, d = v_quant.shape

    if out is None:
        out = torch.empty((t, h, d), dtype=torch.float16, device=v_quant.device)

    v_quant_c = v_quant.contiguous()
    v_scale_c = v_scale.contiguous()

    block_d = triton.next_power_of_2(d)
    grid = lambda META: (
        triton.cdiv(t, META["BLOCK_T"]),
        triton.cdiv(h, META["BLOCK_H"]),
    )

    _dequant_value_kernel_2d[grid](
        v_quant_c,
        v_scale_c,
        out,
        t,
        h,
        d,
        v_quant_c.stride(0),
        v_quant_c.stride(1),
        v_quant_c.stride(2),
        v_scale_c.stride(0),
        v_scale_c.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=block_d,
    )

    return out


def dequant_kv_cache(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    cos_delta: torch.Tensor,
    sin_delta: torch.Tensor,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified entrypoint to dequantize both Key and Value caches in one call."""
    k_res = triton_fused_k_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta, out=k_out)
    v_res = dequant_value(v_quant, v_scale, out=v_out)
    return k_res, v_res
