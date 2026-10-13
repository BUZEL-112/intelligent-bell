"""Production-grade, autotuned OpenAI Triton kernels for SpecCache Phase 1.

This module fuses:
1. Per-token-head INT8 KV-cache dequantization with in-place relative
   Rotary Position Embedding (RoPE) phase-shifting for Key states.
2. High-throughput INT8 dequantization for Value states.

Hardware-Aware Systems Design Principles:
- Fused Single-Pass SRAM Execution: Eliminates intermediate HBM roundtrips
  (INT8 -> FP16 HBM store -> FP16 HBM load -> RoPE -> FP16 HBM store), reducing
  memory bus traffic by ~2.3x.
- Memory Coalescing: Head dimension D is contiguous (stride 1). Threadblocks
  coalesce global memory loads and stores across warps.
- Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) Support:
  The kernel grids over M = T * H token-head vectors, seamlessly scaling across
  any KV head count (e.g., H=1 for MQA, H=8 for Llama 3 / Mistral, H=32 for MHA).
- Uniform & Per-Token Relative Shifting: Stride broadcasting enables a single
  kernel instance to support scalar block shifts [1, D // 2] and per-token
  shifts [T, D // 2] with zero runtime overhead.
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
            triton.Config({"BLOCK_M": 1}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 2}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=3),
        ],
        key=["num_m", "d_half"],
    )
    @triton.jit
    def _fused_dequant_rope_key_kernel(
        k_quant_ptr,
        k_scale_ptr,
        cos_delta_ptr,
        sin_delta_ptr,
        k_out_ptr,
        num_m,
        num_heads,
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
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused INT8 Dequantization + Relative RoPE Phase-Shift Kernel.

        Grids over M = num_tokens * num_heads.
        Each thread block processes BLOCK_M token-head vectors and BLOCK_D head-dim
        elements (where BLOCK_D >= d_half = D // 2).
        """
        pid = tl.program_id(axis=0)
        m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offsets = tl.arange(0, BLOCK_D)

        m_mask = m_offsets < num_m
        d_mask = d_offsets < d_half
        mask_2d = m_mask[:, None] & d_mask[None, :]

        # Decompose linear token-head index into token (t) and head (h)
        t_idx = m_offsets // num_heads
        h_idx = m_offsets % num_heads

        # Key tensor global memory address computation
        # k_quant layout: [T, H, D]
        k_row_offset = t_idx[:, None] * stride_kt + h_idx[:, None] * stride_kh
        d_col_offset = d_offsets[None, :] * stride_kd

        # Pointer to first half: indices [0, D // 2)
        k1_ptrs = k_quant_ptr + k_row_offset + d_col_offset
        # Pointer to second half: indices [D // 2, D)
        k2_ptrs = k1_ptrs + d_half * stride_kd

        # Coalesced global load for quantized INT8 keys
        q1 = tl.load(k1_ptrs, mask=mask_2d, other=0)
        q2 = tl.load(k2_ptrs, mask=mask_2d, other=0)

        # Scale is stored per token-head [T, H, 1]
        scale_ptrs = k_scale_ptr + (t_idx * stride_st + h_idx * stride_sh)
        scale = tl.load(scale_ptrs, mask=m_mask, other=0.0)
        scale_f32 = scale[:, None].to(tl.float32)

        # Dequantize to FP32 in registers to maintain full numerical accuracy
        x1 = q1.to(tl.float32) * scale_f32
        x2 = q2.to(tl.float32) * scale_f32

        # Relative RoPE phase angles
        # If cos_delta has shape [1, D // 2], stride_ct is 0 (broadcasts across all tokens)
        # If cos_delta has shape [T, D // 2], stride_ct is stride(0) (per-token shifts)
        cos_ptrs = cos_delta_ptr + (t_idx[:, None] * stride_ct) + (d_offsets[None, :] * stride_cd)
        sin_ptrs = sin_delta_ptr + (t_idx[:, None] * stride_sit) + (d_offsets[None, :] * stride_sid)

        cos_val = tl.load(cos_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        sin_val = tl.load(sin_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

        # HuggingFace Llama/Qwen Half-Rotated RoPE pairing:
        # out1 = x1 * cos - x2 * sin
        # out2 = x2 * cos + x1 * sin
        out1 = x1 * cos_val - x2 * sin_val
        out2 = x2 * cos_val + x1 * sin_val

        # Convert to FP16 and store directly to destination buffer
        out_row_offset = t_idx[:, None] * stride_ot + h_idx[:, None] * stride_oh
        out_col_offset = d_offsets[None, :] * stride_od

        out1_ptrs = k_out_ptr + out_row_offset + out_col_offset
        out2_ptrs = out1_ptrs + d_half * stride_od

        tl.store(out1_ptrs, out1.to(tl.float16), mask=mask_2d)
        tl.store(out2_ptrs, out2.to(tl.float16), mask=mask_2d)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 2}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=3),
        ],
        key=["num_m", "head_dim"],
    )
    @triton.jit
    def _dequant_value_kernel(
        v_quant_ptr,
        v_scale_ptr,
        v_out_ptr,
        num_m,
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
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused INT8 Dequantization Kernel for Value states."""
        pid = tl.program_id(axis=0)
        m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offsets = tl.arange(0, BLOCK_D)

        m_mask = m_offsets < num_m
        d_mask = d_offsets < head_dim
        mask_2d = m_mask[:, None] & d_mask[None, :]

        t_idx = m_offsets // num_heads
        h_idx = m_offsets % num_heads

        v_ptrs = (
            v_quant_ptr
            + (t_idx[:, None] * stride_vt + h_idx[:, None] * stride_vh)
            + (d_offsets[None, :] * stride_vd)
        )
        q = tl.load(v_ptrs, mask=mask_2d, other=0)

        scale_ptrs = v_scale_ptr + (t_idx * stride_st + h_idx * stride_sh)
        scale = tl.load(scale_ptrs, mask=m_mask, other=0.0)
        scale_f32 = scale[:, None].to(tl.float32)

        out = q.to(tl.float32) * scale_f32

        out_ptrs = (
            v_out_ptr
            + (t_idx[:, None] * stride_ot + h_idx[:, None] * stride_oh)
            + (d_offsets[None, :] * stride_od)
        )
        tl.store(out_ptrs, out.to(tl.float16), mask=mask_2d)


def dequant_and_rope_key(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    cos_delta: torch.Tensor,
    sin_delta: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute production Triton kernel fusing INT8 dequantization with relative RoPE.

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
    num_m = t * h

    if out is None:
        out = torch.empty((t, h, d), dtype=torch.float16, device=k_quant.device)

    # Ensure tensors are contiguous for optimal coalesced memory access
    k_quant_contig = k_quant.contiguous()
    k_scale_contig = k_scale.contiguous()
    cos_delta_contig = cos_delta.contiguous()
    sin_delta_contig = sin_delta.contiguous()

    # Determine token stride for cos and sin:
    # If shape is [1, D // 2], set token stride to 0 for automatic kernel-level broadcasting.
    stride_ct = 0 if cos_delta_contig.shape[0] == 1 else cos_delta_contig.stride(0)
    stride_cd = cos_delta_contig.stride(1)

    stride_sit = 0 if sin_delta_contig.shape[0] == 1 else sin_delta_contig.stride(0)
    stride_sid = sin_delta_contig.stride(1)

    block_d = triton.next_power_of_2(d_half)

    grid = lambda META: (triton.cdiv(num_m, META["BLOCK_M"]),)

    _fused_dequant_rope_key_kernel[grid](
        k_quant_contig,
        k_scale_contig,
        cos_delta_contig,
        sin_delta_contig,
        out,
        num_m,
        h,
        d_half,
        k_quant_contig.stride(0),
        k_quant_contig.stride(1),
        k_quant_contig.stride(2),
        k_scale_contig.stride(0),
        k_scale_contig.stride(1),
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
    num_m = t * h

    if out is None:
        out = torch.empty((t, h, d), dtype=torch.float16, device=v_quant.device)

    v_quant_contig = v_quant.contiguous()
    v_scale_contig = v_scale.contiguous()

    block_d = triton.next_power_of_2(d)
    grid = lambda META: (triton.cdiv(num_m, META["BLOCK_M"]),)

    _dequant_value_kernel[grid](
        v_quant_contig,
        v_scale_contig,
        out,
        num_m,
        h,
        d,
        v_quant_contig.stride(0),
        v_quant_contig.stride(1),
        v_quant_contig.stride(2),
        v_scale_contig.stride(0),
        v_scale_contig.stride(1),
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
    """Unified entrypoint to dequantize both Key and Value caches in one call.

    Key states are relative RoPE phase-shifted, while Value states are dequantized
    directly.
    """
    k_res = dequant_and_rope_key(k_quant, k_scale, cos_delta, sin_delta, out=k_out)
    v_res = dequant_value(v_quant, v_scale, out=v_out)
    return k_res, v_res
