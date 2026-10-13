"""HBM Roofline and Latency Benchmark Suite for SpecCache Phase 1.

This benchmark compares:
1. Eager PyTorch (un-fused baseline: INT8 dequant to FP16, then separate RoPE)
2. torch.compile (PyTorch 2.x JIT graph compilation)
3. SpecCache Fused Triton Kernel (single-pass SRAM dequant + relative RoPE)

Roofline Modeling & Microarchitectural Analysis:
- Algorithm Arithmetic Intensity:
  For each head-dim pair (2 elements):
  - Read: 2 bytes INT8 + 2/D bytes scale + 4/H bytes cos/sin
  - Write: 4 bytes FP16
  - Total Memory: ~6.03 bytes per pair (~3.015 bytes/element)
  - Floating Point Operations: 2 muls (dequant) + 4 ops (RoPE) = 6 FLOPs
  - Arithmetic Intensity = 6 FLOPs / 6.03 Bytes = ~1.0 FLOP/Byte.
- Because 1.0 FLOP/Byte is far below modern GPU ridge points (typically 10 to 20
  FLOP/Byte on A100/H100/L4), this workload is strictly memory-bandwidth bound.
- Un-fused PyTorch incurs an intermediate FP16 write and FP16 re-read in HBM,
  consuming ~7 bytes/element.
- The SpecCache fused kernel consumes ~3 bytes/element, yielding a theoretical
  2.33x reduction in memory traffic and proportional latency speedup.
"""

import argparse
from typing import Dict, List, Optional, Tuple
import torch

try:
    import triton
    import triton.testing
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    dequant_and_rope_key_ref,
    quantize_symmetric_int8,
)

if TRITON_AVAILABLE:
    from speccache.kernels.triton_rope_kv import dequant_and_rope_key


# Known GPU peak HBM bandwidths in GB/s
GPU_PEAK_HBM_BANDWIDTH_GBPS: Dict[str, float] = {
    "A100-SXM4-80GB": 2039.0,
    "A100-SXM4-40GB": 1555.0,
    "A100-PCIE-40GB": 1555.0,
    "H100-SXM": 3350.0,
    "H100-PCIE": 2000.0,
    "L4": 300.0,
    "T4": 320.0,
    "V100-SXM2-32GB": 900.0,
    "V100-PCIE-16GB": 900.0,
    "RTX 4090": 1008.0,
    "RTX 3090": 936.0,
}


def detect_peak_bandwidth(device_name: str) -> float:
    """Infer theoretical peak memory bandwidth from device name or return default."""
    for key, bw in GPU_PEAK_HBM_BANDWIDTH_GBPS.items():
        if key.lower() in device_name.lower():
            return bw
    # Default to modern high-end GPU bandwidth if unrecognized
    return 1500.0


def calculate_data_volume_bytes(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    is_fused: bool,
) -> int:
    """Calculate theoretical global memory transfer volume in bytes.

    For fused kernel:
    - Read k_quant: T * H * D * 1 byte (INT8)
    - Read k_scale: T * H * 1 * 2 bytes (FP16)
    - Read cos/sin: 2 * T * (D // 2) * 2 bytes (FP16)
    - Write k_out:  T * H * D * 2 bytes (FP16)

    For un-fused baseline (with intermediate HBM buffer):
    - Extra intermediate write of dequantized FP16: T * H * D * 2 bytes
    - Extra intermediate read of dequantized FP16:  T * H * D * 2 bytes
    """
    read_quant = num_tokens * num_heads * head_dim * 1
    read_scale = num_tokens * num_heads * 1 * 2
    read_rope = 2 * num_tokens * (head_dim // 2) * 2
    write_out = num_tokens * num_heads * head_dim * 2

    base_fused = read_quant + read_scale + read_rope + write_out
    if is_fused:
        return base_fused

    # Un-fused adds intermediate FP16 store and load
    intermediate_traffic = 2 * (num_tokens * num_heads * head_dim * 2)
    return base_fused + intermediate_traffic


def run_benchmark_for_shape(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> Dict[str, float]:
    """Benchmark eager, compiled, and Triton implementations for a specific shape."""
    # Seed for deterministic tensors
    torch.manual_seed(42)

    raw_k = torch.randn((num_tokens, num_heads, head_dim), dtype=torch.float32, device=device)
    k_quant, k_scale = quantize_symmetric_int8(raw_k)

    deltas = torch.randint(1, 100, (num_tokens,), device=device)
    cos_delta, sin_delta = compute_rope_cos_sin_delta(deltas, head_dim, device=device)

    # Pre-allocate output buffer to isolate kernel computation from memory allocation
    k_out_eager = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
    k_out_compiled = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
    k_out_triton = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)

    # Compiled baseline setup
    compiled_fn = torch.compile(dequant_and_rope_key_ref, mode="reduce-overhead")
    # Warmup torch.compile
    compiled_fn(k_quant, k_scale, cos_delta, sin_delta, out=k_out_compiled)

    # Warmup Triton kernel
    dequant_and_rope_key(k_quant, k_scale, cos_delta, sin_delta, out=k_out_triton)

    # 1. Eager PyTorch timing
    ms_eager = triton.testing.do_bench(
        lambda: dequant_and_rope_key_ref(k_quant, k_scale, cos_delta, sin_delta, out=k_out_eager)
    )

    # 2. torch.compile timing
    ms_compiled = triton.testing.do_bench(
        lambda: compiled_fn(k_quant, k_scale, cos_delta, sin_delta, out=k_out_compiled)
    )

    # 3. SpecCache Triton timing
    ms_triton = triton.testing.do_bench(
        lambda: dequant_and_rope_key(k_quant, k_scale, cos_delta, sin_delta, out=k_out_triton)
    )

    # Theoretical minimum bytes for the fused operation
    min_bytes = calculate_data_volume_bytes(num_tokens, num_heads, head_dim, is_fused=True)

    # Effective Bandwidth = Minimum Required Bytes / Measured Latency
    bw_eager = (min_bytes / (ms_eager * 1e-3)) / 1e9
    bw_compiled = (min_bytes / (ms_compiled * 1e-3)) / 1e9
    bw_triton = (min_bytes / (ms_triton * 1e-3)) / 1e9

    return {
        "ms_eager": ms_eager,
        "ms_compiled": ms_compiled,
        "ms_triton": ms_triton,
        "bw_eager": bw_eager,
        "bw_compiled": bw_compiled,
        "bw_triton": bw_triton,
        "speedup_vs_eager": ms_eager / ms_triton,
        "speedup_vs_compiled": ms_compiled / ms_triton,
    }


def generate_roofline_plots(
    token_counts: List[int],
    results: List[Dict[str, float]],
    peak_bw_gbps: float,
    device_name: str,
    output_path: str,
) -> None:
    """Generate publication-quality roofline and speedup charts."""
    import matplotlib.pyplot as plt

    bw_eager = [r["bw_eager"] for r in results]
    bw_compiled = [r["bw_compiled"] for r in results]
    bw_triton = [r["bw_triton"] for r in results]
    speedup_eager = [r["speedup_vs_eager"] for r in results]
    speedup_compiled = [r["speedup_vs_compiled"] for r in results]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Plot 1: Effective Memory Bandwidth (GB/s) vs Sequence Length
    ax1.plot(token_counts, bw_triton, "o-", color="#1f77b4", linewidth=2.5, label="SpecCache Fused Triton")
    ax1.plot(token_counts, bw_compiled, "s--", color="#2ca02c", linewidth=2.0, label="torch.compile")
    ax1.plot(token_counts, bw_eager, "^:", color="#d62728", linewidth=1.8, label="Eager PyTorch (Un-fused)")

    ax1.axhline(
        peak_bw_gbps,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.5,
        label=f"Hardware Peak Roofline ({peak_bw_gbps:.0f} GB/s)",
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Sequence Length (Tokens)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Effective Memory Bandwidth (GB/s)", fontsize=11, fontweight="bold")
    ax1.set_title(f"HBM Memory Bandwidth Scaling ({device_name})", fontsize=12, fontweight="bold")
    ax1.set_xticks(token_counts)
    ax1.set_xticklabels([str(t) for t in token_counts], rotation=45)
    ax1.legend(frameon=True, loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Speedup Ratio vs Sequence Length
    ax2.plot(token_counts, speedup_eager, "o-", color="#1f77b4", linewidth=2.5, label="Speedup vs Eager PyTorch")
    ax2.plot(token_counts, speedup_compiled, "s--", color="#ff7f0e", linewidth=2.0, label="Speedup vs torch.compile")
    ax2.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.2)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Sequence Length (Tokens)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Speedup Multiplier (x)", fontsize=11, fontweight="bold")
    ax2.set_title("Relative Acceleration across Sequence Lengths", fontsize=12, fontweight="bold")
    ax2.set_xticks(token_counts)
    ax2.set_xticklabels([str(t) for t in token_counts], rotation=45)
    ax2.legend(frameon=True, loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SpecCache Phase 1 HBM Roofline Benchmark")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of KV heads (default 8 for GQA)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (default 128 for Llama 3)")
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="Sequence lengths to evaluate",
    )
    parser.add_argument("--peak-bw", type=float, default=None, help="Theoretical peak GPU HBM BW in GB/s")
    parser.add_argument("--output-plot", type=str, default="benchmarks/roofline_results.png", help="Path for PNG plot")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        print("SpecCache Triton benchmarks require an NVIDIA GPU runtime (e.g. Google Colab or cloud GPU).")
        return

    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    peak_bw = args.peak_bw if args.peak_bw is not None else detect_peak_bandwidth(device_name)

    print("=" * 80)
    print("SpecCache Phase 1: Hardware-Aware Fused RoPE KV-Cache Roofline Benchmark")
    print(f"Device: {device_name}")
    print(f"Hardware Peak Memory Bandwidth: {peak_bw:.1f} GB/s")
    print(f"Configuration: Heads={args.num_heads}, Head Dim={args.head_dim}")
    print("=" * 80)
    header = (
        f"{'Tokens':>8} | {'Eager (ms)':>10} {'Compile (ms)':>12} {'Triton (ms)':>11} | "
        f"{'Triton GB/s':>11} {'% Peak':>7} | {'Speedup (Eager)':>15}"
    )
    print(header)
    print("-" * len(header))

    results: List[Dict[str, float]] = []
    for t in args.tokens:
        res = run_benchmark_for_shape(t, args.num_heads, args.head_dim, device)
        results.append(res)
        pct_peak = (res["bw_triton"] / peak_bw) * 100.0

        print(
            f"{t:>8} | {res['ms_eager']:>10.3f} {res['ms_compiled']:>12.3f} {res['ms_triton']:>11.3f} | "
            f"{res['bw_triton']:>11.1f} {pct_peak:>6.1f}% | {res['speedup_vs_eager']:>14.2f}x"
        )

    print("=" * 80)
    print(f"Generating roofline plot -> {args.output_plot}")
    generate_roofline_plots(args.tokens, results, peak_bw, device_name, args.output_plot)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
