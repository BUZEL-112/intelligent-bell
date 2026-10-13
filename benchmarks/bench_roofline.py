"""HBM Memory Bandwidth Roofline Benchmark for SpecCache Phase 1.

Evaluates:
1. PyTorch Eager (un-fused baseline: INT8 dequant + rotate_half)
2. torch.compile (mode="max-autotune-no-cudagraphs")
3. SpecCache Fused Triton Kernel (fused 2D-tiled dequant + relative RoPE)

Evaluates two representative LLM architectures:
- Llama-3-8B GQA configuration: H = 8 KV heads, D = 128 head dim
- 70B / MHA configuration: H = 32 KV heads, D = 128 head dim

Evaluates sequence lengths: T in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

Memory Traffic Formula:
Bytes Transferred = (T * H * D * 1)   # k_quant (INT8 read)
                  + (T * H * 2)       # k_scale (FP16 read)
                  + (T * D * 2)       # cos/sin (FP16 read)
                  + (T * H * D * 2)   # k_out   (FP16 write)
"""

import argparse
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import torch

try:
    import triton
    import triton.testing
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    pytorch_compiled_dequant_rope_shift,
    pytorch_eager_dequant_rope_shift,
    quantize_kv_int8,
)

if TRITON_AVAILABLE:
    from speccache.kernels.triton_rope_kv import triton_fused_k_dequant_rope_shift


KNOWN_GPU_PEAK_HBM_GBPS: Dict[str, float] = {
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


def detect_peak_hbm_bandwidth(device_name: str) -> float:
    """Infer peak memory bandwidth in GB/s from GPU device name."""
    for key, bw in KNOWN_GPU_PEAK_HBM_GBPS.items():
        if key.lower() in device_name.lower():
            return bw
    return 1500.0


def calculate_exact_bytes_transferred(t: int, h: int, d: int) -> int:
    """Compute exact bytes transferred per the mathematical specification.

    Bytes = T*H*D*1 (k_quant) + T*H*2 (k_scale) + T*D*2 (cos/sin) + T*H*D*2 (k_out)
    """
    return (t * h * d * 1) + (t * h * 2) + (t * d * 2) + (t * h * d * 2)


def benchmark_single_configuration(
    t: int,
    h: int,
    d: int,
    device: torch.device,
) -> Dict[str, float]:
    """Profile Eager, torch.compile, and Triton kernel using triton.testing.do_bench."""
    torch.manual_seed(42)

    raw_k = torch.randn((t, h, d), dtype=torch.float32, device=device)
    k_quant, k_scale = quantize_kv_int8(raw_k)

    deltas = torch.randint(1, 100, (t,), device=device)
    cos_delta, sin_delta = compute_rope_cos_sin_delta(deltas, d, device=device)

    # Pre-allocate output buffers to isolate computation from memory allocation
    out_eager = torch.empty((t, h, d), dtype=torch.float16, device=device)
    out_compiled = torch.empty((t, h, d), dtype=torch.float16, device=device)
    out_triton = torch.empty((t, h, d), dtype=torch.float16, device=device)

    # Warmup torch.compile
    pytorch_compiled_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta, out=out_compiled)

    # Warmup Triton kernel
    triton_fused_k_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta, out=out_triton)

    # Benchmark using warmup=25, rep=100
    ms_eager = triton.testing.do_bench(
        lambda: pytorch_eager_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta, out=out_eager),
        warmup=25,
        rep=100,
    )
    ms_compiled = triton.testing.do_bench(
        lambda: pytorch_compiled_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta, out=out_compiled),
        warmup=25,
        rep=100,
    )
    ms_triton = triton.testing.do_bench(
        lambda: triton_fused_k_dequant_rope_shift(k_quant, k_scale, cos_delta, sin_delta, out=out_triton),
        warmup=25,
        rep=100,
    )

    exact_bytes = calculate_exact_bytes_transferred(t, h, d)

    # Latency in microseconds (us)
    us_eager = ms_eager * 1000.0
    us_compiled = ms_compiled * 1000.0
    us_triton = ms_triton * 1000.0

    # Effective Memory Bandwidth (GB/s) = Bytes / (seconds * 10^9)
    bw_eager = (exact_bytes / (ms_eager * 1e-3)) / 1e9
    bw_compiled = (exact_bytes / (ms_compiled * 1e-3)) / 1e9
    bw_triton = (exact_bytes / (ms_triton * 1e-3)) / 1e9

    speedup_vs_eager = ms_eager / ms_triton
    speedup_vs_compile = ms_compiled / ms_triton

    return {
        "tokens": t,
        "us_eager": us_eager,
        "us_compiled": us_compiled,
        "us_triton": us_triton,
        "bw_eager": bw_eager,
        "bw_compiled": bw_compiled,
        "bw_triton": bw_triton,
        "speedup_vs_eager": speedup_vs_eager,
        "speedup_vs_compile": speedup_vs_compile,
    }


def render_dark_mode_roofline_plots(
    token_counts: List[int],
    results_gqa: List[Dict[str, float]],
    results_mha: List[Dict[str, float]],
    peak_bw_gbps: float,
    device_name: str,
    output_path: str,
) -> None:
    """Save high-resolution dark-mode dual-subplot figure."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Styling colors
    color_gqa_triton = "#00f0ff"    # Vibrant Cyan
    color_mha_triton = "#39ff14"    # Neon Green
    color_compiled = "#ff9900"      # Amber
    color_eager = "#ff3366"         # Crimson
    color_peak = "#888888"          # Neutral Gray

    # Subplot 1: Speedup vs. Sequence Length
    ax1.plot(
        token_counts,
        [r["speedup_vs_eager"] for r in results_gqa],
        "o-",
        color=color_gqa_triton,
        linewidth=2.5,
        markersize=6,
        label="Triton Fused vs Eager (GQA H=8)",
    )
    ax1.plot(
        token_counts,
        [r["speedup_vs_eager"] for r in results_mha],
        "s-",
        color=color_mha_triton,
        linewidth=2.5,
        markersize=6,
        label="Triton Fused vs Eager (MHA H=32)",
    )
    ax1.plot(
        token_counts,
        [r["speedup_vs_compile"] for r in results_gqa],
        "^--",
        color=color_compiled,
        linewidth=1.8,
        markersize=5,
        label="Triton vs torch.compile (GQA H=8)",
    )
    ax1.axhline(1.0, color="#555555", linestyle=":", linewidth=1.2, label="1.0x Baseline")

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Sequence Length T (Tokens)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax1.set_ylabel("Speedup Multiplier (x)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax1.set_title("Relative Speedup vs Sequence Length", fontsize=12, fontweight="bold", color="#ffffff")
    ax1.set_xticks(token_counts)
    ax1.set_xticklabels([str(t) for t in token_counts], rotation=45, color="#e0e0e0")
    ax1.legend(frameon=True, facecolor="#1e1e1e", edgecolor="#333333", loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#444444")

    # Subplot 2: Effective Memory Bandwidth (GB/s) vs. Sequence Length
    ax2.plot(
        token_counts,
        [r["bw_triton"] for r in results_gqa],
        "o-",
        color=color_gqa_triton,
        linewidth=2.5,
        markersize=6,
        label="Triton Fused (GQA H=8)",
    )
    ax2.plot(
        token_counts,
        [r["bw_triton"] for r in results_mha],
        "s-",
        color=color_mha_triton,
        linewidth=2.5,
        markersize=6,
        label="Triton Fused (MHA H=32)",
    )
    ax2.plot(
        token_counts,
        [r["bw_compiled"] for r in results_gqa],
        "^--",
        color=color_compiled,
        linewidth=1.8,
        markersize=5,
        label="torch.compile (GQA H=8)",
    )
    ax2.plot(
        token_counts,
        [r["bw_eager"] for r in results_gqa],
        "x:",
        color=color_eager,
        linewidth=1.6,
        markersize=5,
        label="PyTorch Eager (GQA H=8)",
    )
    ax2.axhline(
        peak_bw_gbps,
        color=color_peak,
        linestyle="--",
        linewidth=1.5,
        label=f"Peak HBM Roofline ({peak_bw_gbps:.0f} GB/s)",
    )

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Sequence Length T (Tokens)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax2.set_ylabel("Effective Bandwidth (GB/s)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax2.set_title(f"Memory Bandwidth Scaling ({device_name})", fontsize=12, fontweight="bold", color="#ffffff")
    ax2.set_xticks(token_counts)
    ax2.set_xticklabels([str(t) for t in token_counts], rotation=45, color="#e0e0e0")
    ax2.legend(frameon=True, facecolor="#1e1e1e", edgecolor="#333333", loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#444444")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#121212")
    plt.close(fig)


def print_markdown_table(
    config_name: str,
    results: List[Dict[str, float]],
    peak_bw: float,
) -> None:
    """Print clean formatted Markdown table to stdout."""
    print(f"\n### Benchmark Results: {config_name}")
    print("| Sequence Length (T) | Eager (us) | Compile (us) | Triton (us) | Speedup vs Eager | Effective GB/s | % Peak HBM |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        pct_peak = (r["bw_triton"] / peak_bw) * 100.0
        print(
            f"| {r['tokens']:>19} "
            f"| {r['us_eager']:>10.1f} "
            f"| {r['us_compiled']:>12.1f} "
            f"| {r['us_triton']:>11.1f} "
            f"| {r['speedup_vs_eager']:>15.2f}x "
            f"| {r['bw_triton']:>14.1f} "
            f"| {pct_peak:>9.1f}% |"
        )


def main():
    parser = argparse.ArgumentParser(description="SpecCache Phase 1 Roofline Benchmark")
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--peak-bw", type=float, default=None, help="Theoretical peak GPU memory BW in GB/s")
    parser.add_argument("--output-plot", type=str, default="benchmarks/phase1_roofline.png", help="Output PNG path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        print("Please run this benchmark on an NVIDIA GPU runtime (e.g. Google Colab with T4/L4/A100).")
        return

    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    peak_bw = args.peak_bw if args.peak_bw is not None else detect_peak_hbm_bandwidth(device_name)

    print("=" * 80)
    print("SpecCache Phase 1: Fused INT8 Dequantization + Relative RoPE Roofline Benchmark")
    print(f"Device: {device_name} (Theoretical Peak HBM: {peak_bw:.1f} GB/s)")
    print("=" * 80)

    # 1. Llama-3-8B GQA Config (H=8, D=128)
    print("\nRunning Llama-3-8B GQA configuration (H=8, D=128)...")
    results_gqa: List[Dict[str, float]] = []
    for t in args.tokens:
        res = benchmark_single_configuration(t, h=8, d=128, device=device)
        results_gqa.append(res)
    print_markdown_table("Llama-3-8B GQA (H=8, D=128)", results_gqa, peak_bw)

    # 2. 70B / MHA Config (H=32, D=128)
    print("\nRunning 70B / MHA configuration (H=32, D=128)...")
    results_mha: List[Dict[str, float]] = []
    for t in args.tokens:
        res = benchmark_single_configuration(t, h=32, d=128, device=device)
        results_mha.append(res)
    print_markdown_table("70B / MHA (H=32, D=128)", results_mha, peak_bw)

    # 3. Generate high-resolution dark-mode dual-subplot figure
    print(f"\nSaving dark-mode roofline plot to: {args.output_plot}")
    render_dark_mode_roofline_plots(args.tokens, results_gqa, results_mha, peak_bw, device_name, args.output_plot)
    print("Roofline benchmark complete.")


if __name__ == "__main__":
    main()
