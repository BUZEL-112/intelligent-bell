"""PCIe DMA Transfer and Compute Overlap Benchmark for SpecCache Phase 2.

Evaluates three host-to-device memory transfer and execution modes:
1. Pageable Host Memory (Synchronous): Standard unpinned CPU tensors transferred
   via blocking tensor.to(device), followed by Phase 1 Triton kernel.
2. Pinned Host Memory (Synchronous): Page-locked CPU tensors transferred via
   pinned DMA, followed by Phase 1 Triton kernel.
3. SpecCache Double-Buffered Pipelined Async DMA: Page-locked CPU memory transferred
   asynchronously over a dedicated transfer stream via non_blocking=True and
   overlapped with Phase 1 Triton kernel execution on a compute stream.

PCIe Data Volume Formula (INT8 KV Cache):
Bytes Transferred = (T * H * D * 1)   # k_quant (INT8)
                  + (T * H * 1 * 2)   # k_scale (FP16)
                  + (T * H * D * 1)   # v_quant (INT8)
                  + (T * H * 1 * 2)   # v_scale (FP16)
"""

import argparse
import os
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from speccache.cache.engine import ChunkRoutingDecision, ChunkRoutingStatus, SpecCacheEngine
from speccache.kernels.reference import compute_rope_cos_sin_delta, quantize_kv_int8

try:
    import triton
    import triton.testing
    from speccache.kernels.triton_rope_kv import (
        TRITON_AVAILABLE,
        dequant_value,
        triton_fused_k_dequant_rope_shift,
    )
except ImportError:
    TRITON_AVAILABLE = False


def calculate_pcie_transfer_bytes(num_tokens: int, num_heads: int, head_dim: int) -> int:
    """Calculate exact byte count transferred across the PCIe bus for INT8 KV chunks."""
    k_quant_bytes = num_tokens * num_heads * head_dim * 1
    k_scale_bytes = num_tokens * num_heads * 1 * 2
    v_quant_bytes = num_tokens * num_heads * head_dim * 1
    v_scale_bytes = num_tokens * num_heads * 1 * 2
    return k_quant_bytes + k_scale_bytes + v_quant_bytes + v_scale_bytes


def benchmark_pageable_synchronous(
    engine: SpecCacheEngine,
    decisions: List[ChunkRoutingDecision],
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    warmup: int = 5,
    rep: int = 20,
) -> float:
    """Measure baseline: Unpinned pageable memory transferred synchronously + kernel."""
    chunk_size = engine.chunk_size
    num_chunks = len(decisions)

    # Prepare unpinned pageable CPU tensors
    pageable_k_q = [engine.pool._host_k_quant[d.block_id].clone() for d in decisions]
    pageable_k_s = [engine.pool._host_k_scale[d.block_id].clone() for d in decisions]
    pageable_v_q = [engine.pool._host_v_quant[d.block_id].clone() for d in decisions]
    pageable_v_s = [engine.pool._host_v_scale[d.block_id].clone() for d in decisions]

    out_k = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
    out_v = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)

    def run_iteration():
        for i, dec in enumerate(decisions):
            offset = i * chunk_size
            # Synchronous blocking transfer from pageable host memory
            kq_dev = pageable_k_q[i].to(device, non_blocking=False)
            ks_dev = pageable_k_s[i].to(device, non_blocking=False)
            vq_dev = pageable_v_q[i].to(device, non_blocking=False)
            vs_dev = pageable_v_s[i].to(device, non_blocking=False)

            cos_d, sin_d = compute_rope_cos_sin_delta(
                torch.tensor([dec.delta_p], device=device), head_dim, device=device
            )

            triton_fused_k_dequant_rope_shift(
                kq_dev, ks_dev, cos_d, sin_d, out=out_k[offset : offset + chunk_size]
            )
            dequant_value(
                vq_dev, vs_dev, out=out_v[offset : offset + chunk_size]
            )
        torch.cuda.synchronize(device)

    # Warmup
    for _ in range(warmup):
        run_iteration()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        run_iteration()
    end_event.record()
    torch.cuda.synchronize(device)

    ms = start_event.elapsed_time(end_event) / rep
    return ms


def benchmark_pinned_synchronous(
    engine: SpecCacheEngine,
    decisions: List[ChunkRoutingDecision],
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    warmup: int = 5,
    rep: int = 20,
) -> float:
    """Measure pinned host memory transferred synchronously + kernel."""
    chunk_size = engine.chunk_size
    out_k = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
    out_v = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)

    def run_iteration():
        for i, dec in enumerate(decisions):
            offset = i * chunk_size
            kq_host, ks_host, vq_host, vs_host = engine.pool.read_chunk(dec.block_id)

            # Synchronous transfer from pinned memory
            kq_dev = kq_host.to(device, non_blocking=False)
            ks_dev = ks_host.to(device, non_blocking=False)
            vq_dev = vq_host.to(device, non_blocking=False)
            vs_dev = vs_host.to(device, non_blocking=False)

            cos_d, sin_d = compute_rope_cos_sin_delta(
                torch.tensor([dec.delta_p], device=device), head_dim, device=device
            )

            triton_fused_k_dequant_rope_shift(
                kq_dev, ks_dev, cos_d, sin_d, out=out_k[offset : offset + chunk_size]
            )
            dequant_value(
                vq_dev, vs_dev, out=out_v[offset : offset + chunk_size]
            )
        torch.cuda.synchronize(device)

    # Warmup
    for _ in range(warmup):
        run_iteration()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        run_iteration()
    end_event.record()
    torch.cuda.synchronize(device)

    ms = start_event.elapsed_time(end_event) / rep
    return ms


def benchmark_pipelined_async_dma(
    engine: SpecCacheEngine,
    decisions: List[ChunkRoutingDecision],
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    warmup: int = 5,
    rep: int = 20,
) -> float:
    """Measure SpecCache Double-Buffered Pipelined Async DMA + Triton Kernel."""
    out_k = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
    out_v = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)

    # Warmup
    for _ in range(warmup):
        engine.execute_async_dma_and_rope(decisions, dest_k_out=out_k, dest_v_out=out_v)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        engine.execute_async_dma_and_rope(decisions, dest_k_out=out_k, dest_v_out=out_v)
    end_event.record()
    torch.cuda.synchronize(device)

    ms = start_event.elapsed_time(end_event) / rep
    return ms


def render_phase2_dark_mode_plots(
    token_counts: List[int],
    results: List[Dict[str, float]],
    device_name: str,
    output_path: str,
) -> None:
    """Generate high-resolution dark-mode dual-subplot benchmark visualization."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # High-contrast color palette
    c_pageable = "#ff3366"  # Crimson
    c_pinned = "#ff9900"    # Amber
    c_pipelined = "#00f0ff" # Neon Cyan

    lat_pageable = [r["ms_pageable"] for r in results]
    lat_pinned = [r["ms_pinned"] for r in results]
    lat_pipelined = [r["ms_pipelined"] for r in results]

    speedup_vs_pageable = [r["speedup_vs_pageable"] for r in results]
    speedup_vs_pinned = [r["speedup_vs_pinned"] for r in results]

    bw_pipelined = [r["bw_pipelined"] for r in results]
    bw_pinned = [r["bw_pinned"] for r in results]
    bw_pageable = [r["bw_pageable"] for r in results]

    # Subplot 1: Total Latency vs. Sequence Length
    ax1.plot(token_counts, lat_pageable, "x:", color=c_pageable, linewidth=1.8, markersize=6, label="Pageable (Sync)")
    ax1.plot(token_counts, lat_pinned, "s--", color=c_pinned, linewidth=2.0, markersize=6, label="Pinned (Sync)")
    ax1.plot(token_counts, lat_pipelined, "o-", color=c_pipelined, linewidth=2.5, markersize=7, label="SpecCache Pipelined Async DMA")

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Sequence Length T (Tokens)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax1.set_ylabel("Total Latency (ms)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax1.set_title(f"PCIe DMA + Kernel Latency ({device_name})", fontsize=12, fontweight="bold", color="#ffffff")
    ax1.set_xticks(token_counts)
    ax1.set_xticklabels([str(t) for t in token_counts], rotation=45, color="#e0e0e0")
    ax1.legend(frameon=True, facecolor="#1e1e1e", edgecolor="#333333", loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#444444")

    # Subplot 2: Effective PCIe Bandwidth (GB/s) vs. Sequence Length
    ax2.plot(token_counts, bw_pipelined, "o-", color=c_pipelined, linewidth=2.5, markersize=7, label="Pipelined Async DMA")
    ax2.plot(token_counts, bw_pinned, "s--", color=c_pinned, linewidth=2.0, markersize=6, label="Pinned Memory")
    ax2.plot(token_counts, bw_pageable, "x:", color=c_pageable, linewidth=1.8, markersize=6, label="Pageable Memory")

    # PCIe Gen4 x16 theoretical line (~26 GB/s)
    ax2.axhline(26.0, color="#888888", linestyle="--", linewidth=1.5, label="PCIe Gen4 x16 Bus (~26 GB/s)")

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Sequence Length T (Tokens)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax2.set_ylabel("Effective PCIe Throughput (GB/s)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax2.set_title("PCIe DMA Throughput Scaling", fontsize=12, fontweight="bold", color="#ffffff")
    ax2.set_xticks(token_counts)
    ax2.set_xticklabels([str(t) for t in token_counts], rotation=45, color="#e0e0e0")
    ax2.legend(frameon=True, facecolor="#1e1e1e", edgecolor="#333333", loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#444444")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#121212")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SpecCache Phase 2 PCIe DMA Benchmark")
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--chunk-size", type=int, default=128, help="Token count per chunk")
    parser.add_argument("--num-heads", type=int, default=8, help="KV head count (default 8)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension D (default 128)")
    parser.add_argument("--output-plot", type=str, default="benchmarks/phase2_dma_overlap.png", help="Plot PNG path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target device")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        print("SpecCache Phase 2 PCIe DMA benchmarks require an NVIDIA GPU runtime (e.g. Google Colab).")
        return

    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)

    print("=" * 85)
    print("SpecCache Phase 2: Pinned Memory & Double-Buffered Async DMA Benchmark")
    print(f"Device: {device_name}")
    print(f"Configuration: Chunk Size={args.chunk_size}, Heads={args.num_heads}, Head Dim={args.head_dim}")
    print("=" * 85)

    max_tokens = max(args.tokens)
    max_blocks = max_tokens // args.chunk_size + 16

    engine = SpecCacheEngine(
        chunk_size=args.chunk_size,
        num_kv_heads=args.num_heads,
        head_dim=args.head_dim,
        max_blocks=max_blocks,
        device=device,
    )

    # Pre-populate pinned pool with dummy chunks
    for i in range(max_blocks):
        toks = list(range(i * 1000, i * 1000 + args.chunk_size))
        raw_k = torch.randn((args.chunk_size, args.num_heads, args.head_dim), dtype=torch.float32)
        raw_v = torch.randn((args.chunk_size, args.num_heads, args.head_dim), dtype=torch.float32)
        kq, ks = quantize_kv_int8(raw_k)
        vq, vs = quantize_kv_int8(raw_v)
        engine.register_chunk(toks, cached_position=i * 64, k_quant=kq, k_scale=ks, v_quant=vq, v_scale=vs)

    print("\n| Tokens | Chunks | Pageable (ms) | Pinned (ms) | Pipelined (ms) | Speedup vs Pageable | Effective GB/s |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    results: List[Dict[str, float]] = []

    for t in args.tokens:
        num_chunks = t // args.chunk_size
        decisions: List[ChunkRoutingDecision] = []
        for c in range(num_chunks):
            decisions.append(
                ChunkRoutingDecision(
                    chunk_index=c,
                    target_start_pos=c * args.chunk_size,
                    target_end_pos=(c + 1) * args.chunk_size,
                    token_ids=tuple(range(c * 1000, c * 1000 + args.chunk_size)),
                    status=ChunkRoutingStatus.EXACT_REUSE,
                    block_id=c,
                    cached_position=c * 64,
                    delta_p=c * 64,
                )
            )

        bytes_xfer = calculate_pcie_transfer_bytes(t, args.num_heads, args.head_dim)

        ms_pageable = benchmark_pageable_synchronous(engine, decisions, t, args.num_heads, args.head_dim, device)
        ms_pinned = benchmark_pinned_synchronous(engine, decisions, t, args.num_heads, args.head_dim, device)
        ms_pipelined = benchmark_pipelined_async_dma(engine, decisions, t, args.num_heads, args.head_dim, device)

        bw_pageable = (bytes_xfer / (ms_pageable * 1e-3)) / 1e9
        bw_pinned = (bytes_xfer / (ms_pinned * 1e-3)) / 1e9
        bw_pipelined = (bytes_xfer / (ms_pipelined * 1e-3)) / 1e9

        speedup_pageable = ms_pageable / ms_pipelined
        speedup_pinned = ms_pinned / ms_pipelined

        results.append({
            "tokens": t,
            "chunks": num_chunks,
            "ms_pageable": ms_pageable,
            "ms_pinned": ms_pinned,
            "ms_pipelined": ms_pipelined,
            "bw_pageable": bw_pageable,
            "bw_pinned": bw_pinned,
            "bw_pipelined": bw_pipelined,
            "speedup_vs_pageable": speedup_pageable,
            "speedup_vs_pinned": speedup_pinned,
        })

        print(
            f"| {t:>6} | {num_chunks:>6} | {ms_pageable:>13.2f} | {ms_pinned:>11.2f} | {ms_pipelined:>14.2f} | "
            f"{speedup_pageable:>18.2f}x | {bw_pipelined:>14.2f} |"
        )

    print("=" * 85)
    print(f"Saving Phase 2 DMA benchmark plot to: {args.output_plot}")
    render_phase2_dark_mode_plots(args.tokens, results, device_name, args.output_plot)
    print("Phase 2 benchmark complete.")


if __name__ == "__main__":
    main()
