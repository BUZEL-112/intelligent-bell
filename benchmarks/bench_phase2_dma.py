"""Comprehensive PCIe DMA Transfer & Index Latency Benchmark for SpecCache Phase 2.

Evaluates:
1. End-to-End Host-to-GPU KV Retrieval + RoPE Reconstruction across 4 baselines:
   - Baseline 1 (Naive FP16 Pageable): Uncompressed FP16 KV from pageable CPU RAM + Eager PyTorch RoPE shift.
   - Baseline 2 (INT8 Pageable + Triton): INT8 KV from pageable CPU RAM + Phase 1 Triton kernel (single stream).
   - Baseline 3 (INT8 Pinned + Triton Single-Stream): INT8 KV from pinned pool + Phase 1 Triton kernel sequentially.
   - Baseline 4 (SpecCache Full: INT8 Pinned + Dual-Stream Async Pipelined DMA + Triton): Overlapped PCIe DMA + Triton kernel.
2. Index Lookup Latency (us) across index sizes N in [1_000, 10_000, 50_000] chunks:
   - ChunkRadixTree (Tier 1 exact match)
   - SemanticHNSWIndex (Tier 2 FAISS HNSW cosine similarity search)
3. Generates high-resolution dark-mode dual-panel plot -> benchmarks/phase2_dma_pipeline.png.
"""

import argparse
import os
import time
from typing import Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from speccache.cache.engine import ChunkRoutingDecision, ChunkRoutingStatus, SpecCacheEngine
from speccache.cache.pinned_pool import PinnedKVPool
from speccache.cache.radix_index import ChunkRadixTree
from speccache.cache.semantic_index import RouteDecision, SemanticHNSWIndex
from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    dequant_value_ref,
    pytorch_eager_dequant_rope_shift,
    quantize_kv_int8,
    rotate_half,
)

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


def calculate_int8_bytes(num_tokens: int, num_heads: int, head_dim: int) -> int:
    """Exact bytes transferred across PCIe for INT8 KV cache."""
    return (
        num_tokens * num_heads * head_dim * 1   # k_quant
        + num_tokens * num_heads * 1 * 2       # k_scale
        + num_tokens * num_heads * head_dim * 1 # v_quant
        + num_tokens * num_heads * 1 * 2       # v_scale
    )


def calculate_fp16_bytes(num_tokens: int, num_heads: int, head_dim: int) -> int:
    """Exact bytes transferred across PCIe for uncompressed FP16 KV cache."""
    return (
        num_tokens * num_heads * head_dim * 2   # k_fp16
        + num_tokens * num_heads * head_dim * 2 # v_fp16
    )


# ==============================================================================
# Baseline 1: Naive FP16 Pageable + Eager PyTorch RoPE Shift
# ==============================================================================

def benchmark_baseline1_fp16_pageable(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    chunk_size: int,
    device: torch.device,
    warmup: int = 3,
    rep: int = 15,
) -> float:
    """Benchmark uncompressed FP16 KV transfer from pageable CPU RAM + Eager PyTorch RoPE."""
    torch.manual_seed(42)
    num_chunks = num_tokens // chunk_size
    k_fp16_chunks = [
        torch.randn((chunk_size, num_heads, head_dim), dtype=torch.float16, device="cpu")
        for _ in range(num_chunks)
    ]
    v_fp16_chunks = [
        torch.randn((chunk_size, num_heads, head_dim), dtype=torch.float16, device="cpu")
        for _ in range(num_chunks)
    ]

    out_k = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
    out_v = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)

    # Pre-allocated GPU staging buffers (zero cudaMalloc inside timing loop)
    stage_k = torch.empty((chunk_size, num_heads, head_dim), dtype=torch.float16, device=device)
    stage_v = torch.empty((chunk_size, num_heads, head_dim), dtype=torch.float16, device=device)

    delta_positions = torch.tensor([c * 64 for c in range(num_chunks)], dtype=torch.float32, device=device)
    cos_all, sin_all = compute_rope_cos_sin_delta(delta_positions, head_dim, device=device)

    def run_iter():
        for i in range(num_chunks):
            offset = i * chunk_size
            stage_k.copy_(k_fp16_chunks[i], non_blocking=False)
            stage_v.copy_(v_fp16_chunks[i], non_blocking=False)

            cos_i = cos_all[i : i + 1]
            sin_i = sin_all[i : i + 1]
            cos_full = torch.cat([cos_i, cos_i], dim=-1)[:, None, :].to(torch.float32)
            sin_full = torch.cat([sin_i, sin_i], dim=-1)[:, None, :].to(torch.float32)

            k_f32 = stage_k.to(torch.float32)
            k_rot = (k_f32 * cos_full + rotate_half(k_f32) * sin_full).to(torch.float16)

            out_k[offset : offset + chunk_size].copy_(k_rot)
            out_v[offset : offset + chunk_size].copy_(stage_v)
        torch.cuda.synchronize(device)

    for _ in range(warmup):
        run_iter()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        run_iter()
    end_event.record()
    torch.cuda.synchronize(device)

    return start_event.elapsed_time(end_event) / rep


# ==============================================================================
# Baseline 2: INT8 Pageable + Phase 1 Triton Kernel (Single Stream)
# ==============================================================================

def benchmark_baseline2_int8_pageable(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    chunk_size: int,
    device: torch.device,
    warmup: int = 3,
    rep: int = 15,
) -> float:
    """Benchmark INT8 KV from pageable CPU RAM + Phase 1 Triton kernel sequentially."""
    torch.manual_seed(42)
    num_chunks = num_tokens // chunk_size
    kq_chunks = []
    ks_chunks = []
    vq_chunks = []
    vs_chunks = []
    for _ in range(num_chunks):
        raw_k = torch.randn((chunk_size, num_heads, head_dim), dtype=torch.float32, device="cpu")
        raw_v = torch.randn((chunk_size, num_heads, head_dim), dtype=torch.float32, device="cpu")
        kq, ks = quantize_kv_int8(raw_k)
        vq, vs = quantize_kv_int8(raw_v)
        kq_chunks.append(kq)
        ks_chunks.append(ks)
        vq_chunks.append(vq)
        vs_chunks.append(vs)

    out_k = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
    out_v = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)

    # Pre-allocated GPU staging buffers (zero cudaMalloc inside timing loop)
    stage_kq = torch.empty((chunk_size, num_heads, head_dim), dtype=torch.int8, device=device)
    stage_ks = torch.empty((chunk_size, num_heads, 1), dtype=torch.float16, device=device)
    stage_vq = torch.empty((chunk_size, num_heads, head_dim), dtype=torch.int8, device=device)
    stage_vs = torch.empty((chunk_size, num_heads, 1), dtype=torch.float16, device=device)

    delta_positions = torch.tensor([c * 64 for c in range(num_chunks)], dtype=torch.float32, device=device)
    cos_all, sin_all = compute_rope_cos_sin_delta(delta_positions, head_dim, device=device)

    def run_iter():
        for i in range(num_chunks):
            offset = i * chunk_size
            stage_kq.copy_(kq_chunks[i], non_blocking=False)
            stage_ks.copy_(ks_chunks[i], non_blocking=False)
            stage_vq.copy_(vq_chunks[i], non_blocking=False)
            stage_vs.copy_(vs_chunks[i], non_blocking=False)

            triton_fused_k_dequant_rope_shift(
                stage_kq,
                stage_ks,
                cos_all[i : i + 1],
                sin_all[i : i + 1],
                out=out_k[offset : offset + chunk_size],
            )
            dequant_value(
                stage_vq,
                stage_vs,
                out=out_v[offset : offset + chunk_size],
            )
        torch.cuda.synchronize(device)

    for _ in range(warmup):
        run_iter()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        run_iter()
    end_event.record()
    torch.cuda.synchronize(device)

    return start_event.elapsed_time(end_event) / rep


# ==============================================================================
# Baseline 3: INT8 Pinned Pool + Phase 1 Triton Kernel (Sequential Single-Stream)
# ==============================================================================

def benchmark_baseline3_int8_pinned_sync(
    engine: SpecCacheEngine,
    decisions: List[ChunkRoutingDecision],
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    warmup: int = 3,
    rep: int = 15,
) -> float:
    """Benchmark INT8 KV from pin_memory=True pool + Phase 1 Triton kernel sequentially."""
    chunk_size = engine.chunk_size
    num_chunks = len(decisions)
    out_k = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
    out_v = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)

    # Pre-allocated GPU staging buffers (zero cudaMalloc inside timing loop)
    stage_kq = torch.empty((chunk_size, num_heads, head_dim), dtype=torch.int8, device=device)
    stage_ks = torch.empty((chunk_size, num_heads, 1), dtype=torch.float16, device=device)
    stage_vq = torch.empty((chunk_size, num_heads, head_dim), dtype=torch.int8, device=device)
    stage_vs = torch.empty((chunk_size, num_heads, 1), dtype=torch.float16, device=device)

    delta_positions = torch.tensor([d.delta_p for d in decisions], dtype=torch.float32, device=device)
    cos_all, sin_all = compute_rope_cos_sin_delta(delta_positions, head_dim, device=device)

    def run_iter():
        for i, dec in enumerate(decisions):
            offset = i * chunk_size
            kq_h, ks_h, vq_h, vs_h = engine.pool.get_block_views(dec.block_id)

            stage_kq.copy_(kq_h, non_blocking=False)
            stage_ks.copy_(ks_h, non_blocking=False)
            stage_vq.copy_(vq_h, non_blocking=False)
            stage_vs.copy_(vs_h, non_blocking=False)

            triton_fused_k_dequant_rope_shift(
                stage_kq,
                stage_ks,
                cos_all[i : i + 1],
                sin_all[i : i + 1],
                out=out_k[offset : offset + chunk_size],
            )
            dequant_value(
                stage_vq,
                stage_vs,
                out=out_v[offset : offset + chunk_size],
            )
        torch.cuda.synchronize(device)

    for _ in range(warmup):
        run_iter()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        run_iter()
    end_event.record()
    torch.cuda.synchronize(device)

    return start_event.elapsed_time(end_event) / rep


# ==============================================================================
# Baseline 4: SpecCache Full (INT8 Pinned + Dual-Stream Async Overlapped DMA + Triton)
# ==============================================================================

def benchmark_baseline4_speccache_async(
    engine: SpecCacheEngine,
    decisions: List[ChunkRoutingDecision],
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    warmup: int = 3,
    rep: int = 15,
) -> float:
    """Benchmark SpecCache Full: Pinned INT8 + Dual-Stream Async Pipelined DMA + Triton."""
    out_k = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
    out_v = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)

    for _ in range(warmup):
        engine.execute_async_dma_and_rope(decisions, dest_k_out=out_k, dest_v_out=out_v)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        engine.execute_async_dma_and_rope(decisions, dest_k_out=out_k, dest_v_out=out_v)
    end_event.record()
    torch.cuda.synchronize(device)

    return start_event.elapsed_time(end_event) / rep


# ==============================================================================
# Index Lookup Latency Benchmark (Radix Tree vs. FAISS HNSW)
# ==============================================================================

def benchmark_index_lookup_latencies(
    chunk_size: int = 256,
    embed_dim: int = 128,
    index_sizes: Sequence[int] = (1000, 10000, 50000),
    num_queries: int = 100,
) -> List[Dict[str, float]]:
    """Benchmark lookup latency in microseconds (us) for Radix Tree vs. FAISS HNSW."""
    results: List[Dict[str, float]] = []

    print("\n### Index Lookup Latency Benchmark (Radix Tree vs. FAISS HNSW)")
    print("| Index Chunks (N) | Radix Lookup (us) | FAISS HNSW Lookup (us) | Radix Advantage |")
    print("|---:|---:|---:|---:|")

    for n in index_sizes:
        radix = ChunkRadixTree(chunk_size=chunk_size)
        semantic = SemanticHNSWIndex(embed_dim=embed_dim, max_elements=n + 1000)

        # Generate N distinct chunks
        np.random.seed(42)
        embeddings = np.random.randn(n, embed_dim).astype(np.float32)

        # Insert chunks
        for i in range(n):
            toks = tuple([i * 10 + j for j in range(chunk_size)])
            radix.insert(token_ids=toks, block_ids=i, cached_position=i * chunk_size)
            semantic.add(embedding=embeddings[i], block_id=i, cached_position=i * chunk_size)

        # Query samples
        query_indices = np.random.choice(n, size=num_queries)
        query_tokens = [tuple([idx * 10 + j for j in range(chunk_size)]) for idx in query_indices]
        query_embs = [embeddings[idx] for idx in query_indices]

        # 1. Benchmark Radix Tree exact match
        t0 = time.perf_counter()
        for q_toks in query_tokens:
            _ = radix.match_exact(q_toks)
        t1 = time.perf_counter()
        radix_us = ((t1 - t0) / num_queries) * 1e6

        # 2. Benchmark FAISS HNSW semantic query
        t0 = time.perf_counter()
        for q_emb in query_embs:
            _ = semantic.query(q_emb)
        t1 = time.perf_counter()
        faiss_us = ((t1 - t0) / num_queries) * 1e6

        advantage = faiss_us / max(radix_us, 1e-3)

        results.append({
            "n_chunks": n,
            "radix_us": radix_us,
            "faiss_us": faiss_us,
            "advantage": advantage,
        })

        print(f"| {n:>16,d} | {radix_us:>17.2f} | {faiss_us:>22.2f} | {advantage:>14.1f}x |")

    return results


# ==============================================================================
# Dual-Panel Dark-Mode Plot Rendering
# ==============================================================================

def render_phase2_dark_mode_plots(
    token_counts: List[int],
    results: List[Dict[str, float]],
    device_name: str,
    output_path: str,
) -> None:
    """Generate high-resolution dark-mode dual-panel visualization."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Color palette
    c_b1 = "#ff3366"  # Crimson: Naive FP16 Pageable
    c_b2 = "#ff9900"  # Amber: INT8 Pageable + Triton
    c_b3 = "#39ff14"  # Neon Green: INT8 Pinned + Triton
    c_b4 = "#00f0ff"  # Vibrant Cyan: SpecCache Full Pipelined DMA

    # Panel 1: End-to-End Latency vs Sequence Length
    ax1.plot(token_counts, [r["ms_b1"] for r in results], "^:", color=c_b1, linewidth=2.0, markersize=7, label="Baseline 1: Naive FP16 Pageable + PyTorch")
    ax1.plot(token_counts, [r["ms_b2"] for r in results], "s--", color=c_b2, linewidth=2.0, markersize=7, label="Baseline 2: INT8 Pageable + Triton (Sync)")
    ax1.plot(token_counts, [r["ms_b3"] for r in results], "d--", color=c_b3, linewidth=2.2, markersize=7, label="Baseline 3: INT8 Pinned + Triton (Sync)")
    ax1.plot(token_counts, [r["ms_b4"] for r in results], "o-", color=c_b4, linewidth=2.8, markersize=8, label="Baseline 4: SpecCache Pipelined Async DMA")

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Total Context Tokens (T)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax1.set_ylabel("Retrieval + Reconstruction Latency (ms)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax1.set_title(f"End-to-End Latency vs Context Length ({device_name})", fontsize=12, fontweight="bold", color="#ffffff")
    ax1.set_xticks(token_counts)
    ax1.set_xticklabels([str(t) for t in token_counts], rotation=45, color="#e0e0e0")
    ax1.legend(frameon=True, facecolor="#1e1e1e", edgecolor="#333333", loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#444444")

    # Panel 2: Effective PCIe Throughput (GB/s) vs Sequence Length
    ax2.plot(token_counts, [r["bw_b4"] for r in results], "o-", color=c_b4, linewidth=2.8, markersize=8, label="SpecCache Pipelined Async DMA")
    ax2.plot(token_counts, [r["bw_b3"] for r in results], "d--", color=c_b3, linewidth=2.2, markersize=7, label="INT8 Pinned + Triton (Sync)")
    ax2.plot(token_counts, [r["bw_b2"] for r in results], "s--", color=c_b2, linewidth=2.0, markersize=7, label="INT8 Pageable + Triton (Sync)")
    ax2.plot(token_counts, [r["bw_b1"] for r in results], "^:", color=c_b1, linewidth=2.0, markersize=7, label="Naive FP16 Pageable")

    # PCIe Gen4 x16 unidirectional practical roofline (~26 GB/s)
    ax2.axhline(26.0, color="#888888", linestyle="--", linewidth=1.5, label="PCIe Gen4 x16 Bus Limit (~26 GB/s)")

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Total Context Tokens (T)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax2.set_ylabel("Effective PCIe Throughput (GB/s)", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax2.set_title("Effective PCIe Throughput Scaling", fontsize=12, fontweight="bold", color="#ffffff")
    ax2.set_xticks(token_counts)
    ax2.set_xticklabels([str(t) for t in token_counts], rotation=45, color="#e0e0e0")
    ax2.legend(frameon=True, facecolor="#1e1e1e", edgecolor="#333333", loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#444444")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#121212")
    plt.close(fig)


# ==============================================================================
# Main Benchmark CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SpecCache Phase 2 PCIe DMA Pipeline Benchmark")
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Token count per chunk (default 256)")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of KV heads (default 8)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension D (default 128)")
    parser.add_argument("--output-plot", type=str, default="benchmarks/phase2_dma_pipeline.png", help="Plot PNG path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target device")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        print("SpecCache Phase 2 PCIe DMA benchmarks require an NVIDIA GPU runtime (e.g. Google Colab).")
        return

    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)

    print("=" * 90)
    print("SpecCache Phase 2: PCIe DMA Pipeline & Double-Buffered Triton Benchmark")
    print(f"Device: {device_name}")
    print(f"Config: Chunk Size={args.chunk_size}, Heads={args.num_heads}, Head Dim={args.head_dim}")
    print("=" * 90)

    max_tokens = max(args.tokens)
    max_blocks = max_tokens // args.chunk_size + 32

    engine = SpecCacheEngine(
        chunk_size=args.chunk_size,
        num_kv_heads=args.num_heads,
        head_dim=args.head_dim,
        max_blocks=max_blocks,
        device=device,
    )

    # Populate pool with synthetic chunks
    for i in range(max_blocks):
        toks = list(range(i * 1000, i * 1000 + args.chunk_size))
        raw_k = torch.randn((args.chunk_size, args.num_heads, args.head_dim), dtype=torch.float32)
        raw_v = torch.randn((args.chunk_size, args.num_heads, args.head_dim), dtype=torch.float32)
        kq, ks = quantize_kv_int8(raw_k)
        vq, vs = quantize_kv_int8(raw_v)
        engine.register_chunk(toks, cached_position=i * 64, k_quant=kq, k_scale=ks, v_quant=vq, v_scale=vs)

    print("\n### End-to-End Host-to-GPU Retrieval + Reconstruction Benchmark")
    print("| Tokens | Chunks | FP16 Pageable (ms) | INT8 Pageable (ms) | INT8 Pinned (ms) | SpecCache Full (ms) | Speedup (vs FP16) | Throughput (GB/s) |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")

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

        int8_bytes = calculate_int8_bytes(t, args.num_heads, args.head_dim)
        fp16_bytes = calculate_fp16_bytes(t, args.num_heads, args.head_dim)

        ms_b1 = benchmark_baseline1_fp16_pageable(t, args.num_heads, args.head_dim, args.chunk_size, device)
        ms_b2 = benchmark_baseline2_int8_pageable(t, args.num_heads, args.head_dim, args.chunk_size, device)
        ms_b3 = benchmark_baseline3_int8_pinned_sync(engine, decisions, t, args.num_heads, args.head_dim, device)
        ms_b4 = benchmark_baseline4_speccache_async(engine, decisions, t, args.num_heads, args.head_dim, device)

        bw_b1 = (fp16_bytes / (ms_b1 * 1e-3)) / 1e9
        bw_b2 = (int8_bytes / (ms_b2 * 1e-3)) / 1e9
        bw_b3 = (int8_bytes / (ms_b3 * 1e-3)) / 1e9
        bw_b4 = (int8_bytes / (ms_b4 * 1e-3)) / 1e9

        speedup = ms_b1 / ms_b4

        results.append({
            "tokens": t,
            "chunks": num_chunks,
            "ms_b1": ms_b1,
            "ms_b2": ms_b2,
            "ms_b3": ms_b3,
            "ms_b4": ms_b4,
            "bw_b1": bw_b1,
            "bw_b2": bw_b2,
            "bw_b3": bw_b3,
            "bw_b4": bw_b4,
            "speedup": speedup,
        })

        print(
            f"| {t:>6} | {num_chunks:>6} | {ms_b1:>18.2f} | {ms_b2:>18.2f} | {ms_b3:>16.2f} | "
            f"{ms_b4:>19.2f} | {speedup:>17.2f}x | {bw_b4:>17.2f} |"
        )

    print("=" * 90)

    # 2. Run Index Lookup Latency Benchmark (N = 1k, 10k, 50k)
    benchmark_index_lookup_latencies(chunk_size=args.chunk_size, embed_dim=128)

    # 3. Save dual-panel dark-mode visualization
    print(f"\nSaving Phase 2 dark-mode benchmark plot -> {args.output_plot}")
    render_phase2_dark_mode_plots(args.tokens, results, device_name, args.output_plot)
    print("Phase 2 benchmark suite complete.")


if __name__ == "__main__":
    main()
