"""End-to-End LLM Serving & Speculative Verification Benchmark for SpecCache Phase 3.

Evaluates:
Panel 1: Time-To-First-Token (TTFT in ms) vs. Prompt Context Length (T in [512, 1024, 2048, 4096, 8192]):
  - Standard Full GPU Prefill (Cold Miss)
  - Naive FP16 Host Swap + Eager RoPE
  - SpecCache (Pinned INT8 DMA + Fused Triton RoPE)
Panel 2: Effective Decoding Throughput (Tokens/sec) across Speculative Acceptance Rates
  (alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] for gamma = 6):
  - Standard Autoregressive Decode (1 tok/pass)
  - SpecCache Speculative Draft Routing

Generates a formatted Markdown summary table and dark-mode dual-panel visualization -> benchmarks/phase3_e2e_serving.png.
"""

import argparse
import os
import time
from typing import Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from speccache.cache.engine import SpecCacheEngine
from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    pytorch_eager_dequant_rope_shift,
    quantize_kv_int8,
    rotate_half,
)
from speccache.serving.kv_injector import MultiLayerKVInjector, SpecDynamicCache
from speccache.serving.pipeline import (
    MockL4TransformerEngine,
    RoutePath,
    SpecCacheLLMRunner,
)
from speccache.serving.speculative import SpeculativeVerifier

try:
    from speccache.kernels.triton_rope_kv import (
        TRITON_AVAILABLE,
        dequant_value,
        triton_fused_k_dequant_rope_shift,
    )
except ImportError:
    TRITON_AVAILABLE = False


# ==============================================================================
# Panel 1 Benchmarks: TTFT vs Context Length
# ==============================================================================

def benchmark_standard_prefill_ttft(
    model: MockL4TransformerEngine,
    num_tokens: int,
    warmup: int = 2,
    iters: int = 6,
    device: torch.device = torch.device("cuda"),
) -> float:
    """Measure TTFT for Standard Full GPU Prefill (Cold Miss)."""
    input_t = torch.randint(0, model.vocab_size, (1, num_tokens), device=device)

    # Warmup
    for _ in range(warmup):
        cache = SpecDynamicCache()
        out = model.forward(input_t, past_key_values=cache)
        _ = out.logits[:, -1, :].argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    timings: List[float] = []

    for _ in range(iters):
        cache = SpecDynamicCache()
        if device.type == "cuda":
            start_event.record()
            out = model.forward(input_t, past_key_values=cache)
            _ = out.logits[:, -1, :].argmax(dim=-1)
            end_event.record()
            end_event.synchronize()
            timings.append(start_event.elapsed_time(end_event))
        else:
            t0 = time.perf_counter()
            out = model.forward(input_t, past_key_values=cache)
            _ = out.logits[:, -1, :].argmax(dim=-1)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def benchmark_naive_fp16_swap_ttft(
    model: MockL4TransformerEngine,
    num_tokens: int,
    delta_p: int = 64,
    warmup: int = 2,
    iters: int = 6,
    device: torch.device = torch.device("cuda"),
) -> float:
    """Measure TTFT for Naive FP16 Host Swap + Eager PyTorch RoPE."""
    # Allocate uncompressed FP16 KV on pageable CPU host memory
    k_host = [
        torch.randn(1, model.num_kv_heads, num_tokens, model.head_dim, dtype=torch.float16)
        for _ in range(model.num_layers)
    ]
    v_host = [
        torch.randn(1, model.num_kv_heads, num_tokens, model.head_dim, dtype=torch.float16)
        for _ in range(model.num_layers)
    ]

    cos_d, sin_d = compute_rope_cos_sin_delta(
        torch.tensor([delta_p], device=device), head_dim=model.head_dim, device=device
    )
    cos_full = torch.cat([cos_d, cos_d], dim=-1)[None, :, None, :].to(torch.float32)
    sin_full = torch.cat([sin_d, sin_d], dim=-1)[None, :, None, :].to(torch.float32)

    last_token = torch.randint(0, model.vocab_size, (1, 1), device=device)

    # Warmup
    for _ in range(warmup):
        cache = SpecDynamicCache()
        for l in range(model.num_layers):
            k_g = k_host[l].to(device)
            v_g = v_host[l].to(device)
            k_rot = (k_g.to(torch.float32) * cos_full + rotate_half(k_g.to(torch.float32)) * sin_full).to(torch.float16)
            cache.update(k_rot, v_g, layer_idx=l)
        out = model.forward(last_token, past_key_values=cache)
        _ = out.logits[:, -1, :].argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cache = SpecDynamicCache()
        for l in range(model.num_layers):
            k_g = k_host[l].to(device)
            v_g = v_host[l].to(device)
            k_rot = (k_g.to(torch.float32) * cos_full + rotate_half(k_g.to(torch.float32)) * sin_full).to(torch.float16)
            cache.update(k_rot, v_g, layer_idx=l)
        out = model.forward(last_token, past_key_values=cache)
        _ = out.logits[:, -1, :].argmax(dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def benchmark_speccache_pinned_dma_ttft(
    model: MockL4TransformerEngine,
    injector: MultiLayerKVInjector,
    num_tokens: int,
    chunk_size: int = 64,
    delta_p: int = 64,
    warmup: int = 2,
    iters: int = 6,
    device: torch.device = torch.device("cuda"),
) -> float:
    """Measure TTFT for SpecCache (Pinned INT8 DMA + Fused Triton RoPE)."""
    num_chunks = num_tokens // chunk_size
    block_ids = list(range(num_chunks))
    delta_ps = [delta_p] * num_chunks
    last_token = torch.randint(0, model.vocab_size, (1, 1), device=device)

    # Warmup
    for _ in range(warmup):
        cache = SpecDynamicCache()
        injector.reconstruct_and_inject_multi_layer(cache, block_ids=block_ids, delta_ps=delta_ps)
        out = model.forward(last_token, past_key_values=cache)
        _ = out.logits[:, -1, :].argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cache = SpecDynamicCache()
        injector.reconstruct_and_inject_multi_layer(cache, block_ids=block_ids, delta_ps=delta_ps)
        out = model.forward(last_token, past_key_values=cache)
        _ = out.logits[:, -1, :].argmax(dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


# ==============================================================================
# Panel 2 Benchmarks: Speculative Throughput vs Acceptance Rate
# ==============================================================================

def benchmark_speculative_acceptance_rates(
    model: MockL4TransformerEngine,
    acceptance_rates: List[float],
    gamma: int = 6,
    prompt_len: int = 256,
    tokens_to_generate: int = 120,
    device: torch.device = torch.device("cuda"),
) -> Tuple[float, List[float]]:
    """Measure effective generation throughput across speculative acceptance rates alpha."""
    prompt_ids = torch.randint(0, model.vocab_size, (1, prompt_len), device=device)

    # Baseline: Standard Autoregressive Decode (1 token per step)
    cache_base = SpecDynamicCache()
    out = model.forward(prompt_ids, past_key_values=cache_base)
    curr_tok = out.logits[:, -1, :].argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0_base = time.perf_counter()
    for _ in range(tokens_to_generate):
        step_in = curr_tok.unsqueeze(0) if curr_tok.ndim == 1 else curr_tok
        out = model.forward(step_in, past_key_values=cache_base)
        curr_tok = out.logits[:, -1, :].argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    base_tps = tokens_to_generate / max(time.perf_counter() - t0_base, 1e-4)

    # Speculative Routing across acceptance rates alpha
    spec_throughputs: List[float] = []

    for alpha in acceptance_rates:
        cache_spec = SpecDynamicCache()
        out = model.forward(prompt_ids, past_key_values=cache_spec)
        curr_tok = out.logits[:, -1, :].argmax(dim=-1)

        generated = 0
        t0_spec = time.perf_counter()

        # Simulate speculative verification step with acceptance rate alpha
        # Expected accepted tokens per pass: round(alpha * gamma)
        k_accept = int(round(alpha * gamma))

        while generated < tokens_to_generate:
            # Parallel forward pass of prompt + draft tokens (length gamma + 1)
            draft_input = torch.randint(0, model.vocab_size, (1, gamma + 1), device=device)
            out_ver = model.forward(draft_input, past_key_values=cache_spec)
            _ = out_ver.logits[:, -1, :].argmax(dim=-1)

            # Emitted tokens: k_accept draft tokens + 1 bonus token
            emitted_count = k_accept + 1
            generated += emitted_count

            # Roll back KV cache
            cache_spec.truncate(prompt_len + generated)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t1_spec = time.perf_counter()
        tps = generated / max(t1_spec - t0_spec, 1e-4)
        spec_throughputs.append(tps)

    return base_tps, spec_throughputs


# ==============================================================================
# Dual-Panel Dark-Mode Plot Rendering
# ==============================================================================

def render_phase3_dark_mode_plots(
    token_counts: List[int],
    ms_prefill: List[float],
    ms_naive: List[float],
    ms_speccache: List[float],
    acceptance_rates: List[float],
    base_tps: float,
    spec_throughputs: List[float],
    device_name: str,
    output_path: str,
) -> None:
    """Generate publication-grade dark-mode dual-panel visualization for Phase 3."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Color palette
    c_prefill = "#ef4444"    # Red / Coral: Full Prefill Baseline
    c_naive = "#f59e0b"      # Amber: Naive FP16 Host Swap
    c_speccache = "#00f0ff"   # Electric Cyan: SpecCache
    c_speedup = "#10b981"    # Emerald: Speedup callouts
    c_spec = "#a855f7"       # Purple: Speculative decode

    # Panel 1: TTFT vs Context Tokens
    ax1.plot(token_counts, ms_prefill, "^--", color=c_prefill, linewidth=2.2, markersize=7, label="Standard Full GPU Prefill (Cold Miss)")
    ax1.plot(token_counts, ms_naive, "s-.", color=c_naive, linewidth=2.2, markersize=7, label="Naive FP16 Host Swap + Eager RoPE")
    ax1.plot(token_counts, ms_speccache, "o-", color=c_speccache, linewidth=3.0, markersize=8, label="SpecCache (Pinned INT8 DMA + Fused Triton RoPE)")

    # Speedup annotation at max context
    max_idx = -1
    speedup_vs_prefill = ms_prefill[max_idx] / max(ms_speccache[max_idx], 1e-4)
    speedup_vs_naive = ms_naive[max_idx] / max(ms_speccache[max_idx], 1e-4)
    ax1.annotate(
        f"{speedup_vs_prefill:.1f}x vs Prefill\n{speedup_vs_naive:.1f}x vs FP16 Swap",
        xy=(token_counts[max_idx], ms_speccache[max_idx]),
        xytext=(token_counts[max_idx] * 0.38, ms_speccache[max_idx] * 4.0),
        arrowprops=dict(facecolor=c_speedup, edgecolor="none", shrink=0.08, width=1.5, headwidth=6),
        color=c_speedup,
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor=c_speedup, alpha=0.9),
    )

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Prompt Context Tokens (T)", fontsize=11, fontweight="bold", color="#e2e8f0")
    ax1.set_ylabel("Time-To-First-Token (TTFT, ms) [Log Scale]", fontsize=11, fontweight="bold", color="#e2e8f0")
    ax1.set_title(f"Time-To-First-Token (TTFT) vs Prompt Context Length ({device_name})", fontsize=12, fontweight="bold", color="#ffffff")
    ax1.set_xticks(token_counts)
    ax1.set_xticklabels([str(t) for t in token_counts], rotation=45, color="#cbd5e1")
    ax1.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", loc="upper left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.25, color="#475569")

    # Panel 2: Speculative Throughput vs Acceptance Rate
    x_positions = np.arange(len(acceptance_rates))
    bars = ax2.bar(
        x_positions,
        spec_throughputs,
        width=0.55,
        color=c_spec,
        edgecolor="#c084fc",
        linewidth=1.2,
        alpha=0.85,
        label="SpecCache Speculative Draft Routing (gamma=6)",
    )

    # Baseline reference line
    ax2.axhline(
        base_tps,
        color=c_prefill,
        linestyle="--",
        linewidth=2.0,
        label=f"Standard Autoregressive Decode (1 tok/pass: {base_tps:.1f} t/s)",
    )

    # Annotate speedup above each bar
    for idx, (bar, tps) in enumerate(zip(bars, spec_throughputs)):
        speedup = tps / max(base_tps, 1e-4)
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + (max(spec_throughputs) * 0.03),
            f"{speedup:.2f}x\n({tps:.1f} t/s)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#38bdf8",
        )

    ax2.set_xlabel("Speculative Acceptance Rate (alpha)", fontsize=11, fontweight="bold", color="#e2e8f0")
    ax2.set_ylabel("Effective Decoding Throughput (tokens / sec)", fontsize=11, fontweight="bold", color="#e2e8f0")
    ax2.set_title("Effective Decoding Throughput across Acceptance Rates (gamma=6)", fontsize=12, fontweight="bold", color="#ffffff")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f"alpha={a:.1f}" for a in acceptance_rates], color="#cbd5e1")
    ax2.set_ylim(0, max(spec_throughputs) * 1.30)
    ax2.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155", loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.25, color="#475569", axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0b0f19")
    plt.close(fig)


# ==============================================================================
# Main Benchmark CLI
# ==============================================================================

def main():
    """Execute Phase 3 End-to-End LLM Serving & Speculative Verification Benchmark."""
    parser = argparse.ArgumentParser(description="SpecCache Phase 3 End-to-End Serving Benchmark")
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="Sequence lengths for TTFT evaluation",
    )
    parser.add_argument(
        "--acceptance-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Speculative acceptance rates alpha",
    )
    parser.add_argument("--gamma", type=int, default=6, help="Draft length gamma (default 6)")
    parser.add_argument("--num-layers", type=int, default=24, help="Transformer layers (default 24)")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="KV heads H_kv (default 8)")
    parser.add_argument("--num-q-heads", type=int, default=16, help="Query heads H_q (default 16)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension D (default 128)")
    parser.add_argument("--chunk-size", type=int, default=64, help="Tokens per chunk (default 64)")
    parser.add_argument("--output-plot", type=str, default="benchmarks/phase3_e2e_serving.png", help="Output PNG path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target device")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("CUDA is not available on this system.")
        print("SpecCache Phase 3 benchmarks require an NVIDIA GPU runtime (e.g. Google Colab).")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    print("=" * 105)
    print("SpecCache Phase 3: End-to-End LLM Serving & Speculative Verification Benchmark")
    print(f"Device: {device_name}")
    print(f"Model Architecture: Layers={args.num_layers}, H_q={args.num_q_heads}, H_kv={args.num_kv_heads}, D={args.head_dim}")
    print(f"Serving Configuration: Chunk Size={args.chunk_size}, Speculative Draft Length gamma={args.gamma}")
    print("=" * 105)

    # 1. Initialize 24-layer L4 Transformer Stack
    model = MockL4TransformerEngine(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        num_q_heads=args.num_q_heads,
        head_dim=args.head_dim,
        hidden_dim=args.num_q_heads * args.head_dim,
        vocab_size=32000,
        device=device,
    )

    max_tokens = max(args.tokens)
    max_blocks = max_tokens // args.chunk_size + 32

    injector = MultiLayerKVInjector(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        device=device,
        max_blocks=max_blocks,
        chunk_size=args.chunk_size,
    )

    # Pre-populate pinned pool with quantized chunks
    print("Pre-populating layer-wise Pinned Host Memory Pools...")
    for b_id in range(max_blocks):
        for l in range(args.num_layers):
            k_raw = torch.randn((args.chunk_size, args.num_kv_heads, args.head_dim), dtype=torch.float32)
            v_raw = torch.randn((args.chunk_size, args.num_kv_heads, args.head_dim), dtype=torch.float32)
            kq, ks = quantize_kv_int8(k_raw)
            vq, vs = quantize_kv_int8(v_raw)
            injector.layer_pools[l].allocate(block_id=b_id, start_pos=b_id * args.chunk_size)
            injector.layer_pools[l].write_block(
                block_id=b_id,
                k_quant=kq,
                k_scale=ks,
                v_quant=vq,
                v_scale=vs,
                start_pos=b_id * args.chunk_size,
            )

    # Panel 1: TTFT Benchmarking
    print("\n### Panel 1: Time-To-First-Token (TTFT) vs Prompt Context Length")
    print("| Context (T) | Full GPU Prefill (ms) | Naive FP16 Swap (ms) | SpecCache INT8 DMA (ms) | Speedup (vs Prefill) | Speedup (vs FP16 Swap) |")
    print("|------------:|----------------------:|---------------------:|------------------------:|---------------------:|-----------------------:|")

    ms_prefill: List[float] = []
    ms_naive: List[float] = []
    ms_speccache: List[float] = []

    for t in args.tokens:
        t_prefill = benchmark_standard_prefill_ttft(model, num_tokens=t, device=device)
        t_naive = benchmark_naive_fp16_swap_ttft(model, num_tokens=t, device=device)
        t_speccache = benchmark_speccache_pinned_dma_ttft(
            model, injector, num_tokens=t, chunk_size=args.chunk_size, device=device
        )

        ms_prefill.append(t_prefill)
        ms_naive.append(t_naive)
        ms_speccache.append(t_speccache)

        sp_prefill = t_prefill / max(t_speccache, 1e-4)
        sp_naive = t_naive / max(t_speccache, 1e-4)

        print(
            f"| {t:>11} | {t_prefill:>21.2f} | {t_naive:>20.2f} | {t_speccache:>23.2f} | "
            f"{sp_prefill:>20.2f}x | {sp_naive:>22.2f}x |"
        )

    # Panel 2: Speculative Throughput Benchmarking
    print(f"\n### Panel 2: Effective Decoding Throughput across Acceptance Rates (gamma={args.gamma})")
    print("| Acceptance Rate (alpha) | Standard Autoregressive (t/s) | SpecCache Speculative (t/s) | Speedup Factor |")
    print("|------------------------:|------------------------------:|----------------------------:|---------------:|")

    base_tps, spec_throughputs = benchmark_speculative_acceptance_rates(
        model,
        acceptance_rates=args.acceptance_rates,
        gamma=args.gamma,
        device=device,
    )

    for a, s_tps in zip(args.acceptance_rates, spec_throughputs):
        sp = s_tps / max(base_tps, 1e-4)
        print(f"| {a:>24.1f} | {base_tps:>29.2f} | {s_tps:>27.2f} | {sp:>14.2f}x |")

    print("=" * 105)

    # Render Dual-Panel Visualization
    print(f"\nSaving Phase 3 publication-grade benchmark plot -> {args.output_plot}")
    render_phase3_dark_mode_plots(
        token_counts=args.tokens,
        ms_prefill=ms_prefill,
        ms_naive=ms_naive,
        ms_speccache=ms_speccache,
        acceptance_rates=args.acceptance_rates,
        base_tps=base_tps,
        spec_throughputs=spec_throughputs,
        device_name=device_name,
        output_path=args.output_plot,
    )
    print("SpecCache Phase 3 benchmark suite complete.")


if __name__ == "__main__":
    main()
