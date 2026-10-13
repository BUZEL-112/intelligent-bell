# SpecCache: Hardware-Aware Semantic KV-Cache Engine

SpecCache is a hardware-aware semantic KV-cache engine designed for high-throughput LLM serving.

---

## Architecture Overview

SpecCache operates across two core tiers:
1. **Phase 1: Fused INT8 Dequantization & Relative RoPE Phase-Shifting Kernel**
   - Single-pass on-chip SRAM execution eliminates intermediate HBM read/write roundtrips.
   - 2D grid tiling (`BLOCK_T`, `BLOCK_H`) ensures high occupancy on Ada Lovelace, Ampere, and Hopper architectures.
   - RoPE identity $R(\theta, m + \Delta p) = R(\theta, \Delta p) R(\theta, m)$ enables in-place chunk phase-shifting without knowing original positions $m$.
2. **Phase 2: Hybrid Radix-Semantic Lookup & Pinned-Memory DMA Pipeline**
   - **Tier 1 (Exact Match):** Token-level Radix Tree for $O(L)$ exact prefix/chunk matching.
   - **Tier 2 (Semantic Match):** FAISS HNSW index (`METRIC_INNER_PRODUCT`, `M=32`, `efSearch=32`) over normalized chunk embeddings.
   - **Three-Way Routing:**
     - Direct KV Reuse ($\text{cos\_sim} \ge 0.92$ or exact Radix hit): Computes $\Delta p = p_{\text{target}} - p_{\text{cached}}$ and executes fused kernel.
     - Speculative Draft Zone ($0.80 \le \text{cos\_sim} < 0.92$): Marks chunk as `SPECULATIVE_DRAFT` and surfaces cached continuation tokens.
     - Cache Miss ($\text{cos\_sim} < 0.80$): Marks chunk for standard GPU prefill computation.
   - **Page-Locked Host Memory Pool:** Contiguous host memory slab allocator (`pin_memory=True`) with LRU eviction management.
   - **Pipelined Double-Buffered Async DMA:** Overlaps PCIe host-to-device transfers of chunk $i+1$ with Phase 1 Triton kernel computation of chunk $i$ across dual CUDA streams.

---

## Repository Structure

```text
speccache/
├── __init__.py             # Public API exports
├── cache/
│   ├── __init__.py
│   ├── pinned_pool.py      # Page-locked host memory slab allocator + LRU eviction
│   ├── radix_index.py      # Token-chunk Radix tree for exact prefix/block hits
│   ├── semantic_index.py   # FAISS HNSW wrapper + cosine threshold routing logic
│   └── engine.py           # SpecCacheEngine: orchestrates lookup, async DMA, and Triton kernels
├── kernels/
│   ├── __init__.py
│   ├── reference.py        # Eager PyTorch reference + torch.compile baseline
│   └── triton_rope_kv.py   # Autotuned 2D-tiled Triton kernels
├── tests/
│   ├── __init__.py
│   ├── test_correctness.py   # Phase 1 numerical verification and rotational equivalence suite
│   └── test_phase2_engine.py # Phase 2 PinnedPool, Radix, FAISS, and Async DMA parity suite
└── benchmarks/
    ├── __init__.py
    ├── bench_roofline.py     # Phase 1 Triton HBM roofline benchmark
    └── bench_phase2_dma.py   # Phase 2 Pageable vs Pinned vs Async DMA PCIe benchmark
├── Makefile                # Unified build, test, and benchmark targets
├── run_phase1.sh           # Phase 1 automated run script
├── run_phase2.sh           # Phase 2 automated run script
└── pyproject.toml          # Project metadata and dependencies
```

---

## Validation & Benchmarking Guide (Google Colab / GPU Instance)

### 1. Automated Execution via Shell Scripts

Run Phase 1 (Kernel verification + HBM roofline benchmark):
```bash
chmod +x run_phase1.sh && ./run_phase1.sh
```

Run Phase 2 (Engine verification + PCIe DMA overlap benchmark):
```bash
chmod +x run_phase2.sh && ./run_phase2.sh
```

Or using the Makefile:
```bash
make test        # Runs both test_correctness.py and test_phase2_engine.py
make benchmark   # Runs both bench_roofline.py and bench_phase2_dma.py
```

### 2. Step-by-Step Validation

#### Run Phase 2 Tests
```bash
pytest tests/test_phase2_engine.py -v
```
Verifies:
- Pinned pool memory allocation and LRU eviction recycling.
- Radix tree exact matching, edge splitting on shared prefixes, and $O(1)$ eviction.
- Semantic HNSW index cosine routing thresholds ($0.92$ and $0.80$).
- Two-tier lookup orchestration in `SpecCacheEngine`.
- Asynchronous dual-stream DMA pipeline numerical parity against eager PyTorch.

#### Run Phase 2 PCIe DMA Benchmark
```bash
python benchmarks/bench_phase2_dma.py --output-plot benchmarks/phase2_dma_overlap.png
```
Produces:
- Formatted Markdown throughput table comparing Pageable (Sync), Pinned (Sync), and Pipelined Double-Buffered Async DMA.
- High-resolution dark-mode dual-subplot plot saved to `benchmarks/phase2_dma_overlap.png`.

---

## Phase 2 Python API Usage

```python
import torch
from speccache import SpecCacheEngine, quantize_kv_int8

# Initialize engine
engine = SpecCacheEngine(
    chunk_size=128,
    num_kv_heads=8,
    head_dim=128,
    max_blocks=1024,
    embedding_dim=128,
    tau_reuse=0.92,
    tau_spec=0.80,
    device=torch.device("cuda"),
)

# Register a cached prompt chunk
token_ids = list(range(100, 228))
raw_k = torch.randn((128, 8, 128), dtype=torch.float32)
raw_v = torch.randn((128, 8, 128), dtype=torch.float32)
k_quant, k_scale = quantize_kv_int8(raw_k)
v_quant, v_scale = quantize_kv_int8(raw_v)
embedding = torch.randn(128)

block_id = engine.register_chunk(
    token_ids=token_ids,
    cached_position=0,
    k_quant=k_quant,
    k_scale=k_scale,
    v_quant=v_quant,
    v_scale=v_scale,
    embedding=embedding,
)

# Two-tier hierarchical lookup for incoming prompt
decisions = engine.lookup_chunks(prompt_tokens=token_ids, chunk_embeddings=[embedding])

# Asynchronously stream reused chunks over PCIe and execute relative RoPE
k_gpu, v_gpu = engine.execute_async_dma_and_rope(decisions)
```