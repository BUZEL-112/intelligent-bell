"""Test suite for SpecCache Phase 2: Hybrid Radix-Semantic Engine and Pinned DMA.

Verifies:
1. PinnedHostPool allocation, chunk read/write, and LRU eviction recycling.
2. RadixTree exact matching, common prefix edge splitting, and O(1) eviction pruning.
3. SemanticIndex cosine threshold routing (DIRECT_REUSE, SPECULATIVE_DRAFT, CACHE_MISS).
4. SpecCacheEngine two-tier hierarchical routing (Tier 1 Radix -> Tier 2 FAISS).
5. Asynchronous dual-stream PCIe DMA pipeline parity against PyTorch eager reference.
"""

from typing import List, Tuple
import numpy as np
import pytest
import torch

from speccache.cache.engine import (
    ChunkRoutingDecision,
    ChunkRoutingStatus,
    SpecCacheEngine,
)
from speccache.cache.pinned_pool import BlockMetadata, PinnedHostPool
from speccache.cache.radix_index import RadixMatchResult, RadixTree
from speccache.cache.semantic_index import (
    SemanticIndex,
    SemanticLookupResult,
    SemanticRoutingAction,
)
from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    dequant_value_ref,
    pytorch_eager_dequant_rope_shift,
    quantize_kv_int8,
)


def _generate_synthetic_chunk(
    chunk_size: int,
    num_heads: int,
    head_dim: int,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate reproducible quantized Key and Value tensors for a chunk."""
    torch.manual_seed(seed)
    raw_k = torch.randn((chunk_size, num_heads, head_dim), dtype=torch.float32, device=device)
    raw_v = torch.randn((chunk_size, num_heads, head_dim), dtype=torch.float32, device=device)
    k_quant, k_scale = quantize_kv_int8(raw_k)
    v_quant, v_scale = quantize_kv_int8(raw_v)
    return k_quant, k_scale, v_quant, v_scale


# ==============================================================================
# PinnedHostPool Tests
# ==============================================================================

def test_pinned_pool_allocation_and_read_write():
    """Verify writing to and reading from pinned host pool slots."""
    chunk_size, heads, dim = 64, 4, 64
    max_blocks = 4
    pool = PinnedHostPool(max_blocks=max_blocks, chunk_size=chunk_size, num_kv_heads=heads, head_dim=dim)

    assert pool.free_count == 4
    assert pool.allocated_count == 0

    k_q, k_s, v_q, v_s = _generate_synthetic_chunk(chunk_size, heads, dim, seed=1)
    tokens = tuple(range(100, 100 + chunk_size))

    block_id = pool.allocate(cached_position=0, token_ids=tokens)
    assert block_id in range(max_blocks)
    assert pool.allocated_count == 1
    assert pool.free_count == 3

    pool.write_chunk(block_id, k_q, k_s, v_q, v_s)

    r_kq, r_ks, r_vq, r_vs = pool.read_chunk(block_id)
    assert torch.equal(r_kq, k_q)
    assert torch.equal(r_ks, k_s)
    assert torch.equal(r_vq, v_q)
    assert torch.equal(r_vs, v_s)


def test_pinned_pool_lru_eviction():
    """Verify that allocating beyond max_blocks evicts the least recently used block."""
    chunk_size, heads, dim = 32, 2, 64
    max_blocks = 3
    evicted_ids: List[int] = []

    def evict_cb(bid: int):
        evicted_ids.append(bid)

    pool = PinnedHostPool(
        max_blocks=max_blocks,
        chunk_size=chunk_size,
        num_kv_heads=heads,
        head_dim=dim,
        on_evict_callback=evict_cb,
    )

    # Allocate 3 blocks to fill arena
    b0 = pool.allocate(cached_position=0, token_ids=tuple(range(32)))
    b1 = pool.allocate(cached_position=32, token_ids=tuple(range(32, 64)))
    b2 = pool.allocate(cached_position=64, token_ids=tuple(range(64, 96)))

    assert pool.allocated_count == 3
    assert pool.free_count == 0

    # Touch b0 so b1 becomes the LRU block
    pool.touch(b0)

    # Allocating a 4th block must evict b1
    b3 = pool.allocate(cached_position=96, token_ids=tuple(range(96, 128)))
    assert len(evicted_ids) == 1
    assert evicted_ids[0] == b1
    assert b3 == b1  # Recycles the evicted block slot
    assert pool.allocated_count == 3


# ==============================================================================
# RadixTree Tests
# ==============================================================================

def test_radix_tree_exact_match_and_split():
    """Verify exact token sequence matching and edge splitting on shared prefixes."""
    tree = RadixTree()

    # Sequence A: [10, 11, 12, 13]
    tree.insert(token_ids=[10, 11, 12, 13], block_id=1, cached_position=0)

    # Sequence B: [10, 11, 20, 21] (shares prefix [10, 11])
    tree.insert(token_ids=[10, 11, 20, 21], block_id=2, cached_position=100)

    # Look up Sequence A
    match_a = tree.match_exact([10, 11, 12, 13])
    assert match_a is not None
    assert match_a.block_id == 1
    assert match_a.cached_position == 0

    # Look up Sequence B
    match_b = tree.match_exact([10, 11, 20, 21])
    assert match_b is not None
    assert match_b.block_id == 2
    assert match_b.cached_position == 100

    # Look up non-existent sequence
    assert tree.match_exact([10, 11, 99]) is None


def test_radix_tree_eviction():
    """Verify O(1) eviction removes a block and prunes the tree."""
    tree = RadixTree()
    tree.insert(token_ids=[1, 2, 3], block_id=10, cached_position=0)
    assert len(tree) == 1
    assert tree.match_exact([1, 2, 3]) is not None

    # Evict block 10
    assert tree.evict_block(10) is True
    assert len(tree) == 0
    assert tree.match_exact([1, 2, 3]) is None


# ==============================================================================
# SemanticIndex Tests
# ==============================================================================

@pytest.mark.parametrize("force_fallback", [False, True])
def test_semantic_index_threshold_routing(force_fallback: bool):
    """Test 3-way routing: DIRECT_REUSE (>=0.92), SPECULATIVE_DRAFT ([0.80, 0.92)), CACHE_MISS (<0.80)."""
    dim = 64
    index = SemanticIndex(
        embedding_dim=dim,
        tau_reuse=0.92,
        tau_spec=0.80,
        force_fallback=force_fallback,
    )

    base_vec = np.zeros(dim, dtype=np.float32)
    base_vec[0] = 1.0  # Unit vector along axis 0

    index.add(
        embedding=base_vec,
        block_id=5,
        cached_position=128,
        token_ids=[1, 2, 3],
        continuation_tokens=[4, 5],
    )

    # 1. Near-identical query vector (similarity ~0.99 >= 0.92) -> DIRECT_REUSE
    reuse_query = np.copy(base_vec)
    reuse_query[1] = 0.05
    res_reuse = index.search(reuse_query)
    assert res_reuse.action == SemanticRoutingAction.DIRECT_REUSE
    assert res_reuse.block_id == 5
    assert res_reuse.similarity >= 0.92

    # 2. Moderately similar query vector (similarity ~0.85 in [0.80, 0.92)) -> SPECULATIVE_DRAFT
    # An angle of ~30 degrees has cos(30 deg) ~ 0.866
    draft_query = np.zeros(dim, dtype=np.float32)
    draft_query[0] = 0.866
    draft_query[1] = 0.5
    res_draft = index.search(draft_query)
    assert res_draft.action == SemanticRoutingAction.SPECULATIVE_DRAFT
    assert res_draft.block_id == 5
    assert 0.80 <= res_draft.similarity < 0.92
    assert res_draft.continuation_tokens == [4, 5]

    # 3. Orthogonal query vector (similarity 0.0 < 0.80) -> CACHE_MISS
    miss_query = np.zeros(dim, dtype=np.float32)
    miss_query[10] = 1.0
    res_miss = index.search(miss_query)
    assert res_miss.action == SemanticRoutingAction.CACHE_MISS
    assert res_miss.similarity < 0.80


# ==============================================================================
# SpecCacheEngine End-to-End Hierarchical Routing & Eviction Tests
# ==============================================================================

def test_engine_two_tier_lookup_and_eviction_cascade():
    """Verify Radix Tier 1 exact match priority, Semantic Tier 2 fallback, and eviction cascade."""
    chunk_size = 64
    heads, dim = 4, 64
    emb_dim = 32

    engine = SpecCacheEngine(
        chunk_size=chunk_size,
        num_kv_heads=heads,
        head_dim=dim,
        max_blocks=2,  # Small pool to force eviction
        embedding_dim=emb_dim,
        force_fallback_semantic=True,
    )

    # Register Chunk 0
    toks0 = list(range(100, 100 + chunk_size))
    kq0, ks0, vq0, vs0 = _generate_synthetic_chunk(chunk_size, heads, dim, seed=10)
    emb0 = np.zeros(emb_dim, dtype=np.float32)
    emb0[0] = 1.0
    b0 = engine.register_chunk(toks0, cached_position=0, k_quant=kq0, k_scale=ks0, v_quant=vq0, v_scale=vs0, embedding=emb0)

    # Register Chunk 1
    toks1 = list(range(200, 200 + chunk_size))
    kq1, ks1, vq1, vs1 = _generate_synthetic_chunk(chunk_size, heads, dim, seed=20)
    emb1 = np.zeros(emb_dim, dtype=np.float32)
    emb1[1] = 1.0
    b1 = engine.register_chunk(toks1, cached_position=64, k_quant=kq1, k_scale=ks1, v_quant=vq1, v_scale=vs1, embedding=emb1)

    # 1. Query with exact tokens for Chunk 0 at new prompt position 128 (Delta p = 128 - 0 = +128)
    decisions = engine.lookup_chunks(prompt_tokens=toks0)
    assert len(decisions) == 1
    assert decisions[0].status == ChunkRoutingStatus.EXACT_REUSE
    assert decisions[0].block_id == b0
    assert decisions[0].delta_p == 0  # target_start_pos=0, cached_pos=0

    # 2. Register Chunk 2 -> Forces eviction of LRU block (b0 was touched, so b1 is evicted!)
    toks2 = list(range(300, 300 + chunk_size))
    kq2, ks2, vq2, vs2 = _generate_synthetic_chunk(chunk_size, heads, dim, seed=30)
    emb2 = np.zeros(emb_dim, dtype=np.float32)
    emb2[2] = 1.0
    b2 = engine.register_chunk(toks2, cached_position=128, k_quant=kq2, k_scale=ks2, v_quant=vq2, v_scale=vs2, embedding=emb2)

    # Chunk 1 was evicted from pinned pool -> Must now be a CACHE_MISS in Radix and Semantic indices!
    decisions_evicted = engine.lookup_chunks(prompt_tokens=toks1, chunk_embeddings=[emb1])
    assert decisions_evicted[0].status == ChunkRoutingStatus.CACHE_MISS


# ==============================================================================
# Async DMA Pipeline Numerical Parity Test
# ==============================================================================

def test_async_dma_pipeline_parity():
    """Verify that pipelined double-buffered async DMA matches synchronous reference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 64
    heads, dim = 8, 128
    emb_dim = 64

    engine = SpecCacheEngine(
        chunk_size=chunk_size,
        num_kv_heads=heads,
        head_dim=dim,
        max_blocks=8,
        embedding_dim=emb_dim,
        device=device,
    )

    num_chunks = 4
    all_tokens: List[int] = []
    chunk_data = []

    for c in range(num_chunks):
        toks = list(range(c * 1000, c * 1000 + chunk_size))
        all_tokens.extend(toks)
        kq, ks, vq, vs = _generate_synthetic_chunk(chunk_size, heads, dim, seed=100 + c)
        bid = engine.register_chunk(toks, cached_position=c * 64, k_quant=kq, k_scale=ks, v_quant=vq, v_scale=vs)
        chunk_data.append((bid, kq, ks, vq, vs))

    # Query all 4 chunks
    decisions = engine.lookup_chunks(prompt_tokens=all_tokens)
    assert len(decisions) == 4
    for d in decisions:
        assert d.status == ChunkRoutingStatus.EXACT_REUSE

    # Execute async DMA pipeline
    k_out_pipelined, v_out_pipelined = engine.execute_async_dma_and_rope(decisions)

    assert k_out_pipelined.shape == (num_chunks * chunk_size, heads, dim)
    assert v_out_pipelined.shape == (num_chunks * chunk_size, heads, dim)

    # Compute reference ground truth sequentially
    for i, dec in enumerate(decisions):
        bid, kq, ks, vq, vs = chunk_data[i]
        delta_p = dec.delta_p
        cos_d, sin_d = compute_rope_cos_sin_delta(torch.tensor([delta_p], device=device), dim, device=device)

        k_ref = pytorch_eager_dequant_rope_shift(kq.to(device), ks.to(device), cos_d, sin_d)
        v_ref = dequant_value_ref(vq.to(device), vs.to(device))

        k_slice = k_out_pipelined[i * chunk_size : (i + 1) * chunk_size]
        v_slice = v_out_pipelined[i * chunk_size : (i + 1) * chunk_size]

        torch.testing.assert_close(k_slice, k_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(v_slice, v_ref, atol=1e-2, rtol=1e-2)
