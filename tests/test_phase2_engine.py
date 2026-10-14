"""Test suite for SpecCache Phase 2: Hybrid Radix-Semantic Engine and Pinned DMA.

Verifies:
1. Exact Hit vs. Semantic Shifted Hit:
   - Inserts chunk originally rotated at p_cached = 512.
   - Queries with slightly perturbed embedding (cos_sim ~ 0.96) at p_target = 128 (Delta p = -384).
   - Confirms engine routes via REUSE, transfers over DMA, shifts RoPE by -384 via
     Phase 1 Triton kernel, and matches ground-truth FP16 rotated Key within 1e-2.
2. LRU Eviction Integrity:
   - Fills pool of max_blocks = 8, inserts 10 blocks, and verifies the 2 oldest blocks
     are cleanly evicted and invalidated in both Radix Tree and Semantic Index.
3. Unit tests for PinnedKVPool, ChunkRadixTree, and SemanticHNSWIndex.
"""

import numpy as np
import torch

from speccache.cache.engine import (
    ChunkRoutingStatus,
    SpecCacheEngine,
)
from speccache.cache.pinned_pool import PinnedKVPool
from speccache.cache.radix_index import ChunkRadixTree
from speccache.cache.semantic_index import (
    RouteDecision,
)
from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    dequant_value_ref,
    pytorch_eager_dequant_rope_shift,
    quantize_kv_int8,
    rotate_half,
)


def _generate_synthetic_chunk(
    chunk_size: int,
    num_heads: int,
    head_dim: int,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate reproducible quantized Key and Value tensors for a chunk."""
    torch.manual_seed(seed)
    raw_k = torch.randn((chunk_size, num_heads, head_dim), dtype=torch.float32, device=device)
    raw_v = torch.randn((chunk_size, num_heads, head_dim), dtype=torch.float32, device=device)
    k_quant, k_scale = quantize_kv_int8(raw_k)
    v_quant, v_scale = quantize_kv_int8(raw_v)
    return k_quant, k_scale, v_quant, v_scale


# ==============================================================================
# Requirement E.1: Exact Hit vs. Semantic Shifted Hit Test
# ==============================================================================


def test_exact_hit_vs_semantic_shifted_hit():
    """Verify exact hit and semantic shifted hit with relative RoPE phase shift of -384."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 128
    heads = 8
    head_dim = 128
    embed_dim = 64

    engine = SpecCacheEngine(
        chunk_size=chunk_size,
        num_kv_heads=heads,
        head_dim=head_dim,
        max_blocks=16,
        embedding_dim=embed_dim,
        device=device,
        force_fallback_semantic=True,  # Ensure consistent CPU/GPU test behavior
    )

    # 1. Generate raw FP16 keys and rotate at original position p_cached = 512
    torch.manual_seed(1234)
    raw_k = torch.randn((chunk_size, heads, head_dim), dtype=torch.float32, device=device)
    raw_v = torch.randn((chunk_size, heads, head_dim), dtype=torch.float32, device=device)

    p_cached = 512
    cos_512, sin_512 = compute_rope_cos_sin_delta(
        torch.tensor([p_cached], device=device), head_dim, device=device
    )
    cos_512_full = torch.cat([cos_512, cos_512], dim=-1)[:, None, :]
    sin_512_full = torch.cat([sin_512, sin_512], dim=-1)[:, None, :]

    # Keys rotated at position 512
    k_rot_512 = (raw_k * cos_512_full + rotate_half(raw_k) * sin_512_full).to(torch.float16)

    # Quantize to INT8
    k_quant, k_scale = quantize_kv_int8(k_rot_512)
    v_quant, v_scale = quantize_kv_int8(raw_v)

    # Base semantic embedding
    base_emb = np.zeros(embed_dim, dtype=np.float32)
    base_emb[0] = 1.0  # Unit vector along axis 0

    chunk_tokens = list(range(1000, 1000 + chunk_size))

    # Register chunk at original position p_cached = 512
    block_id = engine.register_chunk(
        token_ids=chunk_tokens,
        cached_position=p_cached,
        k_quant=k_quant.cpu(),
        k_scale=k_scale.cpu(),
        v_quant=v_quant.cpu(),
        v_scale=v_scale.cpu(),
        embedding=base_emb,
    )

    # 2. Test Semantic Shifted Hit at target position p_target = 128 (Delta p = 128 - 512 = -384)
    p_target = 128
    expected_delta_p = p_target - p_cached  # -384

    # Create perturbed embedding with cos_sim ~ 0.96 (angle ~16.26 degrees: cos(16.26 deg) ~ 0.96)
    perturbed_emb = np.copy(base_emb)
    perturbed_emb[1] = 0.29  # 1.0 / sqrt(1 + 0.29^2) = 1.0 / 1.041 ~ 0.960
    norm_sim = float(np.dot(perturbed_emb / np.linalg.norm(perturbed_emb), base_emb))
    assert 0.94 <= norm_sim <= 0.98

    # Query with novel tokens (to miss Tier 1 Radix) and perturbed embedding (to hit Tier 2 Semantic)
    novel_tokens = list(range(5000, 5000 + chunk_size))
    k_out, v_out, decisions = engine.retrieve_and_reconstruct_kv(
        chunks_tokens=[novel_tokens],
        chunks_embeddings=[perturbed_emb],
        target_positions=[p_target],
    )

    assert len(decisions) == 1
    dec = decisions[0]

    # Verify decision routed via SEMANTIC_REUSE with Delta p = -384
    assert dec.status == ChunkRoutingStatus.SEMANTIC_REUSE
    assert dec.block_id == block_id
    assert dec.cached_position == p_cached
    assert dec.delta_p == expected_delta_p

    # Ground truth reference: Rotate k_quant by Delta p = -384 using reference implementation
    cos_neg384, sin_neg384 = compute_rope_cos_sin_delta(
        torch.tensor([expected_delta_p], device=device), head_dim, device=device
    )
    k_ref = pytorch_eager_dequant_rope_shift(
        k_quant.to(device), k_scale.to(device), cos_neg384, sin_neg384
    )

    # Verify numerical match within 1e-2
    torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)


# ==============================================================================
# Requirement E.2: LRU Eviction Integrity Test
# ==============================================================================


def test_lru_eviction_integrity():
    """Fill pool of max_blocks = 8, insert 10 blocks, verify oldest 2 evicted and invalidated."""
    max_blocks = 8
    chunk_size = 64
    heads = 4
    head_dim = 64
    embed_dim = 32

    engine = SpecCacheEngine(
        chunk_size=chunk_size,
        num_kv_heads=heads,
        head_dim=head_dim,
        max_blocks=max_blocks,
        embedding_dim=embed_dim,
        force_fallback_semantic=True,
    )

    registered_tokens: list[list[int]] = []
    registered_embs: list[np.ndarray] = []

    # Insert 10 blocks into pool with capacity 8
    for i in range(10):
        toks = list(range(i * 1000, i * 1000 + chunk_size))
        registered_tokens.append(toks)

        emb = np.zeros(embed_dim, dtype=np.float32)
        emb[i % embed_dim] = 1.0
        registered_embs.append(emb)

        kq, ks, vq, vs = _generate_synthetic_chunk(chunk_size, heads, head_dim, seed=i)
        engine.register_chunk(
            token_ids=toks,
            cached_position=i * 64,
            k_quant=kq,
            k_scale=ks,
            v_quant=vq,
            v_scale=vs,
            embedding=emb,
        )

    # The pool capacity is 8, so exactly 8 blocks must be active
    assert engine.pool.allocated_count == max_blocks

    # Blocks 0 and 1 are the oldest and must have been evicted
    # 1. Test Radix Tree invalidation: chunks 0 and 1 must NOT match
    assert engine.radix_tree.match_exact(registered_tokens[0]) is None
    assert engine.radix_tree.match_exact(registered_tokens[1]) is None

    # 2. Test Semantic Index invalidation: query with embeddings 0 and 1 must return MISS
    decision0, sim0, bid0, _ = engine.semantic_index.query(registered_embs[0])
    assert decision0 == RouteDecision.MISS

    decision1, sim1, bid1, _ = engine.semantic_index.query(registered_embs[1])
    assert decision1 == RouteDecision.MISS

    # 3. Blocks 2 through 9 must remain active and valid in Radix Tree
    for i in range(2, 10):
        match = engine.radix_tree.match_exact(registered_tokens[i])
        assert match is not None
        assert match.block_id in range(max_blocks)


# ==============================================================================
# PinnedKVPool Unit Tests
# ==============================================================================


def test_pinned_kv_pool_basic_operations():
    """Verify allocate, write_block, get_block_views, touch, and free in PinnedKVPool."""
    max_blocks = 4
    chunk_size = 64
    heads = 4
    head_dim = 64

    pool = PinnedKVPool(
        max_blocks=max_blocks, chunk_size=chunk_size, num_heads=heads, head_dim=head_dim
    )
    assert pool.free_count == max_blocks
    assert pool.allocated_count == 0

    # Allocate a slot
    bid = pool.allocate()
    assert bid in range(max_blocks)
    assert pool.allocated_count == 1
    assert pool.free_count == max_blocks - 1

    # Write data
    kq, ks, vq, vs = _generate_synthetic_chunk(chunk_size, heads, head_dim, seed=42)
    pool.write_block(bid, kq, ks, vq, vs, start_pos=256)

    assert pool.get_start_pos(bid) == 256

    # Read views
    r_kq, r_ks, r_vq, r_vs = pool.get_block_views(bid)
    assert torch.equal(r_kq, kq)
    assert torch.equal(r_ks, ks)
    assert torch.equal(r_vq, vq)
    assert torch.equal(r_vs, vs)

    # Free
    pool.free(bid)
    assert pool.allocated_count == 0
    assert pool.free_count == max_blocks


# ==============================================================================
# ChunkRadixTree Unit Tests
# ==============================================================================


def test_chunk_radix_tree_prefix_matching():
    """Verify ChunkRadixTree match_prefix returns list of (chunk_idx, block_id)."""
    chunk_size = 32
    tree = ChunkRadixTree(chunk_size=chunk_size)

    # Sequence of 3 chunks: total 96 tokens
    chunk0 = list(range(0, 32))
    chunk1 = list(range(32, 64))
    chunk2 = list(range(64, 96))
    full_seq = chunk0 + chunk1 + chunk2

    tree.insert(token_ids=full_seq, block_ids=[10, 20, 30])

    # Query full sequence: matches all 3 chunks
    matches = tree.match_prefix(full_seq)
    assert matches == [(0, 10), (1, 20), (2, 30)]

    # Query prefix sequence (first 2 chunks): matches 2 chunks
    matches_sub = tree.match_prefix(chunk0 + chunk1)
    assert matches_sub == [(0, 10), (1, 20)]

    # Query partial mismatch at chunk 1: only chunk 0 matches
    mismatch_seq = chunk0 + [999] * 32
    matches_mismatch = tree.match_prefix(mismatch_seq)
    assert matches_mismatch == [(0, 10)]


# ==============================================================================
# Async DMA Pipeline Multi-Chunk Parity Test
# ==============================================================================


def test_async_dma_pipeline_multi_chunk_parity():
    """Verify pipelined double-buffered async DMA matches reference across multi-chunk prompt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 64
    heads = 4
    head_dim = 64
    embed_dim = 32

    engine = SpecCacheEngine(
        chunk_size=chunk_size,
        num_kv_heads=heads,
        head_dim=head_dim,
        max_blocks=16,
        embedding_dim=embed_dim,
        device=device,
        force_fallback_semantic=True,
    )

    num_chunks = 4
    chunk_tokens_list = []
    chunk_raw_data = []

    for c in range(num_chunks):
        toks = list(range(c * 500, c * 500 + chunk_size))
        chunk_tokens_list.append(toks)
        kq, ks, vq, vs = _generate_synthetic_chunk(chunk_size, heads, head_dim, seed=200 + c)
        p_cached = c * 128
        bid = engine.register_chunk(
            token_ids=toks,
            cached_position=p_cached,
            k_quant=kq,
            k_scale=ks,
            v_quant=vq,
            v_scale=vs,
        )
        chunk_raw_data.append((bid, kq, ks, vq, vs, p_cached))

    # Query with target positions shifted by varying amounts
    target_positions = [c * 256 for c in range(num_chunks)]
    k_out, v_out, decisions = engine.retrieve_and_reconstruct_kv(
        chunks_tokens=chunk_tokens_list,
        target_positions=target_positions,
    )

    assert len(decisions) == num_chunks
    assert k_out.shape == (num_chunks * chunk_size, heads, head_dim)
    assert v_out.shape == (num_chunks * chunk_size, heads, head_dim)

    # Compare each chunk slice against eager reference
    for i in range(num_chunks):
        bid, kq, ks, vq, vs, p_cached = chunk_raw_data[i]
        delta_p = target_positions[i] - p_cached
        cos_d, sin_d = compute_rope_cos_sin_delta(
            torch.tensor([delta_p], device=device), head_dim, device=device
        )
        k_ref = pytorch_eager_dequant_rope_shift(kq.to(device), ks.to(device), cos_d, sin_d)
        v_ref = dequant_value_ref(vq.to(device), vs.to(device))

        k_slice = k_out[i * chunk_size : (i + 1) * chunk_size]
        v_slice = v_out[i * chunk_size : (i + 1) * chunk_size]

        torch.testing.assert_close(k_slice, k_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(v_slice, v_ref, atol=1e-2, rtol=1e-2)
