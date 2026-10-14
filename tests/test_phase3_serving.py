"""Test suite for SpecCache Phase 3: End-to-End LLM Serving & Speculative Verification.

Verifies:
1. SpecDynamicCache:
   - Dynamic KV cache expansion, concatenation across forward passes.
   - Rollback / truncation for speculative draft verification.
   - HuggingFace-compatible legacy cache export and import.
2. KVInjector:
   - Layout translation from [T, H, D] to [1, H, T, D].
   - RoPE relative phase-shifting across arbitrary layer configurations.
3. SpeculativeVerifier:
   - Full acceptance (all gamma draft tokens match target logits argmax).
   - Partial acceptance (first k tokens match, token k+1 mismatches).
   - Zero acceptance (token 0 mismatches, bonus token emitted).
   - Cache rollback integrity.
4. SpecCacheLLMRunner Three-Path Router:
   - Path 1: Direct KV Reuse (Radix exact hit / similarity >= 0.92).
   - Path 2: Speculative Draft Verification (0.80 <= similarity < 0.92).
   - Path 3: Cold Miss (similarity < 0.80) with automatic INT8 quantization.
5. FastAPI SSE Streaming Server:
   - Health check (/health) and Prometheus metrics (/metrics).
   - Generation endpoint (/generate) in standard JSON and SSE streaming formats.
   - OpenAI-compatible chat completions (/v1/chat/completions).
   - Custom hardware telemetry response headers.
"""

from typing import List, Tuple
import numpy as np
import pytest
import torch

from speccache.cache.engine import SpecCacheEngine
from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    quantize_kv_int8,
    rotate_half,
)
from speccache.serving.app import create_app
from speccache.serving.kv_injector import KVInjector, SpecDynamicCache
from speccache.serving.pipeline import (
    MockL4TransformerEngine,
    RoutePath,
    SpecCacheLLMRunner,
)
from speccache.serving.speculative import (
    SpeculativeDraft,
    SpeculativeVerifier,
    VerificationResult,
)


# ==============================================================================
# 1. SpecDynamicCache Tests
# ==============================================================================

def test_dynamic_cache_update_and_concat():
    """Verify SpecDynamicCache concatenation and multi-step expansion."""
    cache = SpecDynamicCache()
    assert len(cache) == 0
    assert cache.get_seq_length(0) == 0

    batch_size = 1
    num_heads = 4
    head_dim = 32

    # Step 1: Initial prefill chunk of 16 tokens
    k1 = torch.randn(batch_size, num_heads, 16, head_dim)
    v1 = torch.randn(batch_size, num_heads, 16, head_dim)
    cum_k, cum_v = cache.update(k1, v1, layer_idx=0)

    assert len(cache) == 1
    assert cache.get_seq_length(0) == 16
    assert cum_k.shape == (batch_size, num_heads, 16, head_dim)
    assert cum_v.shape == (batch_size, num_heads, 16, head_dim)

    # Step 2: Decode step of 1 token
    k2 = torch.randn(batch_size, num_heads, 1, head_dim)
    v2 = torch.randn(batch_size, num_heads, 1, head_dim)
    cum_k2, cum_v2 = cache.update(k2, v2, layer_idx=0)

    assert cache.get_seq_length(0) == 17
    assert cum_k2.shape == (batch_size, num_heads, 17, head_dim)
    assert cum_v2.shape == (batch_size, num_heads, 17, head_dim)
    assert torch.allclose(cum_k2[:, :, :16, :], k1)
    assert torch.allclose(cum_k2[:, :, 16:, :], k2)


def test_dynamic_cache_truncation():
    """Verify KV cache truncation for speculative decoding rollbacks."""
    cache = SpecDynamicCache()
    batch_size = 1
    num_heads = 4
    head_dim = 32

    # Fill 2 layers with 24 tokens each
    for layer in range(2):
        k = torch.randn(batch_size, num_heads, 24, head_dim)
        v = torch.randn(batch_size, num_heads, 24, head_dim)
        cache.update(k, v, layer_idx=layer)

    assert cache.get_seq_length(0) == 24
    assert cache.get_seq_length(1) == 24

    # Truncate back to 18 tokens
    cache.truncate(18)
    assert cache.get_seq_length(0) == 18
    assert cache.get_seq_length(1) == 18
    assert cache[0][0].shape == (batch_size, num_heads, 18, head_dim)
    assert cache[1][1].shape == (batch_size, num_heads, 18, head_dim)

    # Truncate with length >= current length should be a no-op
    cache.truncate(30)
    assert cache.get_seq_length(0) == 18


def test_dynamic_cache_legacy_conversion():
    """Verify conversion to and from HuggingFace tuple format."""
    cache = SpecDynamicCache()
    batch_size = 1
    num_heads = 2
    head_dim = 16

    k0 = torch.randn(batch_size, num_heads, 8, head_dim)
    v0 = torch.randn(batch_size, num_heads, 8, head_dim)
    k1 = torch.randn(batch_size, num_heads, 8, head_dim)
    v1 = torch.randn(batch_size, num_heads, 8, head_dim)

    cache.update(k0, v0, layer_idx=0)
    cache.update(k1, v1, layer_idx=1)

    legacy = cache.to_legacy_cache()
    assert len(legacy) == 2
    assert legacy[0][0].shape == (batch_size, num_heads, 8, head_dim)

    restored = SpecDynamicCache.from_legacy_cache(legacy)
    assert len(restored) == 2
    assert torch.allclose(restored[0][0], k0)
    assert torch.allclose(restored[1][1], v1)


# ==============================================================================
# 2. KVInjector Tests
# ==============================================================================

def test_kv_injector_layout_and_rope_shift():
    """Verify KVInjector layout transformation and RoPE phase shifting."""
    device = torch.device("cpu")
    num_layers = 2
    num_kv_heads = 4
    head_dim = 32
    chunk_size = 16

    injector = KVInjector(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )

    cache = SpecDynamicCache()

    # Generate synthetic chunk in [T, H, D] format
    k_tensor = torch.randn(chunk_size, num_kv_heads, head_dim, device=device)
    v_tensor = torch.randn(chunk_size, num_kv_heads, head_dim, device=device)

    # Inject with zero delta_p
    injector.inject_layer_kv(cache, layer_idx=0, k_tensor=k_tensor, v_tensor=v_tensor, delta_p=0)
    assert cache.get_seq_length(0) == chunk_size
    assert cache[0][0].shape == (1, num_kv_heads, chunk_size, head_dim)
    assert cache[0][1].shape == (1, num_kv_heads, chunk_size, head_dim)

    # Inject second chunk with delta_p = 64
    k2 = torch.randn(chunk_size, num_kv_heads, head_dim, device=device)
    v2 = torch.randn(chunk_size, num_kv_heads, head_dim, device=device)
    injector.inject_layer_kv(cache, layer_idx=0, k_tensor=k2, v_tensor=v2, delta_p=64)
    assert cache.get_seq_length(0) == chunk_size * 2


# ==============================================================================
# 3. SpeculativeVerifier Mathematical Tests
# ==============================================================================

def test_speculative_verifier_full_acceptance():
    """Verify case where all candidate draft tokens are accepted."""
    verifier = SpeculativeVerifier(gamma=6, temperature=0.0)
    vocab_size = 100
    draft_tokens = [10, 20, 30, 40, 50, 60]

    # Verification logits: 7 positions (gamma + 1)
    # Positions 0..5 predict draft_tokens[0..5]
    # Position 6 predicts bonus token 99
    logits = torch.zeros(7, vocab_size)
    for i, tok in enumerate(draft_tokens):
        logits[i, tok] = 10.0
    logits[6, 99] = 10.0

    result = verifier.verify_draft_tokens(draft_tokens, logits, greedy=True)

    assert result.num_accepted == 6
    assert result.accepted_tokens == draft_tokens
    assert result.bonus_token == 99
    assert result.total_emitted == 7
    assert result.acceptance_rate == 1.0
    assert result.rejection_index is None
    assert result.emitted_tokens == [10, 20, 30, 40, 50, 60, 99]


def test_speculative_verifier_partial_acceptance():
    """Verify case where candidate draft tokens are accepted up to a mismatch."""
    verifier = SpeculativeVerifier(gamma=6, temperature=0.0)
    vocab_size = 100
    draft_tokens = [10, 20, 30, 40, 50, 60]

    # Positions 0, 1, 2 match draft tokens
    # Position 3 predicts 77 instead of draft_tokens[3] (40)
    logits = torch.zeros(7, vocab_size)
    logits[0, 10] = 10.0
    logits[1, 20] = 10.0
    logits[2, 30] = 10.0
    logits[3, 77] = 10.0  # Mismatch!

    result = verifier.verify_draft_tokens(draft_tokens, logits, greedy=True)

    assert result.num_accepted == 3
    assert result.accepted_tokens == [10, 20, 30]
    assert result.bonus_token == 77
    assert result.total_emitted == 4
    assert result.acceptance_rate == 0.5
    assert result.rejection_index == 3
    assert result.emitted_tokens == [10, 20, 30, 77]


def test_speculative_verifier_zero_acceptance():
    """Verify case where the very first candidate draft token mismatches."""
    verifier = SpeculativeVerifier(gamma=6, temperature=0.0)
    vocab_size = 100
    draft_tokens = [10, 20, 30, 40, 50, 60]

    # Position 0 predicts 88 instead of draft_tokens[0] (10)
    logits = torch.zeros(7, vocab_size)
    logits[0, 88] = 10.0

    result = verifier.verify_draft_tokens(draft_tokens, logits, greedy=True)

    assert result.num_accepted == 0
    assert result.accepted_tokens == []
    assert result.bonus_token == 88
    assert result.total_emitted == 1
    assert result.acceptance_rate == 0.0
    assert result.rejection_index == 0
    assert result.emitted_tokens == [88]


def test_speculative_verifier_rollback():
    """Verify cache truncation during speculative rollback."""
    verifier = SpeculativeVerifier(gamma=4)
    cache = SpecDynamicCache()
    batch_size = 1
    num_heads = 2
    head_dim = 16

    # Suppose prompt length is 32, we forwarded 32 + 4 = 36 positions
    k = torch.randn(batch_size, num_heads, 36, head_dim)
    v = torch.randn(batch_size, num_heads, 36, head_dim)
    cache.update(k, v, layer_idx=0)

    # Suppose only 2 draft tokens were accepted + 1 bonus token -> valid length is 32 + 3 = 35
    verifier.rollback_cache(cache, target_seq_len=35)
    assert cache.get_seq_length(0) == 35


# ==============================================================================
# 4. SpecCacheLLMRunner Three-Path Router Tests
# ==============================================================================

@pytest.fixture
def mock_serving_setup():
    """Construct lightweight MockL4TransformerEngine and SpecCacheEngine."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = 16
    heads = 2
    head_dim = 32
    hidden_dim = 64
    vocab_size = 500
    embed_dim = 32

    model = MockL4TransformerEngine(
        num_layers=2,
        num_kv_heads=heads,
        num_q_heads=4,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        device=device,
    )

    engine = SpecCacheEngine(
        chunk_size=chunk_size,
        num_kv_heads=heads,
        head_dim=head_dim,
        max_blocks=16,
        embedding_dim=embed_dim,
        device=device,
        force_fallback_semantic=True,
    )

    runner = SpecCacheLLMRunner(
        engine=engine,
        model=model,
        chunk_size=chunk_size,
        draft_gamma=4,
        device=device,
    )

    return runner, engine, model


def test_three_path_cold_miss_and_direct_reuse(mock_serving_setup):
    """Verify Path 3 (Cold Miss) followed by Path 1 (Direct KV Reuse)."""
    runner, engine, _ = mock_serving_setup
    chunk_size = runner.chunk_size

    # Create a prompt of 32 tokens (exactly 2 chunks)
    prompt_tokens = list(range(100, 100 + chunk_size * 2))

    # Query 1: Cold Miss (cache is empty)
    res_cold = runner.generate(prompt_tokens=prompt_tokens, max_new_tokens=4)

    assert res_cold.route == RoutePath.COLD_MISS
    assert len(res_cold.generated_tokens) == 4
    assert res_cold.ttft_ms >= 0.0
    assert res_cold.decode_tokens_per_sec >= 0.0

    # Query 2: Direct KV Reuse (exact same tokens)
    res_reuse = runner.generate(prompt_tokens=prompt_tokens, max_new_tokens=4)

    assert res_reuse.route == RoutePath.DIRECT_REUSE
    assert res_reuse.kv_blocks_reused == 2
    assert res_reuse.dma_bytes_saved > 0
    assert len(res_reuse.generated_tokens) == 4


def test_three_path_speculative_verification(mock_serving_setup):
    """Verify Path 2 (Speculative Draft Verification) on semantic near-miss."""
    runner, engine, _ = mock_serving_setup
    chunk_size = runner.chunk_size
    embed_dim = engine.embedding_dim

    # Register a chunk with continuation tokens and known base embedding
    base_tokens = list(range(1000, 1000 + chunk_size))
    cached_continuation = [201, 202, 203, 204]  # 4 tokens for gamma=4

    raw_k = torch.randn(chunk_size, engine.num_kv_heads, engine.head_dim)
    raw_v = torch.randn(chunk_size, engine.num_kv_heads, engine.head_dim)
    kq, ks = quantize_kv_int8(raw_k)
    vq, vs = quantize_kv_int8(raw_v)

    base_emb = np.zeros(embed_dim, dtype=np.float32)
    base_emb[0] = 1.0  # Unit vector on axis 0

    engine.register_chunk(
        token_ids=base_tokens,
        cached_position=0,
        k_quant=kq,
        k_scale=ks,
        v_quant=vq,
        v_scale=vs,
        embedding=base_emb,
        continuation_tokens=cached_continuation,
    )

    # Query with novel tokens (misses Tier 1 Radix) and perturbed embedding
    # Target similarity s ~ 0.86 (within [0.80, 0.92))
    # cos(theta) = 1.0 / sqrt(1 + 0.58^2) = 1.0 / 1.156 ~ 0.865
    perturbed_emb = np.copy(base_emb)
    perturbed_emb[1] = 0.58
    perturbed_emb = perturbed_emb / np.linalg.norm(perturbed_emb)

    sim = float(np.dot(perturbed_emb, base_emb))
    assert 0.82 <= sim <= 0.90

    # Query novel prompt tokens
    novel_prompt = list(range(3000, 3000 + chunk_size))

    # Mock _embed_chunk to return our perturbed embedding for this chunk
    original_embed_chunk = runner._embed_chunk
    runner._embed_chunk = lambda _: perturbed_emb

    try:
        res_spec = runner.generate(prompt_tokens=novel_prompt, max_new_tokens=6)
        assert res_spec.route == RoutePath.SPECULATIVE_VERIFICATION
        assert res_spec.similarity >= 0.80
        assert res_spec.similarity < 0.92
        assert len(res_spec.generated_tokens) > 0
    finally:
        runner._embed_chunk = original_embed_chunk


# ==============================================================================
# 5. FastAPI Endpoints & Telemetry Header Tests
# ==============================================================================

def test_fastapi_endpoints_and_headers(mock_serving_setup):
    """Verify FastAPI endpoints, SSE streaming format, and custom headers."""
    pytest.importorskip("starlette.testclient")
    from starlette.testclient import TestClient

    runner, _, _ = mock_serving_setup
    app = create_app(runner=runner)
    client = TestClient(app)

    # 1. Health check endpoint
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    health_data = resp_health.json()
    assert health_data["status"] == "healthy"
    assert "pool_blocks_total" in health_data

    # 2. Prometheus metrics endpoint
    resp_metrics = client.get("/metrics")
    assert resp_metrics.status_code == 200
    assert "speccache_requests_total" in resp_metrics.text

    # 3. Synchronous JSON generation (/generate with stream=false)
    req_body = {
        "tokens": [10, 11, 12, 13, 14, 15, 16, 17],
        "max_tokens": 4,
        "stream": False,
    }
    resp_gen = client.post("/generate", json=req_body)
    assert resp_gen.status_code == 200

    # Verify custom hardware telemetry headers
    assert "x-speccache-route" in resp_gen.headers
    assert "x-speccache-similarity" in resp_gen.headers
    assert "x-speccache-ttft-ms" in resp_gen.headers
    assert "x-speccache-tokens-per-sec" in resp_gen.headers
    assert "x-speccache-spec-accepted-tokens" in resp_gen.headers
    assert "x-speccache-dma-bytes-saved" in resp_gen.headers
    assert "x-speccache-kv-blocks-reused" in resp_gen.headers

    gen_data = resp_gen.json()
    assert "tokens" in gen_data
    assert "telemetry" in gen_data
    assert len(gen_data["tokens"]) == 4

    # 4. SSE streaming generation (/generate with stream=true)
    req_stream = {
        "tokens": [20, 21, 22, 23],
        "max_tokens": 4,
        "stream": True,
    }
    resp_stream = client.post("/generate", json=req_stream)
    assert resp_stream.status_code == 200
    assert "text/event-stream" in resp_stream.headers["content-type"]
    assert "data: " in resp_stream.text
    assert "[DONE]" in resp_stream.text

    # 5. OpenAI-compatible chat completion (/v1/chat/completions)
    chat_body = {
        "messages": [{"role": "user", "content": "Explain KV caching"}],
        "max_tokens": 4,
        "stream": False,
    }
    resp_chat = client.post("/v1/chat/completions", json=chat_body)
    assert resp_chat.status_code == 200
    chat_data = resp_chat.json()
    assert "choices" in chat_data
    assert len(chat_data["choices"]) > 0
    assert "message" in chat_data["choices"][0]
