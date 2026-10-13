"""SpecCacheEngine: Integrated Hybrid Radix-Semantic Lookup & Pinned DMA Engine.

Integrates:
- PinnedKVPool: Page-locked host memory slab allocator with O(1) LRU eviction.
- ChunkRadixTree: Token-level prefix tree for Tier 1 O(L) exact chunk matching.
- SemanticHNSWIndex: FAISS HNSW vector search for Tier 2 semantic matching with
  threshold routing (EXACT_OR_REUSE >= 0.92, SPECULATIVE [0.80, 0.92), MISS < 0.80).
- Phase 1 Triton Kernels: Fused INT8 dequantization + relative RoPE phase-shifting.
- Asynchronous Double-Buffered DMA Pipeline: Overlapped PCIe host-to-device transfers
  and Triton kernel execution on dedicated CUDA streams.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from speccache.cache.pinned_pool import OriginalPositionMetadata, PinnedKVPool
from speccache.cache.radix_index import ChunkRadixTree, RadixMatchResult
from speccache.cache.semantic_index import (
    RouteDecision,
    SemanticHNSWIndex,
    SemanticLookupResult,
)
from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    dequant_value_ref,
    pytorch_eager_dequant_rope_shift,
)

try:
    from speccache.kernels.triton_rope_kv import (
        TRITON_AVAILABLE,
        dequant_value,
        triton_fused_k_dequant_rope_shift,
    )
except ImportError:
    TRITON_AVAILABLE = False


class ChunkRoutingStatus(str, Enum):
    """Routing state assigned to each prompt chunk."""
    EXACT_REUSE = "EXACT_REUSE"              # Tier 1 exact Radix hit
    SEMANTIC_REUSE = "SEMANTIC_REUSE"        # Tier 2 cosine similarity >= 0.92
    SPECULATIVE_DRAFT = "SPECULATIVE_DRAFT"  # Tier 2 0.80 <= similarity < 0.92
    CACHE_MISS = "CACHE_MISS"                # Similarity < 0.80


@dataclass
class ChunkRoutingDecision:
    """Routing decision and transfer metadata for a single prompt chunk."""
    chunk_index: int
    target_start_pos: int
    target_end_pos: int
    token_ids: Tuple[int, ...]
    status: ChunkRoutingStatus
    similarity: float = 0.0
    block_id: Optional[int] = None
    cached_position: Optional[int] = None
    delta_p: int = 0
    continuation_tokens: List[int] = field(default_factory=list)


class SpecCacheEngine:
    """Orchestration engine for SpecCache Phase 2."""

    def __init__(
        self,
        chunk_size: int = 128,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_blocks: int = 1024,
        embedding_dim: int = 128,
        tau_reuse: float = 0.92,
        tau_spec: float = 0.80,
        device: Optional[torch.device] = None,
        force_fallback_semantic: bool = False,
    ) -> None:
        """Initialize the SpecCache engine.

        Args:
            chunk_size: Token count per chunk (e.g., 128, 256).
            num_kv_heads: Number of KV heads (e.g. 8 for Llama-3-8B GQA).
            head_dim: Head dimension D (e.g. 128).
            max_blocks: Maximum chunks stored in PinnedKVPool.
            embedding_dim: Dimensionality of semantic chunk embedding vectors.
            tau_reuse: Cosine similarity threshold for direct KV reuse (0.92).
            tau_spec: Cosine similarity threshold for speculative drafting (0.80).
            device: Target GPU device.
            force_fallback_semantic: Use PyTorch fallback index instead of FAISS.
        """
        self.chunk_size = chunk_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_blocks = max_blocks
        self.embedding_dim = embedding_dim
        self.tau_reuse = tau_reuse
        self.tau_spec = tau_spec
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Pinned Host Memory Slab Allocator with LRU Eviction
        self.pool = PinnedKVPool(
            max_blocks=max_blocks,
            chunk_size=chunk_size,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            on_evict_callback=self._handle_lru_eviction,
        )

        # 2. Tier 1: Token-Level Chunk Radix Tree
        self.radix_tree = ChunkRadixTree(chunk_size=chunk_size)

        # 3. Tier 2: FAISS HNSW Semantic Vector Index
        self.semantic_index = SemanticHNSWIndex(
            embed_dim=embedding_dim,
            tau_reuse=tau_reuse,
            tau_spec=tau_spec,
            force_fallback=force_fallback_semantic,
        )

        # 4. Asynchronous Transfer Streams and Double-Buffered Staging Slabs
        self._is_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        if self._is_cuda:
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            self.compute_stream = torch.cuda.Stream(device=self.device)
            self._stage_k_quant = [
                torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
                torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
            ]
            self._stage_k_scale = [
                torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
                torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
            ]
            self._stage_v_quant = [
                torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
                torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
            ]
            self._stage_v_scale = [
                torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
                torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
            ]
        else:
            self.transfer_stream = None
            self.compute_stream = None
            self._stage_k_quant = None
            self._stage_k_scale = None
            self._stage_v_quant = None
            self._stage_v_scale = None

    def _handle_lru_eviction(self, block_id: int) -> None:
        """Cascade eviction from pinned pool down into Radix and Semantic indices."""
        self.radix_tree.evict_block(block_id)
        self.semantic_index.remove(block_id)

    def register_chunk(
        self,
        token_ids: Sequence[int],
        cached_position: int,
        k_quant: torch.Tensor,
        k_scale: torch.Tensor,
        v_quant: torch.Tensor,
        v_scale: torch.Tensor,
        embedding: Optional[Union[np.ndarray, torch.Tensor]] = None,
        continuation_tokens: Optional[List[int]] = None,
    ) -> int:
        """Register a new quantized INT8 KV chunk into the host cache and indices.

        Args:
            token_ids: Sequence of integer token IDs (length equal to chunk_size).
            cached_position: Token position p_cached when originally rotated.
            k_quant: INT8 quantized keys [chunk_size, num_heads, head_dim].
            k_scale: FP16 scales [chunk_size, num_heads, 1].
            v_quant: INT8 quantized values [chunk_size, num_heads, head_dim].
            v_scale: FP16 scales [chunk_size, num_heads, 1].
            embedding: Optional semantic embedding vector.
            continuation_tokens: Optional draft tokens.

        Returns:
            block_id: Assigned block index in PinnedKVPool.
        """
        tokens = tuple(token_ids)
        if len(tokens) != self.chunk_size:
            raise ValueError(
                f"token_ids length ({len(tokens)}) must equal chunk_size ({self.chunk_size})."
            )

        block_id = self.pool.allocate()

        self.pool.write_block(
            block_id=block_id,
            k_q=k_quant,
            k_s=k_scale,
            v_q=v_quant,
            v_s=v_scale,
            start_pos=cached_position,
            token_ids=tokens,
        )

        self.radix_tree.insert(
            token_ids=tokens,
            block_ids=block_id,
            cached_position=cached_position,
        )

        if embedding is not None:
            self.semantic_index.add(
                embedding=embedding,
                block_id=block_id,
                cached_position=cached_position,
                token_ids=tokens,
                continuation_tokens=continuation_tokens,
            )

        return block_id

    def retrieve_and_reconstruct_kv(
        self,
        chunks_tokens: Sequence[Sequence[int]],
        chunks_embeddings: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        target_positions: Optional[Sequence[int]] = None,
        cos_table: Optional[torch.Tensor] = None,
        sin_table: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[ChunkRoutingDecision]]:
        """Retrieve cached blocks via two-tier routing and reconstruct GPU FP16 KV caches.

        Executes:
        1. Tier 1 Radix exact match and Tier 2 Semantic HNSW lookup per chunk.
        2. Computes relative displacement Delta p = p_target - p_cached.
        3. Executes pipelined double-buffered asynchronous PCIe DMA + Triton kernel.

        Args:
            chunks_tokens: List of token sequences (each of length chunk_size).
            chunks_embeddings: Optional list of semantic embedding vectors.
            target_positions: List of target token start positions for each chunk.
            cos_table: Optional precomputed RoPE cos table.
            sin_table: Optional precomputed RoPE sin table.

        Returns:
            Tuple of (k_cache, v_cache, decisions).
        """
        num_chunks = len(chunks_tokens)
        if target_positions is None:
            target_positions = [c * self.chunk_size for c in range(num_chunks)]

        decisions: List[ChunkRoutingDecision] = []

        for i in range(num_chunks):
            c_toks = tuple(chunks_tokens[i])
            p_target = target_positions[i]
            emb = chunks_embeddings[i] if chunks_embeddings is not None and i < len(chunks_embeddings) else None

            # ------------------------------------------------------------------
            # Tier 1: Token-Level Radix Tree Exact Match
            # ------------------------------------------------------------------
            radix_hit = self.radix_tree.match_exact(c_toks)
            if radix_hit is not None:
                p_cached = self.pool.get_start_pos(radix_hit.block_id)
                delta_p = p_target - p_cached
                decisions.append(
                    ChunkRoutingDecision(
                        chunk_index=i,
                        target_start_pos=p_target,
                        target_end_pos=p_target + self.chunk_size,
                        token_ids=c_toks,
                        status=ChunkRoutingStatus.EXACT_REUSE,
                        similarity=1.0,
                        block_id=radix_hit.block_id,
                        cached_position=p_cached,
                        delta_p=delta_p,
                    )
                )
                self.pool.touch(radix_hit.block_id)
                continue

            # ------------------------------------------------------------------
            # Tier 2: FAISS HNSW Semantic Vector Match
            # ------------------------------------------------------------------
            if emb is not None:
                decision, similarity, block_id, cached_pos = self.semantic_index.query(emb)

                if decision == RouteDecision.EXACT_OR_REUSE and block_id is not None:
                    p_cached = cached_pos if cached_pos is not None else self.pool.get_start_pos(block_id)
                    delta_p = p_target - p_cached
                    decisions.append(
                        ChunkRoutingDecision(
                            chunk_index=i,
                            target_start_pos=p_target,
                            target_end_pos=p_target + self.chunk_size,
                            token_ids=c_toks,
                            status=ChunkRoutingStatus.SEMANTIC_REUSE,
                            similarity=similarity,
                            block_id=block_id,
                            cached_position=p_cached,
                            delta_p=delta_p,
                        )
                    )
                    self.pool.touch(block_id)
                    continue

                if decision == RouteDecision.SPECULATIVE and block_id is not None:
                    decisions.append(
                        ChunkRoutingDecision(
                            chunk_index=i,
                            target_start_pos=p_target,
                            target_end_pos=p_target + self.chunk_size,
                            token_ids=c_toks,
                            status=ChunkRoutingStatus.SPECULATIVE_DRAFT,
                            similarity=similarity,
                            block_id=block_id,
                            cached_position=cached_pos,
                            delta_p=0,
                        )
                    )
                    continue

            # ------------------------------------------------------------------
            # Cache Miss: Standard GPU Prefill Required
            # ------------------------------------------------------------------
            decisions.append(
                ChunkRoutingDecision(
                    chunk_index=i,
                    target_start_pos=p_target,
                    target_end_pos=p_target + self.chunk_size,
                    token_ids=c_toks,
                    status=ChunkRoutingStatus.CACHE_MISS,
                    similarity=0.0,
                    block_id=None,
                    cached_position=None,
                    delta_p=0,
                )
            )

        # Execute Double-Buffered Asynchronous DMA + Triton Kernel Pipelining
        reused_chunks = [
            d for d in decisions
            if d.status in (ChunkRoutingStatus.EXACT_REUSE, ChunkRoutingStatus.SEMANTIC_REUSE)
            and d.block_id is not None
        ]

        if not reused_chunks:
            empty = torch.empty((0, self.num_kv_heads, self.head_dim), dtype=torch.float16, device=self.device)
            return empty, empty, decisions

        total_tokens = len(reused_chunks) * self.chunk_size
        k_out = torch.empty((total_tokens, self.num_kv_heads, self.head_dim), dtype=torch.float16, device=self.device)
        v_out = torch.empty((total_tokens, self.num_kv_heads, self.head_dim), dtype=torch.float16, device=self.device)

        k_reconstructed, v_reconstructed = self.execute_async_dma_and_rope(
            reused_decisions=reused_chunks,
            dest_k_out=k_out,
            dest_v_out=v_out,
        )

        return k_reconstructed, v_reconstructed, decisions

    def lookup_chunks(
        self,
        prompt_tokens: Sequence[int],
        chunk_embeddings: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
    ) -> List[ChunkRoutingDecision]:
        """Convenience method chunking prompt_tokens and querying indices."""
        tokens = list(prompt_tokens)
        num_chunks = len(tokens) // self.chunk_size
        chunks_tokens = [tokens[c * self.chunk_size : (c + 1) * self.chunk_size] for c in range(num_chunks)]
        target_positions = [c * self.chunk_size for c in range(num_chunks)]
        _, _, decisions = self.retrieve_and_reconstruct_kv(
            chunks_tokens=chunks_tokens,
            chunks_embeddings=chunk_embeddings,
            target_positions=target_positions,
        )
        return decisions

    def execute_async_dma_and_rope(
        self,
        reused_decisions: List[ChunkRoutingDecision],
        dest_k_out: Optional[torch.Tensor] = None,
        dest_v_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Double-buffered asynchronous PCIe DMA pipeline overlapping transfer with Triton kernel."""
        valid_chunks = [
            d for d in reused_decisions
            if d.status in (ChunkRoutingStatus.EXACT_REUSE, ChunkRoutingStatus.SEMANTIC_REUSE)
            and d.block_id is not None
        ]

        if not valid_chunks:
            empty = torch.empty((0, self.num_kv_heads, self.head_dim), dtype=torch.float16, device=self.device)
            return empty, empty

        total_tokens = len(valid_chunks) * self.chunk_size

        if dest_k_out is None:
            dest_k_out = torch.empty((total_tokens, self.num_kv_heads, self.head_dim), dtype=torch.float16, device=self.device)
        if dest_v_out is None:
            dest_v_out = torch.empty((total_tokens, self.num_kv_heads, self.head_dim), dtype=torch.float16, device=self.device)

        if not self._is_cuda or not TRITON_AVAILABLE:
            return self._execute_cpu_fallback(valid_chunks, dest_k_out, dest_v_out)

        num_chunks = len(valid_chunks)

        # Ensure staging buffers are initialized and match current geometry
        if (
            self._stage_k_quant is None
            or self._stage_k_quant[0].shape != (self.chunk_size, self.num_kv_heads, self.head_dim)
        ):
            self._stage_k_quant = [
                torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
                torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
            ]
            self._stage_k_scale = [
                torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
                torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
            ]
            self._stage_v_quant = [
                torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
                torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
            ]
            self._stage_v_scale = [
                torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
                torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
            ]

        # Precompute RoPE cos/sin factor tables for all chunks in a single GPU operation
        delta_positions = torch.tensor(
            [d.delta_p for d in valid_chunks],
            dtype=torch.float32,
            device=self.device,
        )
        cos_all, sin_all = compute_rope_cos_sin_delta(
            delta_positions, self.head_dim, device=self.device
        )

        # Allocate per-chunk synchronization events (zero-CPU-wait stream-level coordination)
        transfer_events = [torch.cuda.Event() for _ in range(num_chunks)]
        compute_events = [torch.cuda.Event() for _ in range(num_chunks)]

        # Prime first chunk into slot 0
        kq_0, ks_0, vq_0, vs_0 = self.pool.get_block_views(valid_chunks[0].block_id)
        with torch.cuda.stream(self.transfer_stream):
            self._stage_k_quant[0].copy_(kq_0, non_blocking=True)
            self._stage_k_scale[0].copy_(ks_0, non_blocking=True)
            self._stage_v_quant[0].copy_(vq_0, non_blocking=True)
            self._stage_v_scale[0].copy_(vs_0, non_blocking=True)
            transfer_events[0].record(self.transfer_stream)

        # Pipelined loop: host dispatches non-blocking DMA and compute without any CPU waits
        for i in range(num_chunks):
            curr_slot = i % 2
            next_slot = (i + 1) % 2

            # Stream 1: Transfer chunk (i + 1) while chunk (i) is computed
            if i + 1 < num_chunks:
                next_meta = valid_chunks[i + 1]
                assert next_meta.block_id is not None
                next_kq, next_ks, next_vq, next_vs = self.pool.get_block_views(next_meta.block_id)

                with torch.cuda.stream(self.transfer_stream):
                    # If this staging slot was used by chunk (i - 1), wait on GPU for compute to finish
                    if i >= 1:
                        self.transfer_stream.wait_event(compute_events[i - 1])

                    self._stage_k_quant[next_slot].copy_(next_kq, non_blocking=True)
                    self._stage_k_scale[next_slot].copy_(next_ks, non_blocking=True)
                    self._stage_v_quant[next_slot].copy_(next_vq, non_blocking=True)
                    self._stage_v_scale[next_slot].copy_(next_vs, non_blocking=True)
                    transfer_events[i + 1].record(self.transfer_stream)

            # Stream 2: Waits on transfer event and executes Triton kernel for chunk (i)
            out_offset = i * self.chunk_size
            k_dest_slice = dest_k_out[out_offset : out_offset + self.chunk_size]
            v_dest_slice = dest_v_out[out_offset : out_offset + self.chunk_size]
            cos_d = cos_all[i : i + 1]
            sin_d = sin_all[i : i + 1]

            with torch.cuda.stream(self.compute_stream):
                self.compute_stream.wait_event(transfer_events[i])

                triton_fused_k_dequant_rope_shift(
                    self._stage_k_quant[curr_slot],
                    self._stage_k_scale[curr_slot],
                    cos_d,
                    sin_d,
                    out=k_dest_slice,
                )
                dequant_value(
                    self._stage_v_quant[curr_slot],
                    self._stage_v_scale[curr_slot],
                    out=v_dest_slice,
                )
                compute_events[i].record(self.compute_stream)

        # Synchronize only compute_stream at the end of the entire pipeline
        self.compute_stream.synchronize()
        return dest_k_out, dest_v_out

    def _execute_cpu_fallback(
        self,
        valid_chunks: List[ChunkRoutingDecision],
        dest_k_out: torch.Tensor,
        dest_v_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU fallback when CUDA or Triton is unavailable."""
        for i, meta in enumerate(valid_chunks):
            assert meta.block_id is not None
            k_q, k_s, v_q, v_s = self.pool.get_block_views(meta.block_id)

            delta_pos = torch.tensor([meta.delta_p], device=self.device)
            cos_delta, sin_delta = compute_rope_cos_sin_delta(delta_pos, self.head_dim, device=self.device)

            out_offset = i * self.chunk_size
            k_slice = dest_k_out[out_offset : out_offset + self.chunk_size]
            v_slice = dest_v_out[out_offset : out_offset + self.chunk_size]

            pytorch_eager_dequant_rope_shift(k_q.to(self.device), k_s.to(self.device), cos_delta, sin_delta, out=k_slice)
            dequant_value_ref(v_q.to(self.device), v_s.to(self.device), out=v_slice)

        return dest_k_out, dest_v_out
