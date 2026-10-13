"""SpecCacheEngine: Hybrid Radix-Semantic Lookup & Pinned-Memory DMA Pipeline.

Orchestrates:
1. Chunking prompt token sequences into fixed-size blocks (chunk_size).
2. Two-tier hierarchical cache lookup:
   - Tier 1: O(L) exact token-sequence matching via RadixTree.
   - Tier 2: FAISS HNSW cosine similarity search over chunk embeddings.
3. Three-way routing decisions:
   - DIRECT_REUSE: cos_sim >= 0.92 or exact match -> relative RoPE shift + DMA.
   - SPECULATIVE_DRAFT: 0.80 <= cos_sim < 0.92 -> surfaces draft candidate tokens.
   - CACHE_MISS: cos_sim < 0.80 -> marks chunk for GPU prefill computation.
4. Asynchronous double-buffered PCIe DMA pipeline overlapping host-to-device
   transfers of chunk i+1 with Phase 1 Triton kernel execution of chunk i.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from speccache.cache.pinned_pool import PinnedHostPool
from speccache.cache.radix_index import RadixMatchResult, RadixTree
from speccache.cache.semantic_index import (
    SemanticLookupResult,
    SemanticRoutingAction,
    SemanticIndex,
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
    CACHE_MISS = "CACHE_MISS"                # Similarity < 0.80 (must compute prefill)


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
    """Production orchestration engine for SpecCache Phase 2.

    Integrates pinned host memory allocation, Radix exact lookup, FAISS semantic
    indexing, and double-buffered asynchronous PCIe DMA pipeline execution.
    """

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
            chunk_size: Number of tokens per fixed-size chunk block.
            num_kv_heads: Number of Key-Value heads (e.g. 8 for Llama-3-8B GQA).
            head_dim: Hidden dimension D of each head (e.g. 128).
            max_blocks: Maximum chunks stored in the pinned host memory pool.
            embedding_dim: Dimensionality of semantic chunk embedding vectors.
            tau_reuse: Cosine similarity threshold for direct KV reuse (0.92).
            tau_spec: Cosine similarity threshold for speculative drafting (0.80).
            device: Target GPU device (default cuda:0 if available, else cpu).
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
        self.pool = PinnedHostPool(
            max_blocks=max_blocks,
            chunk_size=chunk_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=self.device,
            on_evict_callback=self._handle_lru_eviction,
        )

        # 2. Tier 1: Token-Level Radix Tree for Exact Matching
        self.radix_tree = RadixTree()

        # 3. Tier 2: FAISS HNSW Semantic Vector Index
        self.semantic_index = SemanticIndex(
            embedding_dim=embedding_dim,
            tau_reuse=tau_reuse,
            tau_spec=tau_spec,
            force_fallback=force_fallback_semantic,
        )

        # 4. Asynchronous Transfer Streams (when CUDA is active)
        self._is_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        if self._is_cuda:
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            self.compute_stream = torch.cuda.Stream(device=self.device)
        else:
            self.transfer_stream = None
            self.compute_stream = None

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
            token_ids: Token integer sequence of length equal to chunk_size.
            cached_position: Token index m where the keys were originally rotated.
            k_quant: INT8 quantized keys [chunk_size, num_kv_heads, head_dim].
            k_scale: FP16 scales [chunk_size, num_kv_heads, 1].
            v_quant: INT8 quantized values [chunk_size, num_kv_heads, head_dim].
            v_scale: FP16 scales [chunk_size, num_kv_heads, 1].
            embedding: Optional semantic embedding vector of length embedding_dim.
            continuation_tokens: Optional continuation tokens following this chunk.

        Returns:
            block_id: Assigned block index in the pinned pool.
        """
        tokens = tuple(token_ids)
        if len(tokens) != self.chunk_size:
            raise ValueError(
                f"token_ids length ({len(tokens)}) must match chunk_size ({self.chunk_size})."
            )

        # Allocate slot in pinned host memory pool
        block_id = self.pool.allocate(
            cached_position=cached_position,
            token_ids=tokens,
            continuation_tokens=continuation_tokens,
        )

        # Copy INT8 KV tensors into the pinned slab
        self.pool.write_chunk(
            block_id=block_id,
            k_quant=k_quant,
            k_scale=k_scale,
            v_quant=v_quant,
            v_scale=v_scale,
        )

        # Insert into Tier 1 Radix Tree
        self.radix_tree.insert(
            token_ids=tokens,
            block_id=block_id,
            cached_position=cached_position,
            continuation_tokens=continuation_tokens,
        )

        # Insert into Tier 2 Semantic Index
        if embedding is not None:
            self.semantic_index.add(
                embedding=embedding,
                block_id=block_id,
                cached_position=cached_position,
                token_ids=tokens,
                continuation_tokens=continuation_tokens,
            )

        return block_id

    def lookup_chunks(
        self,
        prompt_tokens: Sequence[int],
        chunk_embeddings: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
    ) -> List[ChunkRoutingDecision]:
        """Perform two-tier lookup across all fixed-size chunks in the prompt sequence.

        Args:
            prompt_tokens: Complete token ID sequence of the prompt.
            chunk_embeddings: Optional semantic embeddings for each chunk block.

        Returns:
            decisions: List of ChunkRoutingDecision objects for each chunk.
        """
        tokens = list(prompt_tokens)
        num_chunks = len(tokens) // self.chunk_size
        decisions: List[ChunkRoutingDecision] = []

        for i in range(num_chunks):
            start_pos = i * self.chunk_size
            end_pos = start_pos + self.chunk_size
            chunk_tokens = tuple(tokens[start_pos:end_pos])
            embedding = chunk_embeddings[i] if chunk_embeddings and i < len(chunk_embeddings) else None

            # ------------------------------------------------------------------
            # Tier 1: Token-Level Radix Tree Exact Match
            # ------------------------------------------------------------------
            radix_hit = self.radix_tree.match_exact(chunk_tokens)
            if radix_hit is not None:
                delta_p = start_pos - radix_hit.cached_position
                decisions.append(
                    ChunkRoutingDecision(
                        chunk_index=i,
                        target_start_pos=start_pos,
                        target_end_pos=end_pos,
                        token_ids=chunk_tokens,
                        status=ChunkRoutingStatus.EXACT_REUSE,
                        similarity=1.0,
                        block_id=radix_hit.block_id,
                        cached_position=radix_hit.cached_position,
                        delta_p=delta_p,
                        continuation_tokens=radix_hit.continuation_tokens,
                    )
                )
                self.pool.touch(radix_hit.block_id)
                continue

            # ------------------------------------------------------------------
            # Tier 2: FAISS HNSW Semantic Vector Match
            # ------------------------------------------------------------------
            if embedding is not None:
                semantic_hit = self.semantic_index.search(embedding)

                if semantic_hit.action == SemanticRoutingAction.DIRECT_REUSE:
                    delta_p = start_pos - (semantic_hit.cached_position or 0)
                    decisions.append(
                        ChunkRoutingDecision(
                            chunk_index=i,
                            target_start_pos=start_pos,
                            target_end_pos=end_pos,
                            token_ids=chunk_tokens,
                            status=ChunkRoutingStatus.SEMANTIC_REUSE,
                            similarity=semantic_hit.similarity,
                            block_id=semantic_hit.block_id,
                            cached_position=semantic_hit.cached_position,
                            delta_p=delta_p,
                            continuation_tokens=semantic_hit.continuation_tokens,
                        )
                    )
                    if semantic_hit.block_id is not None:
                        self.pool.touch(semantic_hit.block_id)
                    continue

                if semantic_hit.action == SemanticRoutingAction.SPECULATIVE_DRAFT:
                    decisions.append(
                        ChunkRoutingDecision(
                            chunk_index=i,
                            target_start_pos=start_pos,
                            target_end_pos=end_pos,
                            token_ids=chunk_tokens,
                            status=ChunkRoutingStatus.SPECULATIVE_DRAFT,
                            similarity=semantic_hit.similarity,
                            block_id=semantic_hit.block_id,
                            cached_position=semantic_hit.cached_position,
                            delta_p=0,
                            continuation_tokens=semantic_hit.continuation_tokens,
                        )
                    )
                    continue

            # ------------------------------------------------------------------
            # Cache Miss: Standard GPU Prefill Required
            # ------------------------------------------------------------------
            decisions.append(
                ChunkRoutingDecision(
                    chunk_index=i,
                    target_start_pos=start_pos,
                    target_end_pos=end_pos,
                    token_ids=chunk_tokens,
                    status=ChunkRoutingStatus.CACHE_MISS,
                    similarity=0.0,
                    block_id=None,
                    cached_position=None,
                    delta_p=0,
                )
            )

        return decisions

    def execute_async_dma_and_rope(
        self,
        reused_decisions: List[ChunkRoutingDecision],
        dest_k_out: Optional[torch.Tensor] = None,
        dest_v_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute double-buffered async PCIe DMA transfer overlapped with Triton kernel.

        Overlaps host-to-device PCIe transfer of chunk i+1 with Phase 1 kernel
        execution of chunk i across dedicated CUDA streams.

        Args:
            reused_decisions: Decisions marked EXACT_REUSE or SEMANTIC_REUSE.
            dest_k_out: Optional pre-allocated destination tensor [total_tokens, H, D].
            dest_v_out: Optional pre-allocated destination tensor [total_tokens, H, D].

        Returns:
            Tuple of (dest_k_out, dest_v_out) in FP16 on target device.
        """
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

        # If running on CPU or without Triton, execute sequential reference
        if not self._is_cuda or not TRITON_AVAILABLE:
            return self._execute_cpu_fallback(valid_chunks, dest_k_out, dest_v_out)

        # ----------------------------------------------------------------------
        # Double-Buffered GPU Staging Buffers for Overlapped DMA + Compute
        # ----------------------------------------------------------------------
        # Buffer 0 and Buffer 1 in GPU HBM
        stage_k_quant = [
            torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
            torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
        ]
        stage_k_scale = [
            torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
            torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
        ]
        stage_v_quant = [
            torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
            torch.empty((self.chunk_size, self.num_kv_heads, self.head_dim), dtype=torch.int8, device=self.device),
        ]
        stage_v_scale = [
            torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
            torch.empty((self.chunk_size, self.num_kv_heads, 1), dtype=torch.float16, device=self.device),
        ]

        # Synchronization events
        transfer_events = [torch.cuda.Event(), torch.cuda.Event()]
        compute_events = [torch.cuda.Event(), torch.cuda.Event()]

        num_chunks = len(valid_chunks)

        # Helper to initiate non-blocking PCIe DMA for chunk into buffer slot
        def initiate_dma(chunk_idx: int, slot: int):
            meta = valid_chunks[chunk_idx]
            assert meta.block_id is not None
            k_q_host, k_s_host, v_q_host, v_s_host = self.pool.read_chunk(meta.block_id)

            with torch.cuda.stream(self.transfer_stream):
                # Wait for prior compute using this buffer slot to complete
                if chunk_idx >= 2:
                    self.transfer_stream.wait_event(compute_events[slot])

                stage_k_quant[slot].copy_(k_q_host, non_blocking=True)
                stage_k_scale[slot].copy_(k_s_host, non_blocking=True)
                stage_v_quant[slot].copy_(v_q_host, non_blocking=True)
                stage_v_scale[slot].copy_(v_s_host, non_blocking=True)

                transfer_events[slot].record(self.transfer_stream)

        # Helper to execute Triton kernel on staged buffer slot
        def execute_kernel(chunk_idx: int, slot: int):
            meta = valid_chunks[chunk_idx]
            out_offset = chunk_idx * self.chunk_size

            with torch.cuda.stream(self.compute_stream):
                # Wait for DMA transfer of this slot to complete
                self.compute_stream.wait_event(transfer_events[slot])

                # Destination slices
                k_dest_slice = dest_k_out[out_offset : out_offset + self.chunk_size]
                v_dest_slice = dest_v_out[out_offset : out_offset + self.chunk_size]

                # Compute relative RoPE cos and sin tables
                delta_pos = torch.tensor([meta.delta_p], device=self.device)
                cos_delta, sin_delta = compute_rope_cos_sin_delta(
                    delta_pos, self.head_dim, device=self.device
                )

                # Phase 1 Triton Kernels: Fused Key Dequant + RoPE, and Value Dequant
                triton_fused_k_dequant_rope_shift(
                    stage_k_quant[slot],
                    stage_k_scale[slot],
                    cos_delta,
                    sin_delta,
                    out=k_dest_slice,
                )
                dequant_value(
                    stage_v_quant[slot],
                    stage_v_scale[slot],
                    out=v_dest_slice,
                )

                compute_events[slot].record(self.compute_stream)

        # ----------------------------------------------------------------------
        # Pipeline Execution: Prime Buffer 0, then overlap Loop
        # ----------------------------------------------------------------------
        # Prime first chunk into slot 0
        initiate_dma(0, slot=0)

        for i in range(num_chunks):
            curr_slot = i % 2
            next_slot = (i + 1) % 2

            # Dispatch DMA for chunk i+1 while chunk i is computed
            if i + 1 < num_chunks:
                initiate_dma(i + 1, slot=next_slot)

            # Compute chunk i
            execute_kernel(i, slot=curr_slot)

        # Synchronize streams with host
        self.compute_stream.synchronize()
        return dest_k_out, dest_v_out

    def _execute_cpu_fallback(
        self,
        valid_chunks: List[ChunkRoutingDecision],
        dest_k_out: torch.Tensor,
        dest_v_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential CPU reference fallback when CUDA or Triton is unavailable."""
        for i, meta in enumerate(valid_chunks):
            assert meta.block_id is not None
            k_q, k_s, v_q, v_s = self.pool.read_chunk(meta.block_id)

            delta_pos = torch.tensor([meta.delta_p], device=self.device)
            cos_delta, sin_delta = compute_rope_cos_sin_delta(delta_pos, self.head_dim, device=self.device)

            out_offset = i * self.chunk_size
            k_slice = dest_k_out[out_offset : out_offset + self.chunk_size]
            v_slice = dest_v_out[out_offset : out_offset + self.chunk_size]

            pytorch_eager_dequant_rope_shift(k_q.to(self.device), k_s.to(self.device), cos_delta, sin_delta, out=k_slice)
            dequant_value_ref(v_q.to(self.device), v_s.to(self.device), out=v_slice)

        return dest_k_out, dest_v_out
