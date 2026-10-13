"""FAISS HNSW semantic vector index with cosine similarity threshold routing.

Tier 2 Semantic Routing Specifications:
- Queries normalized chunk embeddings using IndexHNSWFlat with inner product.
- Direct KV Reuse (cos_sim >= 0.92):
  Retrieve cached block_id, compute relative displacement Delta p = p_target - p_cached,
  and trigger fused INT8 dequantization + relative RoPE phase-shifting.
- Speculative Draft Zone (0.80 <= cos_sim < 0.92):
  Mark chunk as SPECULATIVE_DRAFT and surface cached continuation tokens for drafting.
- Cache Miss (cos_sim < 0.80):
  Forward chunk to standard GPU prefill computation.

Includes a robust PyTorch/NumPy cosine fallback when native FAISS C++ binaries
are not installed in the execution environment.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class SemanticRoutingAction(str, Enum):
    """Routing actions based on semantic similarity thresholds."""
    DIRECT_REUSE = "DIRECT_REUSE"
    SPECULATIVE_DRAFT = "SPECULATIVE_DRAFT"
    CACHE_MISS = "CACHE_MISS"


@dataclass
class SemanticLookupResult:
    """Result returned by Tier 2 semantic index lookup."""
    action: SemanticRoutingAction
    similarity: float
    block_id: Optional[int] = None
    cached_position: Optional[int] = None
    token_ids: Optional[Tuple[int, ...]] = None
    continuation_tokens: List[int] = field(default_factory=list)


@dataclass
class _ChunkVectorEntry:
    """Internal entry linking vector index IDs to cached chunk metadata."""
    faiss_id: int
    block_id: int
    cached_position: int
    token_ids: Tuple[int, ...]
    continuation_tokens: List[int]
    is_active: bool = True


class SemanticIndex:
    """Low-latency semantic chunk index with HNSW vector search and three-way routing."""

    def __init__(
        self,
        embedding_dim: int,
        tau_reuse: float = 0.92,
        tau_spec: float = 0.80,
        m: int = 32,
        ef_search: int = 32,
        ef_construction: int = 64,
        force_fallback: bool = False,
    ) -> None:
        """Initialize the semantic index.

        Args:
            embedding_dim: Dimensionality of chunk representation vectors.
            tau_reuse: Threshold for direct KV reuse (default 0.92).
            tau_spec: Threshold for speculative draft candidates (default 0.80).
            m: HNSW graph degree parameter M (default 32).
            ef_search: HNSW search depth parameter efSearch (default 32).
            ef_construction: HNSW build depth parameter efConstruction (default 64).
            force_fallback: Force pure PyTorch cosine search instead of FAISS.
        """
        if tau_spec >= tau_reuse:
            raise ValueError(f"tau_spec ({tau_spec}) must be strictly less than tau_reuse ({tau_reuse}).")

        self.embedding_dim = embedding_dim
        self.tau_reuse = tau_reuse
        self.tau_spec = tau_spec
        self.m = m
        self.ef_search = ef_search
        self.ef_construction = ef_construction

        self._use_faiss = FAISS_AVAILABLE and not force_fallback

        # Primary vector index
        if self._use_faiss:
            # Inner product metric over L2-normalized vectors is mathematically cosine similarity
            self._index = faiss.IndexHNSWFlat(embedding_dim, m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efSearch = ef_search
            self._index.hnsw.efConstruction = ef_construction
        else:
            # PyTorch fallback storage: [N, embedding_dim] float32 tensor
            self._fallback_embeddings = torch.empty((0, embedding_dim), dtype=torch.float32)

        # Vector ID tracking and reverse index mapping
        self._entries: Dict[int, _ChunkVectorEntry] = {}
        self._block_to_id: Dict[int, int] = {}
        self._next_id = 0

    @property
    def is_using_faiss(self) -> bool:
        """Return True if native FAISS HNSW backend is active."""
        return self._use_faiss

    def __len__(self) -> int:
        """Number of active semantic chunk vectors in the index."""
        return len(self._block_to_id)

    def _normalize(self, vector: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """L2-normalize input embedding to unit sphere."""
        if isinstance(vector, torch.Tensor):
            arr = vector.detach().cpu().to(torch.float32).numpy()
        else:
            arr = np.asarray(vector, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        # Avoid division by zero
        norm = np.where(norm < 1e-12, 1.0, norm)
        return (arr / norm).astype(np.float32)

    def add(
        self,
        embedding: Union[np.ndarray, torch.Tensor],
        block_id: int,
        cached_position: int,
        token_ids: Sequence[int],
        continuation_tokens: Optional[List[int]] = None,
    ) -> int:
        """Insert a normalized chunk embedding into the semantic index.

        Args:
            embedding: Vector representation of chunk semantics.
            block_id: Corresponding block index in the pinned pool.
            cached_position: Original token position m when key was rotated.
            token_ids: Token integer sequence for this chunk.
            continuation_tokens: Optional tokens following this chunk.

        Returns:
            assigned_id: Internal vector index integer ID.
        """
        # If block_id is already registered, remove old vector mapping first
        if block_id in self._block_to_id:
            self.remove(block_id)

        norm_vec = self._normalize(embedding)
        assigned_id = self._next_id
        self._next_id += 1

        entry = _ChunkVectorEntry(
            faiss_id=assigned_id,
            block_id=block_id,
            cached_position=cached_position,
            token_ids=tuple(token_ids),
            continuation_tokens=continuation_tokens or [],
            is_active=True,
        )
        self._entries[assigned_id] = entry
        self._block_to_id[block_id] = assigned_id

        if self._use_faiss:
            self._index.add(norm_vec)
        else:
            tensor_vec = torch.from_numpy(norm_vec)
            self._fallback_embeddings = torch.cat([self._fallback_embeddings, tensor_vec], dim=0)

        return assigned_id

    def search(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
    ) -> SemanticLookupResult:
        """Query the semantic index with cosine similarity threshold routing.

        Returns:
            SemanticLookupResult indicating DIRECT_REUSE, SPECULATIVE_DRAFT, or CACHE_MISS.
        """
        if not self._block_to_id:
            return SemanticLookupResult(
                action=SemanticRoutingAction.CACHE_MISS,
                similarity=0.0,
            )

        norm_query = self._normalize(query_embedding)

        # 1. Vector Search
        if self._use_faiss:
            # Query top-1 candidate (or top-k if handling soft-deleted IDs)
            top_k = min(8, self._index.ntotal)
            distances, indices = self._index.search(norm_query, top_k)
            best_sim = -1.0
            best_entry: Optional[_ChunkVectorEntry] = None

            for sim, idx in zip(distances[0], indices[0]):
                if idx in self._entries:
                    entry = self._entries[idx]
                    if entry.is_active:
                        best_sim = float(sim)
                        best_entry = entry
                        break
        else:
            # PyTorch normalized inner product search
            query_t = torch.from_numpy(norm_query)
            sims = torch.matmul(self._fallback_embeddings, query_t.T).squeeze(-1)
            # Find best active match
            sorted_sims, sorted_indices = torch.sort(sims, descending=True)
            best_sim = -1.0
            best_entry = None
            for s, idx in zip(sorted_sims.tolist(), sorted_indices.tolist()):
                if idx in self._entries and self._entries[idx].is_active:
                    best_sim = float(s)
                    best_entry = self._entries[idx]
                    break

        if best_entry is None or best_sim < self.tau_spec:
            return SemanticLookupResult(
                action=SemanticRoutingAction.CACHE_MISS,
                similarity=max(0.0, best_sim),
            )

        if best_sim >= self.tau_reuse:
            return SemanticLookupResult(
                action=SemanticRoutingAction.DIRECT_REUSE,
                similarity=best_sim,
                block_id=best_entry.block_id,
                cached_position=best_entry.cached_position,
                token_ids=best_entry.token_ids,
                continuation_tokens=best_entry.continuation_tokens,
            )

        # 0.80 <= best_sim < 0.92 -> Speculative Draft Zone
        return SemanticLookupResult(
            action=SemanticRoutingAction.SPECULATIVE_DRAFT,
            similarity=best_sim,
            block_id=best_entry.block_id,
            cached_position=best_entry.cached_position,
            token_ids=best_entry.token_ids,
            continuation_tokens=best_entry.continuation_tokens,
        )

    def remove(self, block_id: int) -> bool:
        """Evict block vector from the active search set in O(1)."""
        if block_id not in self._block_to_id:
            return False

        faiss_id = self._block_to_id.pop(block_id)
        if faiss_id in self._entries:
            self._entries[faiss_id].is_active = False

        return True

    def clear(self) -> None:
        """Reset the semantic index."""
        if self._use_faiss:
            self._index.reset()
        else:
            self._fallback_embeddings = torch.empty((0, self.embedding_dim), dtype=torch.float32)

        self._entries.clear()
        self._block_to_id.clear()
        self._next_id = 0
