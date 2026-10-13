"""FAISS HNSW semantic vector index with cosine similarity threshold routing.

Implements SemanticHNSWIndex:
- Uses faiss.IndexHNSWFlat(embed_dim, m, faiss.METRIC_INNER_PRODUCT) (with exact
  normalized PyTorch fallback when faiss is not installed).
- L2-normalizes all vectors before insertion and search so inner product equals cosine similarity.
- Maps internal FAISS sequential labels back to block_ids and filters out stale/evicted blocks.
- Classifies query matches into:
  * RouteDecision.EXACT_OR_REUSE (>= 0.92)
  * RouteDecision.SPECULATIVE ([0.80, 0.92))
  * RouteDecision.MISS (< 0.80)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class RouteDecision(str, Enum):
    """Three-way routing classification for semantic query similarity."""
    EXACT_OR_REUSE = "EXACT_OR_REUSE"  # Similarity >= 0.92
    SPECULATIVE = "SPECULATIVE"        # 0.80 <= Similarity < 0.92
    MISS = "MISS"                      # Similarity < 0.80


# Backward compatibility aliases
SemanticRoutingAction = RouteDecision
RouteDecision.DIRECT_REUSE = RouteDecision.EXACT_OR_REUSE  # type: ignore
RouteDecision.SPECULATIVE_DRAFT = RouteDecision.SPECULATIVE  # type: ignore
RouteDecision.CACHE_MISS = RouteDecision.MISS  # type: ignore


@dataclass
class SemanticLookupResult:
    """Result returned by semantic index query."""
    action: RouteDecision
    similarity: float
    block_id: Optional[int] = None
    cached_position: Optional[int] = None
    token_ids: Optional[Tuple[int, ...]] = None
    continuation_tokens: List[int] = field(default_factory=list)


@dataclass
class _VectorMeta:
    """Metadata for an indexed chunk vector."""
    label: int
    block_id: int
    cached_position: int
    is_active: bool = True


class SemanticHNSWIndex:
    """Low-latency semantic chunk index using FAISS HNSW with cosine threshold routing."""

    def __init__(
        self,
        embed_dim: int,
        max_elements: int = 50000,
        m: int = 32,
        ef_search: int = 32,
        ef_construction: int = 64,
        tau_reuse: float = 0.92,
        tau_spec: float = 0.80,
        force_fallback: bool = False,
    ) -> None:
        """Initialize the semantic HNSW index.

        Args:
            embed_dim: Hidden dimension of chunk semantic representation vectors.
            max_elements: Maximum capacity of the index.
            m: HNSW graph node degree parameter M (default 32).
            ef_search: HNSW search exploration depth efSearch (default 32).
            ef_construction: HNSW build exploration depth efConstruction (default 64).
            tau_reuse: Cosine similarity threshold for direct KV reuse (0.92).
            tau_spec: Cosine similarity threshold for speculative drafting (0.80).
            force_fallback: Force PyTorch normalized matrix search instead of FAISS.
        """
        if tau_spec >= tau_reuse:
            raise ValueError(f"tau_spec ({tau_spec}) must be strictly less than tau_reuse ({tau_reuse}).")

        self.embed_dim = embed_dim
        self.max_elements = max_elements
        self.m = m
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.tau_reuse = tau_reuse
        self.tau_spec = tau_spec

        self._use_faiss = FAISS_AVAILABLE and not force_fallback

        if self._use_faiss:
            # Inner product metric over L2-normalized vectors is cosine similarity
            self._index = faiss.IndexHNSWFlat(embed_dim, m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efSearch = ef_search
            self._index.hnsw.efConstruction = ef_construction
        else:
            self._fallback_embeddings = torch.empty((0, embed_dim), dtype=torch.float32)

        # Mapping: sequential_label -> _VectorMeta
        self._label_to_meta: Dict[int, _VectorMeta] = {}
        # Mapping: block_id -> sequential_label
        self._block_to_label: Dict[int, int] = {}
        self._next_label = 0

    @property
    def is_using_faiss(self) -> bool:
        """Return True if native FAISS HNSW backend is active."""
        return self._use_faiss

    def __len__(self) -> int:
        """Number of active semantic vectors in the index."""
        return len(self._block_to_label)

    def _normalize(self, vector: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """L2-normalize input embedding to unit sphere."""
        if isinstance(vector, torch.Tensor):
            arr = vector.detach().cpu().to(torch.float32).numpy()
        else:
            arr = np.asarray(vector, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        norm = np.where(norm < 1e-12, 1.0, norm)
        return (arr / norm).astype(np.float32)

    def add(
        self,
        embedding: Union[np.ndarray, torch.Tensor],
        block_id: int,
        cached_position: int = 0,
        token_ids: Optional[Sequence[int]] = None,
        continuation_tokens: Optional[List[int]] = None,
    ) -> int:
        """L2-normalize and insert embedding vector into the HNSW index.

        Args:
            embedding: Semantic representation vector.
            block_id: Corresponding block_id in PinnedKVPool.
            cached_position: Original token position p_cached.
            token_ids: Optional token IDs.
            continuation_tokens: Optional draft tokens.

        Returns:
            assigned_label: Internal sequential integer ID.
        """
        # If block_id already registered, invalidate previous entry
        if block_id in self._block_to_label:
            self.remove(block_id)

        norm_vec = self._normalize(embedding)
        label = self._next_label
        self._next_label += 1

        meta = _VectorMeta(
            label=label,
            block_id=block_id,
            cached_position=cached_position,
            is_active=True,
        )
        self._label_to_meta[label] = meta
        self._block_to_label[block_id] = label

        if self._use_faiss:
            self._index.add(norm_vec)
        else:
            tensor_vec = torch.from_numpy(norm_vec)
            self._fallback_embeddings = torch.cat([self._fallback_embeddings, tensor_vec], dim=0)

        return label

    def query(
        self,
        embedding: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[RouteDecision, float, Optional[int], Optional[int]]:
        """Query the index and classify match into RouteDecision.

        Returns:
            Tuple of (decision, similarity, block_id, cached_position).
        """
        res = self.search(embedding)
        return res.action, res.similarity, res.block_id, res.cached_position

    def search(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
    ) -> SemanticLookupResult:
        """Search the semantic index with cosine similarity threshold routing.

        Returns:
            SemanticLookupResult.
        """
        if not self._block_to_label:
            return SemanticLookupResult(
                action=RouteDecision.MISS,
                similarity=0.0,
            )

        norm_query = self._normalize(query_embedding)

        if self._use_faiss:
            top_k = min(16, self._index.ntotal)
            distances, indices = self._index.search(norm_query, top_k)
            best_sim = -1.0
            best_meta: Optional[_VectorMeta] = None

            for sim, idx in zip(distances[0], indices[0]):
                if idx in self._label_to_meta:
                    meta = self._label_to_meta[idx]
                    if meta.is_active:
                        best_sim = float(sim)
                        best_meta = meta
                        break
        else:
            query_t = torch.from_numpy(norm_query)
            sims = torch.matmul(self._fallback_embeddings, query_t.T).squeeze(-1)
            sorted_sims, sorted_indices = torch.sort(sims, descending=True)
            best_sim = -1.0
            best_meta = None
            for s, idx in zip(sorted_sims.tolist(), sorted_indices.tolist()):
                if idx in self._label_to_meta and self._label_to_meta[idx].is_active:
                    best_sim = float(s)
                    best_meta = self._label_to_meta[idx]
                    break

        if best_meta is None or best_sim < self.tau_spec:
            return SemanticLookupResult(
                action=RouteDecision.MISS,
                similarity=max(0.0, best_sim),
            )

        if best_sim >= self.tau_reuse:
            return SemanticLookupResult(
                action=RouteDecision.EXACT_OR_REUSE,
                similarity=best_sim,
                block_id=best_meta.block_id,
                cached_position=best_meta.cached_position,
            )

        return SemanticLookupResult(
            action=RouteDecision.SPECULATIVE,
            similarity=best_sim,
            block_id=best_meta.block_id,
            cached_position=best_meta.cached_position,
        )

    def remove(self, block_id: int) -> bool:
        """Invalidate block_id from active search set in O(1)."""
        if block_id not in self._block_to_label:
            return False

        label = self._block_to_label.pop(block_id)
        if label in self._label_to_meta:
            self._label_to_meta[label].is_active = False

        return True

    def clear(self) -> None:
        """Reset the semantic index."""
        if self._use_faiss:
            self._index.reset()
        else:
            self._fallback_embeddings = torch.empty((0, self.embed_dim), dtype=torch.float32)

        self._label_to_meta.clear()
        self._block_to_label.clear()
        self._next_label = 0


# Backward compatibility aliases
SemanticIndex = SemanticHNSWIndex
