"""Cache management, indexing, and DMA pipeline subpackage for SpecCache Phase 2."""

from speccache.cache.engine import (
    ChunkRoutingDecision,
    ChunkRoutingStatus,
    SpecCacheEngine,
)
from speccache.cache.pinned_pool import BlockMetadata, PinnedHostPool
from speccache.cache.radix_index import RadixMatchResult, RadixNode, RadixTree
from speccache.cache.semantic_index import (
    FAISS_AVAILABLE,
    SemanticIndex,
    SemanticLookupResult,
    SemanticRoutingAction,
)

__all__ = [
    "BlockMetadata",
    "ChunkRoutingDecision",
    "ChunkRoutingStatus",
    "FAISS_AVAILABLE",
    "PinnedHostPool",
    "RadixMatchResult",
    "RadixNode",
    "RadixTree",
    "SemanticIndex",
    "SemanticLookupResult",
    "SemanticRoutingAction",
    "SpecCacheEngine",
]
