"""Cache management, indexing, and DMA pipeline subpackage for SpecCache Phase 2."""

from speccache.cache.engine import (
    ChunkRoutingDecision,
    ChunkRoutingStatus,
    SpecCacheEngine,
)
from speccache.cache.pinned_pool import (
    BlockMetadata,
    OriginalPositionMetadata,
    PinnedHostPool,
    PinnedKVPool,
)
from speccache.cache.radix_index import (
    ChunkRadixNode,
    ChunkRadixTree,
    RadixMatchResult,
    RadixNode,
    RadixTree,
)
from speccache.cache.semantic_index import (
    FAISS_AVAILABLE,
    RouteDecision,
    SemanticHNSWIndex,
    SemanticIndex,
    SemanticLookupResult,
    SemanticRoutingAction,
)

__all__ = [
    "BlockMetadata",
    "ChunkRadixNode",
    "ChunkRadixTree",
    "ChunkRoutingDecision",
    "ChunkRoutingStatus",
    "FAISS_AVAILABLE",
    "OriginalPositionMetadata",
    "PinnedHostPool",
    "PinnedKVPool",
    "RadixMatchResult",
    "RadixNode",
    "RadixTree",
    "RouteDecision",
    "SemanticHNSWIndex",
    "SemanticIndex",
    "SemanticLookupResult",
    "SemanticRoutingAction",
    "SpecCacheEngine",
]
