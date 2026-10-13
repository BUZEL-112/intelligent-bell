"""Chunk-level Radix Tree for exact prefix and block matching in SpecCache Phase 2.

Implements ChunkRadixTree:
- Maps tuples of token IDs along a prefix tree to block_ids stored in PinnedKVPool.
- Supports insert(token_ids, block_ids) across single or multi-chunk prompt sequences.
- Supports match_prefix(token_ids) -> list[tuple[int, int]] returning matched (chunk_idx, block_id).
- Handles clean node pruning when PinnedKVPool evicts a block_id.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class RadixMatchResult:
    """Result returned when a token sequence or prefix matches in the Radix tree."""
    block_id: int
    cached_position: int
    token_ids: Tuple[int, ...]
    matched_length: int
    continuation_tokens: List[int] = field(default_factory=list)


class ChunkRadixNode:
    """Node in the chunk prefix tree representing one fixed-size chunk block."""

    def __init__(
        self,
        chunk_tokens: Tuple[int, ...],
        parent: Optional["ChunkRadixNode"] = None,
        block_id: Optional[int] = None,
        cached_position: int = 0,
    ) -> None:
        self.chunk_tokens = chunk_tokens
        self.parent = parent
        self.block_id = block_id
        self.cached_position = cached_position
        # Child nodes keyed by their chunk_tokens tuple for exact chunk matching
        self.children: Dict[Tuple[int, ...], ChunkRadixNode] = {}

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class ChunkRadixTree:
    """Prefix tree mapping token chunk sequences to PinnedKVPool block_ids."""

    def __init__(self, chunk_size: int = 128) -> None:
        """Initialize the chunk radix tree.

        Args:
            chunk_size: Token count per chunk block.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.chunk_size = chunk_size
        self.root = ChunkRadixNode(chunk_tokens=())
        # Reverse map for O(1) eviction: block_id -> set of ChunkRadixNode
        self._block_to_nodes: Dict[int, List[ChunkRadixNode]] = {}
        self._total_blocks = 0

    def __len__(self) -> int:
        """Number of active block references in the tree."""
        return self._total_blocks

    def insert(
        self,
        token_ids: Sequence[int],
        block_ids: Union[int, Sequence[int]],
        cached_position: int = 0,
    ) -> None:
        """Insert token sequence mapped to block_ids along the prefix tree.

        Args:
            token_ids: Full sequence of token IDs (length should be multiple of chunk_size).
            block_ids: Single block_id (if len(token_ids) == chunk_size) or list of block_ids.
            cached_position: Starting token position of the sequence.
        """
        tokens = list(token_ids)
        if len(tokens) == 0:
            return

        if isinstance(block_ids, int):
            b_ids = [block_ids]
        else:
            b_ids = list(block_ids)

        num_chunks = len(tokens) // self.chunk_size
        if num_chunks == 0:
            raise ValueError(
                f"token_ids length ({len(tokens)}) is less than chunk_size ({self.chunk_size})."
            )

        if len(b_ids) != num_chunks:
            # If a single block_id was passed for 1 chunk, allow it
            if len(b_ids) == 1 and num_chunks == 1:
                pass
            else:
                raise ValueError(
                    f"Number of block_ids ({len(b_ids)}) does not match chunk count ({num_chunks})."
                )

        curr = self.root
        for c in range(num_chunks):
            chunk_toks = tuple(tokens[c * self.chunk_size : (c + 1) * self.chunk_size])
            bid = b_ids[c]
            chunk_pos = cached_position + c * self.chunk_size

            if chunk_toks in curr.children:
                child = curr.children[chunk_toks]
                # Update block_id if newly provided
                if child.block_id != bid:
                    child.block_id = bid
                    child.cached_position = chunk_pos
                    if bid not in self._block_to_nodes:
                        self._block_to_nodes[bid] = []
                    self._block_to_nodes[bid].append(child)
                curr = child
            else:
                new_node = ChunkRadixNode(
                    chunk_tokens=chunk_toks,
                    parent=curr,
                    block_id=bid,
                    cached_position=chunk_pos,
                )
                curr.children[chunk_toks] = new_node
                if bid not in self._block_to_nodes:
                    self._block_to_nodes[bid] = []
                self._block_to_nodes[bid].append(new_node)
                self._total_blocks += 1
                curr = new_node

    def match_prefix(self, token_ids: Sequence[int]) -> List[Tuple[int, int]]:
        """Match as many contiguous chunks as possible from the prefix of token_ids.

        Args:
            token_ids: Sequence of prompt token IDs.

        Returns:
            List of (chunk_idx, block_id) pairs for each matched chunk.
        """
        tokens = list(token_ids)
        num_chunks = len(tokens) // self.chunk_size
        matches: List[Tuple[int, int]] = []

        curr = self.root
        for c in range(num_chunks):
            chunk_toks = tuple(tokens[c * self.chunk_size : (c + 1) * self.chunk_size])
            if chunk_toks in curr.children:
                child = curr.children[chunk_toks]
                if child.block_id is not None:
                    matches.append((c, child.block_id))
                    curr = child
                else:
                    break
            else:
                break

        return matches

    def match_exact(self, chunk_tokens: Sequence[int]) -> Optional[RadixMatchResult]:
        """Look up a single exact chunk in the root's immediate children."""
        toks = tuple(chunk_tokens)
        if len(toks) != self.chunk_size:
            return None

        if toks in self.root.children:
            node = self.root.children[toks]
            if node.block_id is not None:
                return RadixMatchResult(
                    block_id=node.block_id,
                    cached_position=node.cached_position,
                    token_ids=toks,
                    matched_length=self.chunk_size,
                )
        return None

    def evict_block(self, block_id: int) -> bool:
        """Evict all occurrences of block_id and prune empty leaf branches upward.

        Args:
            block_id: The ID of the block to evict.

        Returns:
            True if one or more nodes were evicted, False otherwise.
        """
        if block_id not in self._block_to_nodes:
            return False

        nodes = self._block_to_nodes.pop(block_id)
        for node in nodes:
            node.block_id = None
            self._total_blocks = max(0, self._total_blocks - 1)
            # Prune upward if leaf
            self._prune_upward(node)

        return True

    def _prune_upward(self, node: ChunkRadixNode) -> None:
        """Recursively prune empty non-terminating leaves up to root."""
        curr = node
        while curr.parent is not None and curr.is_leaf and curr.block_id is None:
            parent = curr.parent
            if curr.chunk_tokens in parent.children:
                del parent.children[curr.chunk_tokens]
            curr = parent

    def clear(self) -> None:
        """Reset the radix tree."""
        self.root = ChunkRadixNode(chunk_tokens=())
        self._block_to_nodes.clear()
        self._total_blocks = 0


# Backward compatibility aliases
RadixTree = ChunkRadixTree
RadixNode = ChunkRadixNode
