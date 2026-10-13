"""Token-level Radix Tree for exact prefix and chunk cache matching.

Provides:
- O(L) exact prefix and chunk lookup for reusable prompt blocks.
- Common-prefix edge compression and splitting.
- Reverse index mapping for O(1) eviction when the pinned memory pool recycles blocks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RadixMatchResult:
    """Result returned when a token sequence or prefix matches in the Radix tree."""
    block_id: int
    cached_position: int
    token_ids: Tuple[int, ...]
    matched_length: int
    continuation_tokens: List[int] = field(default_factory=list)


class RadixNode:
    """Internal node in the Radix tree storing token sub-sequences."""

    def __init__(
        self,
        token_subsequence: Tuple[int, ...],
        parent: Optional["RadixNode"] = None,
        block_id: Optional[int] = None,
        cached_position: Optional[int] = None,
        continuation_tokens: Optional[List[int]] = None,
    ) -> None:
        self.token_subsequence = token_subsequence
        self.parent = parent
        self.block_id = block_id
        self.cached_position = cached_position
        self.continuation_tokens: List[int] = continuation_tokens or []
        # Children mapped by the first token of their sub-sequence
        self.children: Dict[int, RadixNode] = {}

    @property
    def is_leaf(self) -> bool:
        """True if the node has no children."""
        return len(self.children) == 0

    @property
    def is_block_terminator(self) -> bool:
        """True if this node represents a complete cached KV chunk."""
        return self.block_id is not None


class RadixTree:
    """Token-level Radix Tree supporting exact chunk matches and prefix queries."""

    def __init__(self) -> None:
        # Root node holds an empty token sequence
        self.root = RadixNode(token_subsequence=())
        # Reverse map for O(1) eviction: block_id -> RadixNode
        self._block_to_node: Dict[int, RadixNode] = {}
        self._total_nodes = 1

    def __len__(self) -> int:
        """Number of active cached blocks stored in the tree."""
        return len(self._block_to_node)

    def insert(
        self,
        token_ids: Sequence[int],
        block_id: int,
        cached_position: int,
        continuation_tokens: Optional[List[int]] = None,
    ) -> RadixNode:
        """Insert a token sequence representing a cached chunk into the tree.

        Handles edge splitting when a shared prefix already exists.

        Args:
            token_ids: Full sequence of token IDs for the chunk.
            block_id: Corresponding block index in the pinned pool.
            cached_position: Original token position m when the key was rotated.
            continuation_tokens: Optional continuation tokens for drafting.

        Returns:
            The terminal RadixNode storing the block.
        """
        tokens = tuple(token_ids)
        if not tokens:
            raise ValueError("Cannot insert empty token sequence.")

        curr_node = self.root
        idx = 0
        total_len = len(tokens)

        while idx < total_len:
            first_tok = tokens[idx]
            if first_tok not in curr_node.children:
                # No child branch exists; create new node with remaining tokens
                new_node = RadixNode(
                    token_subsequence=tokens[idx:],
                    parent=curr_node,
                    block_id=block_id,
                    cached_position=cached_position,
                    continuation_tokens=continuation_tokens,
                )
                curr_node.children[first_tok] = new_node
                self._block_to_node[block_id] = new_node
                self._total_nodes += 1
                return new_node

            child = curr_node.children[first_tok]
            subseq = child.token_subsequence

            # Calculate length of shared prefix
            common_len = 0
            while (
                common_len < len(subseq)
                and (idx + common_len) < total_len
                and subseq[common_len] == tokens[idx + common_len]
            ):
                common_len += 1

            if common_len < len(subseq):
                # Split existing child edge at common_len
                split_node = RadixNode(
                    token_subsequence=subseq[:common_len],
                    parent=curr_node,
                    block_id=None,
                    cached_position=None,
                )
                curr_node.children[first_tok] = split_node

                # Update existing child to become child of split_node with truncated subsequence
                child.token_subsequence = subseq[common_len:]
                child.parent = split_node
                split_node.children[child.token_subsequence[0]] = child

                curr_node = split_node
                self._total_nodes += 1
                idx += common_len

                # If all tokens consumed, split_node is our destination
                if idx == total_len:
                    split_node.block_id = block_id
                    split_node.cached_position = cached_position
                    split_node.continuation_tokens = continuation_tokens or []
                    self._block_to_node[block_id] = split_node
                    return split_node
            else:
                # Fully matched child edge; advance
                curr_node = child
                idx += len(subseq)

        # Existing node matched exactly; update block metadata
        curr_node.block_id = block_id
        curr_node.cached_position = cached_position
        curr_node.continuation_tokens = continuation_tokens or []
        self._block_to_node[block_id] = curr_node
        return curr_node

    def match_exact(self, token_ids: Sequence[int]) -> Optional[RadixMatchResult]:
        """Look up an exact token chunk in the Radix tree.

        Returns:
            RadixMatchResult if an exact cached block matches all token_ids, else None.
        """
        tokens = tuple(token_ids)
        if not tokens:
            return None

        curr_node = self.root
        idx = 0
        total_len = len(tokens)

        while idx < total_len:
            first_tok = tokens[idx]
            if first_tok not in curr_node.children:
                return None

            child = curr_node.children[first_tok]
            subseq = child.token_subsequence
            sub_len = len(subseq)

            if total_len - idx < sub_len or tokens[idx : idx + sub_len] != subseq:
                return None

            curr_node = child
            idx += sub_len

        if curr_node.is_block_terminator and curr_node.block_id is not None:
            return RadixMatchResult(
                block_id=curr_node.block_id,
                cached_position=curr_node.cached_position or 0,
                token_ids=tokens,
                matched_length=total_len,
                continuation_tokens=curr_node.continuation_tokens,
            )
        return None

    def match_prefix(self, token_ids: Sequence[int]) -> List[RadixMatchResult]:
        """Greedily traverse and return all contiguous matching chunk blocks from the start."""
        tokens = tuple(token_ids)
        matches: List[RadixMatchResult] = []
        if not tokens:
            return matches

        curr_node = self.root
        idx = 0
        total_len = len(tokens)

        while idx < total_len:
            first_tok = tokens[idx]
            if first_tok not in curr_node.children:
                break

            child = curr_node.children[first_tok]
            subseq = child.token_subsequence
            sub_len = len(subseq)

            if total_len - idx < sub_len or tokens[idx : idx + sub_len] != subseq:
                break

            curr_node = child
            idx += sub_len

            if curr_node.is_block_terminator and curr_node.block_id is not None:
                matches.append(
                    RadixMatchResult(
                        block_id=curr_node.block_id,
                        cached_position=curr_node.cached_position or 0,
                        token_ids=tokens[:idx],
                        matched_length=idx,
                        continuation_tokens=curr_node.continuation_tokens,
                    )
                )

        return matches

    def evict_block(self, block_id: int) -> bool:
        """Evict a block by block_id, pruning unneeded internal nodes in O(1).

        Args:
            block_id: The ID of the block to evict.

        Returns:
            True if the block was found and removed, False otherwise.
        """
        if block_id not in self._block_to_node:
            return False

        node = self._block_to_node.pop(block_id)
        node.block_id = None
        node.cached_position = None
        node.continuation_tokens = []

        # If leaf, prune upward
        if node.is_leaf and node.parent is not None:
            self._prune_upward(node)

        return True

    def _prune_upward(self, node: RadixNode) -> None:
        """Recursively prune empty non-terminating leaves up to root."""
        curr = node
        while curr.parent is not None and curr.is_leaf and not curr.is_block_terminator:
            parent = curr.parent
            first_tok = curr.token_subsequence[0]
            if first_tok in parent.children:
                del parent.children[first_tok]
                self._total_nodes -= 1
            curr = parent

    def clear(self) -> None:
        """Reset the Radix tree."""
        self.root = RadixNode(token_subsequence=())
        self._block_to_node.clear()
        self._total_nodes = 1
