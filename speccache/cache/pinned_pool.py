"""Page-locked (pinned) host memory pool and slab allocator for SpecCache Phase 2.

Systems & Hardware Grounding:
Standard pageable CPU memory forces the CUDA runtime to stage transfers through
an internal intermediate bounce-buffer, halving effective PCIe throughput.
PinnedKVPool pre-allocates a contiguous, page-locked (pin_memory=True) CPU host
slab allocator for quantized INT8 Key/Value states and FP16 scales.

Provides:
- O(1) LRU eviction via collections.OrderedDict.
- Zero-copy sliced views for asynchronous PCIe DMA transfers (cudaMemcpyAsync).
- Integer metadata tracking original token start positions (p_cached) for relative RoPE.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import time
from typing import Callable, Dict, List, Optional, Tuple
import torch


@dataclass
class OriginalPositionMetadata:
    """Metadata tracking a cached KV block inside the pinned host pool."""
    block_id: int
    start_pos: int
    token_ids: Tuple[int, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


# Backward compatibility alias
BlockMetadata = OriginalPositionMetadata


class PinnedKVPool:
    """Page-locked (pinned) host memory pool and LRU slab allocator.

    Pre-allocates contiguous memory slabs on host CPU with pin_memory=True
    to enable maximum PCIe DMA bandwidth without intermediate staging.
    """

    def __init__(
        self,
        max_blocks: int,
        chunk_size: int,
        num_heads: int,
        head_dim: int,
        on_evict_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Initialize page-locked host memory pool.

        Args:
            max_blocks: Capacity in number of cached chunk blocks.
            chunk_size: Number of tokens per chunk (e.g., 128, 256).
            num_heads: Number of KV heads (e.g., 8 for GQA, 32 for MHA).
            head_dim: Head dimension D (e.g., 64, 128).
            on_evict_callback: Optional callback invoked when a block_id is evicted.
        """
        if max_blocks <= 0:
            raise ValueError(f"max_blocks must be positive, got {max_blocks}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.max_blocks = max_blocks
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.on_evict_callback = on_evict_callback

        # Contiguous CPU memory slabs
        self.k_quant = torch.empty(
            (max_blocks, chunk_size, num_heads, head_dim),
            dtype=torch.int8,
            device="cpu",
        )
        self.k_scale = torch.empty(
            (max_blocks, chunk_size, num_heads, 1),
            dtype=torch.float16,
            device="cpu",
        )
        self.v_quant = torch.empty(
            (max_blocks, chunk_size, num_heads, head_dim),
            dtype=torch.int8,
            device="cpu",
        )
        self.v_scale = torch.empty(
            (max_blocks, chunk_size, num_heads, 1),
            dtype=torch.float16,
            device="cpu",
        )

        # Track original token start position (p_cached) for relative RoPE phase shift
        self.original_start_pos = torch.full((max_blocks,), -1, dtype=torch.long, device="cpu")

        # Page-lock host memory if CUDA runtime is available
        self._is_pinned = False
        if torch.cuda.is_available():
            try:
                self.k_quant = self.k_quant.pin_memory()
                self.k_scale = self.k_scale.pin_memory()
                self.v_quant = self.v_quant.pin_memory()
                self.v_scale = self.v_scale.pin_memory()
                self.original_start_pos = self.original_start_pos.pin_memory()
                self._is_pinned = True
            except Exception:
                self._is_pinned = False

        # Slab allocator: free slot stack
        self._free_block_ids: List[int] = list(range(max_blocks))

        # O(1) LRU eviction tracking: block_id -> OriginalPositionMetadata
        self._active_lru: OrderedDict[int, OriginalPositionMetadata] = OrderedDict()

    @property
    def is_pinned(self) -> bool:
        """Return True if host memory slabs are page-locked."""
        return self._is_pinned

    @property
    def allocated_count(self) -> int:
        """Number of active blocks currently held in the pool."""
        return len(self._active_lru)

    @property
    def free_count(self) -> int:
        """Number of unoccupied block slots available without eviction."""
        return len(self._free_block_ids)

    @property
    def allocated_blocks(self) -> List[int]:
        """List of active block IDs currently held in the pool."""
        return list(self._active_lru.keys())

    def allocate(self, block_id: Optional[int] = None, start_pos: int = -1) -> int:
        """Allocate a block slot from the arena in O(1).

        If block_id is specified, reserves that slot directly.
        Otherwise pops from free slots or evicts LRU block.

        Args:
            block_id: Optional specific block index in [0, max_blocks).
            start_pos: Initial token start position tracking.

        Returns:
            block_id: Integer index in [0, max_blocks).
        """
        if block_id is not None:
            if block_id in self._free_block_ids:
                self._free_block_ids.remove(block_id)
            meta = OriginalPositionMetadata(
                block_id=block_id,
                start_pos=start_pos,
                created_at=time.time(),
                last_accessed=time.time(),
            )
            self._active_lru[block_id] = meta
            return block_id

        if self._free_block_ids:
            block_id = self._free_block_ids.pop()
        else:
            # Evict least recently used block (first entry in OrderedDict)
            block_id, old_meta = self._active_lru.popitem(last=False)
            if self.on_evict_callback is not None:
                self.on_evict_callback(block_id)

        meta = OriginalPositionMetadata(
            block_id=block_id,
            start_pos=start_pos,
            created_at=time.time(),
            last_accessed=time.time(),
        )
        self._active_lru[block_id] = meta
        return block_id

    def touch(self, block_id: int) -> None:
        """Mark block as recently accessed in O(1) LRU ordering."""
        if block_id in self._active_lru:
            meta = self._active_lru[block_id]
            meta.last_accessed = time.time()
            self._active_lru.move_to_end(block_id)

    def write_block(
        self,
        block_id: int,
        k_q: torch.Tensor,
        k_s: torch.Tensor,
        v_q: torch.Tensor,
        v_s: torch.Tensor,
        start_pos: int,
        token_ids: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Write quantized Key and Value tensors into the assigned block slot.

        Args:
            block_id: Target block index in [0, max_blocks).
            k_q: Quantized keys [chunk_size, num_heads, head_dim], int8.
            k_s: Scale factors [chunk_size, num_heads, 1], float16.
            v_q: Quantized values [chunk_size, num_heads, head_dim], int8.
            v_s: Scale factors [chunk_size, num_heads, 1], float16.
            start_pos: Original token start position p_cached.
            token_ids: Optional tuple of token IDs for this chunk.
        """
        if block_id not in self._active_lru:
            raise KeyError(f"Block ID {block_id} is not an allocated active block.")

        expected_kv_shape = (self.chunk_size, self.num_heads, self.head_dim)
        expected_s_shape = (self.chunk_size, self.num_heads, 1)

        if k_q.shape != expected_kv_shape or k_q.dtype != torch.int8:
            raise ValueError(f"k_q shape {k_q.shape} dtype {k_q.dtype} invalid.")
        if k_s.shape != expected_s_shape or k_s.dtype != torch.float16:
            raise ValueError(f"k_s shape {k_s.shape} dtype {k_s.dtype} invalid.")
        if v_q.shape != expected_kv_shape or v_q.dtype != torch.int8:
            raise ValueError(f"v_q shape {v_q.shape} dtype {v_q.dtype} invalid.")
        if v_s.shape != expected_s_shape or v_s.dtype != torch.float16:
            raise ValueError(f"v_s shape {v_s.shape} dtype {v_s.dtype} invalid.")

        self.k_quant[block_id].copy_(k_q, non_blocking=False)
        self.k_scale[block_id].copy_(k_s, non_blocking=False)
        self.v_quant[block_id].copy_(v_q, non_blocking=False)
        self.v_scale[block_id].copy_(v_s, non_blocking=False)
        self.original_start_pos[block_id] = start_pos

        meta = self._active_lru[block_id]
        meta.start_pos = start_pos
        if token_ids is not None:
            meta.token_ids = token_ids
        self.touch(block_id)

    def get_block_views(
        self,
        block_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return zero-copy slice views into pinned host slabs for DMA transfer.

        Args:
            block_id: Target block index in [0, max_blocks).

        Returns:
            Tuple containing (k_quant, k_scale, v_quant, v_scale) pinned host tensor views.
        """
        if block_id not in self._active_lru:
            raise KeyError(f"Block ID {block_id} is not an allocated active block.")

        self.touch(block_id)
        return (
            self.k_quant[block_id],
            self.k_scale[block_id],
            self.v_quant[block_id],
            self.v_scale[block_id],
        )

    def get_start_pos(self, block_id: int) -> int:
        """Fetch the original token start position (p_cached) for a block."""
        if block_id not in self._active_lru:
            raise KeyError(f"Block ID {block_id} is not an allocated active block.")
        return int(self.original_start_pos[block_id].item())

    def free(self, block_id: int) -> None:
        """Explicitly release an active block back to the pool."""
        if block_id in self._active_lru:
            del self._active_lru[block_id]
            self.original_start_pos[block_id] = -1
            self._free_block_ids.append(block_id)
            if self.on_evict_callback is not None:
                self.on_evict_callback(block_id)

    def clear(self) -> None:
        """Reset the allocator and release all blocks."""
        all_ids = list(self._active_lru.keys())
        for bid in all_ids:
            if self.on_evict_callback is not None:
                self.on_evict_callback(bid)
        self._active_lru.clear()
        self.original_start_pos.fill_(-1)
        self._free_block_ids = list(range(self.max_blocks))


# Backward compatibility alias
PinnedHostPool = PinnedKVPool
