"""Page-locked (pinned) host memory pool and LRU slab allocator for SpecCache.

Hardware & Systems Motivation:
Standard pageable CPU memory forces the CUDA runtime to copy buffers through an
internal page-locked staging area before dispatching PCIe DMA requests, reducing
effective PCIe Gen4 bandwidth by up to 50%. By allocating a pre-sized contiguous
host arena backed by page-locked memory (cudaHostAlloc / pin_memory), DMA engines
read host memory directly via cudaMemcpyAsync (non_blocking=True), achieving peak
PCIe transfer speeds.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import time
from typing import Callable, Dict, List, Optional, Tuple
import torch


@dataclass
class BlockMetadata:
    """Metadata describing a cached KV chunk stored in the pinned pool."""
    block_id: int
    cached_position: int
    token_ids: Tuple[int, ...]
    continuation_tokens: List[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class PinnedHostPool:
    """Page-locked host memory slab allocator with LRU eviction management.

    Pre-allocates contiguous memory arenas on the host for INT8 quantized Key and
    Value tensors along with their per-token-head FP16 scales.
    """

    def __init__(
        self,
        max_blocks: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        on_evict_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Initialize the pinned host memory pool.

        Args:
            max_blocks: Maximum number of chunk blocks stored in the arena.
            chunk_size: Token count per chunk (e.g., 64, 128, 256).
            num_kv_heads: Number of Key-Value heads (GQA/MHA).
            head_dim: Head dimension D (e.g., 64, 128).
            device: Target GPU device for pinned transfers.
            on_evict_callback: Optional callback invoked when a block_id is evicted.
        """
        if max_blocks <= 0:
            raise ValueError(f"max_blocks must be positive, got {max_blocks}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.max_blocks = max_blocks
        self.chunk_size = chunk_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.on_evict_callback = on_evict_callback

        # Allocate contiguous CPU memory arenas
        self._host_k_quant = torch.empty(
            (max_blocks, chunk_size, num_kv_heads, head_dim),
            dtype=torch.int8,
            device="cpu",
        )
        self._host_k_scale = torch.empty(
            (max_blocks, chunk_size, num_kv_heads, 1),
            dtype=torch.float16,
            device="cpu",
        )
        self._host_v_quant = torch.empty(
            (max_blocks, chunk_size, num_kv_heads, head_dim),
            dtype=torch.int8,
            device="cpu",
        )
        self._host_v_scale = torch.empty(
            (max_blocks, chunk_size, num_kv_heads, 1),
            dtype=torch.float16,
            device="cpu",
        )

        # Pin memory if CUDA is supported and available
        self._is_pinned = False
        if torch.cuda.is_available():
            try:
                self._host_k_quant = self._host_k_quant.pin_memory()
                self._host_k_scale = self._host_k_scale.pin_memory()
                self._host_v_quant = self._host_v_quant.pin_memory()
                self._host_v_scale = self._host_v_scale.pin_memory()
                self._is_pinned = True
            except Exception:
                self._is_pinned = False

        # Slab allocator free list
        self._free_block_ids: List[int] = list(range(max_blocks))

        # LRU eviction tracking: block_id -> BlockMetadata
        self._active_blocks: OrderedDict[int, BlockMetadata] = OrderedDict()

    @property
    def is_pinned(self) -> bool:
        """Return True if host memory slabs are page-locked."""
        return self._is_pinned

    @property
    def allocated_count(self) -> int:
        """Number of active blocks currently held in the pool."""
        return len(self._active_blocks)

    @property
    def free_count(self) -> int:
        """Number of unoccupied block slots available without eviction."""
        return len(self._free_block_ids)

    def allocate(
        self,
        cached_position: int,
        token_ids: Tuple[int, ...],
        continuation_tokens: Optional[List[int]] = None,
    ) -> int:
        """Allocate a block slot from the arena, evicting the LRU block if full.

        Args:
            cached_position: Original token position m for RoPE displacement calculation.
            token_ids: Sequence of token integers for this chunk.
            continuation_tokens: Following tokens for speculative draft candidates.

        Returns:
            block_id: Integer index into the pinned slab arena in [0, max_blocks).
        """
        if self._free_block_ids:
            block_id = self._free_block_ids.pop()
        else:
            # Evict least-recently used block (first entry in OrderedDict)
            block_id, old_meta = self._active_blocks.popitem(last=False)
            if self.on_evict_callback is not None:
                self.on_evict_callback(block_id)

        meta = BlockMetadata(
            block_id=block_id,
            cached_position=cached_position,
            token_ids=token_ids,
            continuation_tokens=continuation_tokens or [],
            created_at=time.time(),
            last_accessed=time.time(),
        )
        self._active_blocks[block_id] = meta
        return block_id

    def write_chunk(
        self,
        block_id: int,
        k_quant: torch.Tensor,
        k_scale: torch.Tensor,
        v_quant: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        """Copy quantized Key and Value tensors into the assigned block slot.

        Args:
            block_id: Target block index in [0, max_blocks).
            k_quant: Tensor of shape [chunk_size, num_kv_heads, head_dim], int8.
            k_scale: Tensor of shape [chunk_size, num_kv_heads, 1], float16.
            v_quant: Tensor of shape [chunk_size, num_kv_heads, head_dim], int8.
            v_scale: Tensor of shape [chunk_size, num_kv_heads, 1], float16.
        """
        if block_id not in self._active_blocks:
            raise KeyError(f"Block ID {block_id} is not an allocated active block.")

        expected_k_shape = (self.chunk_size, self.num_kv_heads, self.head_dim)
        expected_s_shape = (self.chunk_size, self.num_kv_heads, 1)

        if k_quant.shape != expected_k_shape or k_quant.dtype != torch.int8:
            raise ValueError(f"k_quant shape {k_quant.shape} dtype {k_quant.dtype} invalid.")
        if k_scale.shape != expected_s_shape or k_scale.dtype != torch.float16:
            raise ValueError(f"k_scale shape {k_scale.shape} dtype {k_scale.dtype} invalid.")
        if v_quant.shape != expected_k_shape or v_quant.dtype != torch.int8:
            raise ValueError(f"v_quant shape {v_quant.shape} dtype {v_quant.dtype} invalid.")
        if v_scale.shape != expected_s_shape or v_scale.dtype != torch.float16:
            raise ValueError(f"v_scale shape {v_scale.shape} dtype {v_scale.dtype} invalid.")

        # Copy data into host memory slab
        self._host_k_quant[block_id].copy_(k_quant, non_blocking=False)
        self._host_k_scale[block_id].copy_(k_scale, non_blocking=False)
        self._host_v_quant[block_id].copy_(v_quant, non_blocking=False)
        self._host_v_scale[block_id].copy_(v_scale, non_blocking=False)

        self.touch(block_id)

    def read_chunk(
        self,
        block_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve memory slice references for a block and update its LRU recency.

        Returns:
            Tuple containing (k_quant, k_scale, v_quant, v_scale) pinned host slices.
        """
        if block_id not in self._active_blocks:
            raise KeyError(f"Block ID {block_id} is not allocated.")

        self.touch(block_id)
        return (
            self._host_k_quant[block_id],
            self._host_k_scale[block_id],
            self._host_v_quant[block_id],
            self._host_v_scale[block_id],
        )

    def touch(self, block_id: int) -> None:
        """Mark block as recently used in LRU ordering."""
        if block_id in self._active_blocks:
            meta = self._active_blocks[block_id]
            meta.last_accessed = time.time()
            self._active_blocks.move_to_end(block_id)

    def get_metadata(self, block_id: int) -> Optional[BlockMetadata]:
        """Fetch metadata for a cached block without modifying LRU recency."""
        return self._active_blocks.get(block_id)

    def free_block(self, block_id: int) -> None:
        """Explicitly return a block to the free list."""
        if block_id in self._active_blocks:
            del self._active_blocks[block_id]
            self._free_block_ids.append(block_id)
            if self.on_evict_callback is not None:
                self.on_evict_callback(block_id)

    def clear(self) -> None:
        """Reset allocator state, freeing all blocks."""
        all_ids = list(self._active_blocks.keys())
        for bid in all_ids:
            if self.on_evict_callback is not None:
                self.on_evict_callback(bid)
        self._active_blocks.clear()
        self._free_block_ids = list(range(self.max_blocks))
