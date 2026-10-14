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

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from speccache.utils import get_logger

logger = get_logger(__name__)


@dataclass
class OriginalPositionMetadata:
    """Metadata tracking a cached KV block inside the pinned host pool."""

    block_id: int
    start_pos: int
    token_ids: tuple[int, ...] = field(default_factory=tuple)
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
        on_evict_callback: Callable[[int], None] | None = None,
        k_quant_slab: torch.Tensor | None = None,
        k_scale_slab: torch.Tensor | None = None,
        v_quant_slab: torch.Tensor | None = None,
        v_scale_slab: torch.Tensor | None = None,
    ) -> None:
        """Initialize page-locked host memory pool.

        Args:
            max_blocks: Capacity in number of cached chunk blocks.
            chunk_size: Number of tokens per chunk (e.g., 128, 256).
            num_heads: Number of KV heads (e.g., 8 for GQA, 32 for MHA).
            head_dim: Head dimension D (e.g., 64, 128).
            on_evict_callback: Optional callback invoked when a block_id is evicted.
            k_quant_slab: Optional pre-allocated slab view for quantized keys.
            k_scale_slab: Optional pre-allocated slab view for key scales.
            v_quant_slab: Optional pre-allocated slab view for quantized values.
            v_scale_slab: Optional pre-allocated slab view for value scales.
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
        if k_quant_slab is not None:
            self.k_quant = k_quant_slab
            self.k_scale = (
                k_scale_slab
                if k_scale_slab is not None
                else torch.empty(
                    (max_blocks, chunk_size, num_heads, 1), dtype=torch.float16, device="cpu"
                )
            )
            self.v_quant = (
                v_quant_slab
                if v_quant_slab is not None
                else torch.empty(
                    (max_blocks, chunk_size, num_heads, head_dim), dtype=torch.int8, device="cpu"
                )
            )
            self.v_scale = (
                v_scale_slab
                if v_scale_slab is not None
                else torch.empty(
                    (max_blocks, chunk_size, num_heads, 1), dtype=torch.float16, device="cpu"
                )
            )
            self._is_pinned = getattr(self.k_quant, "is_pinned", lambda: False)()
        else:
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
            self._is_pinned = False
            if torch.cuda.is_available():
                try:
                    self.k_quant = self.k_quant.pin_memory()
                    self.k_scale = self.k_scale.pin_memory()
                    self.v_quant = self.v_quant.pin_memory()
                    self.v_scale = self.v_scale.pin_memory()
                    self._is_pinned = True
                except Exception as e:
                    logger.warning("Failed to pin host memory: %s", e)
                    self._is_pinned = False

        # Track original token start position (p_cached) for relative RoPE phase shift
        self.original_start_pos = torch.full((max_blocks,), -1, dtype=torch.long, device="cpu")
        if self._is_pinned:
            try:
                self.original_start_pos = self.original_start_pos.pin_memory()
            except Exception:
                pass

        # Page-locked staging buffers for coalescing non-contiguous block DMA transfers
        self._staging_capacity = max(max_blocks, 64)
        self._stage_kq = torch.empty(
            (self._staging_capacity * chunk_size, num_heads, head_dim),
            dtype=torch.int8,
            device="cpu",
        )
        self._stage_ks = torch.empty(
            (self._staging_capacity * chunk_size, num_heads, 1),
            dtype=torch.float16,
            device="cpu",
        )
        self._stage_vq = torch.empty(
            (self._staging_capacity * chunk_size, num_heads, head_dim),
            dtype=torch.int8,
            device="cpu",
        )
        self._stage_vs = torch.empty(
            (self._staging_capacity * chunk_size, num_heads, 1),
            dtype=torch.float16,
            device="cpu",
        )
        if self._is_pinned:
            try:
                self._stage_kq = self._stage_kq.pin_memory()
                self._stage_ks = self._stage_ks.pin_memory()
                self._stage_vq = self._stage_vq.pin_memory()
                self._stage_vs = self._stage_vs.pin_memory()
            except Exception as e:
                logger.warning("Failed to pin staging host memory: %s", e)

        if self._is_pinned:
            logger.info(
                "PinnedKVPool initialized: max_blocks=%d chunk_size=%d heads=%d head_dim=%d (pinned DMA active)",
                max_blocks,
                chunk_size,
                num_heads,
                head_dim,
            )
        else:
            logger.info(
                "PinnedKVPool initialized: max_blocks=%d chunk_size=%d heads=%d head_dim=%d (unpinned host memory)",
                max_blocks,
                chunk_size,
                num_heads,
                head_dim,
            )

        # Slab allocator: free slot stack
        self._free_block_ids: list[int] = list(range(max_blocks))

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
    def allocated_blocks(self) -> list[int]:
        """List of active block IDs currently held in the pool."""
        return list(self._active_lru.keys())

    def allocate(self, block_id: int | None = None, start_pos: int = -1) -> int:
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
            logger.debug("LRU evicted block_id=%d (start_pos=%d)", block_id, old_meta.start_pos)
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
        k_q: torch.Tensor | None = None,
        k_s: torch.Tensor | None = None,
        v_q: torch.Tensor | None = None,
        v_s: torch.Tensor | None = None,
        start_pos: int = -1,
        token_ids: tuple[int, ...] | None = None,
        *,
        k_quant: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_quant: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> None:
        """Write quantized Key and Value tensors into the assigned block slot.

        Supports positional parameters or keyword parameters with either
        shorthand names (k_q, k_s, v_q, v_s) or full names (k_quant, k_scale, v_quant, v_scale).

        Args:
            block_id: Target block index in [0, max_blocks).
            k_q: Quantized keys [chunk_size, num_heads, head_dim], int8.
            k_s: Scale factors [chunk_size, num_heads, 1], float16.
            v_q: Quantized values [chunk_size, num_heads, head_dim], int8.
            v_s: Scale factors [chunk_size, num_heads, 1], float16.
            start_pos: Original token start position p_cached.
            token_ids: Optional tuple of token IDs for this chunk.
            k_quant: Alias for k_q.
            k_scale: Alias for k_s.
            v_quant: Alias for v_q.
            v_scale: Alias for v_s.
        """
        if block_id not in self._active_lru:
            logger.error("write_block called on unallocated block_id=%d", block_id)
            raise KeyError(f"Block ID {block_id} is not an allocated active block.")

        k_target = k_q if k_q is not None else k_quant
        ks_target = k_s if k_s is not None else k_scale
        v_target = v_q if v_q is not None else v_quant
        vs_target = v_s if v_s is not None else v_scale

        if k_target is None or ks_target is None or v_target is None or vs_target is None:
            logger.error("write_block missing tensor components for block_id=%d", block_id)
            raise ValueError("Quantized key/value and scale factor tensors must be provided.")

        expected_kv_shape = (self.chunk_size, self.num_heads, self.head_dim)
        expected_s_shape = (self.chunk_size, self.num_heads, 1)

        if k_target.shape != expected_kv_shape or k_target.dtype != torch.int8:
            logger.error(
                "write_block k_quant mismatch: shape=%s expected=%s dtype=%s expected=int8",
                k_target.shape,
                expected_kv_shape,
                k_target.dtype,
            )
            raise ValueError(f"k_q shape {k_target.shape} dtype {k_target.dtype} invalid.")
        if ks_target.shape != expected_s_shape or ks_target.dtype != torch.float16:
            logger.error(
                "write_block k_scale mismatch: shape=%s expected=%s dtype=%s expected=float16",
                ks_target.shape,
                expected_s_shape,
                ks_target.dtype,
            )
            raise ValueError(f"k_s shape {ks_target.shape} dtype {ks_target.dtype} invalid.")
        if v_target.shape != expected_kv_shape or v_target.dtype != torch.int8:
            logger.error(
                "write_block v_quant mismatch: shape=%s expected=%s dtype=%s expected=int8",
                v_target.shape,
                expected_kv_shape,
                v_target.dtype,
            )
            raise ValueError(f"v_q shape {v_target.shape} dtype {v_target.dtype} invalid.")
        if vs_target.shape != expected_s_shape or vs_target.dtype != torch.float16:
            logger.error(
                "write_block v_scale mismatch: shape=%s expected=%s dtype=%s expected=float16",
                vs_target.shape,
                expected_s_shape,
                vs_target.dtype,
            )
            raise ValueError(f"v_s shape {vs_target.shape} dtype {vs_target.dtype} invalid.")

        self.k_quant[block_id].copy_(k_target, non_blocking=False)
        self.k_scale[block_id].copy_(ks_target, non_blocking=False)
        self.v_quant[block_id].copy_(v_target, non_blocking=False)
        self.v_scale[block_id].copy_(vs_target, non_blocking=False)
        self.original_start_pos[block_id] = start_pos

        meta = self._active_lru[block_id]
        meta.start_pos = start_pos
        if token_ids is not None:
            meta.token_ids = token_ids
        self.touch(block_id)

    def get_block_views(
        self,
        block_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def get_coalesced_views(
        self,
        block_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return coalesced pinned host memory tensor views for a sequence of blocks.

        For contiguous block sequences (e.g., range(b_start, b_end)), returns
        a zero-copy reshaped view [len(block_ids) * chunk_size, num_heads, head_dim]
        directly from the pinned memory slab.

        For non-contiguous block sequences, copies blocks into an internal
        page-locked staging buffer to provide a single contiguous pinned tensor
        for PCIe DMA transfer.

        Args:
            block_ids: List of allocated block indices.

        Returns:
            Tuple of (k_quant, k_scale, v_quant, v_scale) contiguous pinned host tensors.
        """
        if not block_ids:
            empty_kv = torch.empty(
                (0, self.num_heads, self.head_dim), dtype=torch.int8, device="cpu"
            )
            empty_s = torch.empty((0, self.num_heads, 1), dtype=torch.float16, device="cpu")
            return empty_kv, empty_s, empty_kv, empty_s

        for b in block_ids:
            self.touch(b)

        if len(block_ids) == 1:
            b0 = block_ids[0]
            return (
                self.k_quant[b0],
                self.k_scale[b0],
                self.v_quant[b0],
                self.v_scale[b0],
            )

        is_contiguous = block_ids[-1] - block_ids[0] == len(block_ids) - 1 and all(
            block_ids[i] == block_ids[0] + i for i in range(len(block_ids))
        )
        total_tokens = len(block_ids) * self.chunk_size

        if is_contiguous:
            b_start = block_ids[0]
            b_end = block_ids[-1] + 1
            # Zero-copy slice view into page-locked pinned memory
            return (
                self.k_quant[b_start:b_end].reshape(total_tokens, self.num_heads, self.head_dim),
                self.k_scale[b_start:b_end].reshape(total_tokens, self.num_heads, 1),
                self.v_quant[b_start:b_end].reshape(total_tokens, self.num_heads, self.head_dim),
                self.v_scale[b_start:b_end].reshape(total_tokens, self.num_heads, 1),
            )

        # Non-contiguous sequence: coalesce into internal page-locked staging buffer
        if len(block_ids) > self._staging_capacity:
            self._expand_staging(len(block_ids))

        for idx, b in enumerate(block_ids):
            st = idx * self.chunk_size
            en = st + self.chunk_size
            self._stage_kq[st:en].copy_(self.k_quant[b], non_blocking=False)
            self._stage_ks[st:en].copy_(self.k_scale[b], non_blocking=False)
            self._stage_vq[st:en].copy_(self.v_quant[b], non_blocking=False)
            self._stage_vs[st:en].copy_(self.v_scale[b], non_blocking=False)

        return (
            self._stage_kq[:total_tokens],
            self._stage_ks[:total_tokens],
            self._stage_vq[:total_tokens],
            self._stage_vs[:total_tokens],
        )

    def _expand_staging(self, required_blocks: int) -> None:
        """Expand page-locked staging buffers to hold required_blocks."""
        new_cap = max(required_blocks, self._staging_capacity * 2)
        kq = torch.empty(
            (new_cap * self.chunk_size, self.num_heads, self.head_dim),
            dtype=torch.int8,
            device="cpu",
        )
        ks = torch.empty(
            (new_cap * self.chunk_size, self.num_heads, 1),
            dtype=torch.float16,
            device="cpu",
        )
        vq = torch.empty(
            (new_cap * self.chunk_size, self.num_heads, self.head_dim),
            dtype=torch.int8,
            device="cpu",
        )
        vs = torch.empty(
            (new_cap * self.chunk_size, self.num_heads, 1),
            dtype=torch.float16,
            device="cpu",
        )
        if self._is_pinned:
            try:
                kq = kq.pin_memory()
                ks = ks.pin_memory()
                vq = vq.pin_memory()
                vs = vs.pin_memory()
            except Exception as e:
                logger.warning("Failed to pin expanded staging buffers: %s", e)

        self._stage_kq = kq
        self._stage_ks = ks
        self._stage_vq = vq
        self._stage_vs = vs
        self._staging_capacity = new_cap

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
