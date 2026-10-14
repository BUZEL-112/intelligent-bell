"""Multi-layer DynamicCache and past_key_values injection with relative RoPE phase-shift.

Provides:
- SpecDynamicCache: High-performance Dynamic KV Cache compatible with HuggingFace DynamicCache.
- MultiLayerKVInjector (KVInjector): Coordinates layer-wise PinnedKVPool slabs across num_layers
  (e.g., 24, 28, or 32 layers). Coarsens DMA transfers across layers into past_key_values
  to minimize Python loop overhead, applying triton_fused_k_dequant_rope_shift for keys
  and dequantizing values.
"""

from collections.abc import Sequence
from typing import Any

import torch

from speccache.cache.pinned_pool import PinnedKVPool
from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    dequant_value_ref,
    pytorch_eager_dequant_rope_shift,
    rotate_half,
)
from speccache.utils import get_logger

logger = get_logger(__name__)

try:
    from speccache.kernels.triton_rope_kv import (
        TRITON_AVAILABLE,
        dequant_value,
        triton_fused_k_dequant_rope_shift,
    )
except ImportError:
    TRITON_AVAILABLE = False
    dequant_value = dequant_value_ref  # type: ignore


class SpecDynamicCache:
    """High-performance Dynamic KV Cache compatible with HuggingFace DynamicCache interface.

    Maintains per-layer Key and Value tensors in layout:
    [batch_size, num_heads, seq_len, head_dim].
    Supports rollback/truncation for speculative draft verification.
    """

    def __init__(self) -> None:
        """Initialize empty Dynamic KV cache."""
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def __len__(self) -> int:
        """Number of layers registered in cache."""
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch (key, value) pair for layer_idx."""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return sequence length stored for specified layer."""
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> int:
        """Return maximum sequence length across all layers."""
        if not self.key_cache:
            return 0
        return max(self.get_seq_length(layer_idx) for layer_idx in range(len(self.key_cache)))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for layer_idx and return full cumulative (key, value) tensors.

        Args:
            key_states: Keys of shape [batch_size, num_heads, seq_len, head_dim].
            value_states: Values of shape [batch_size, num_heads, seq_len, head_dim].
            layer_idx: Transformer layer index.
            cache_kwargs: Additional kwargs for HuggingFace compatibility.

        Returns:
            Tuple of (cumulative_keys, cumulative_values) for the layer.
        """
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)  # type: ignore
            self.value_cache.append(None)  # type: ignore

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def truncate(self, max_seq_len: int) -> None:
        """Truncate all layers along sequence dimension to max_seq_len.

        Used by speculative decoding rollback when candidate tokens are rejected.

        Args:
            max_seq_len: Target prefix length to keep.
        """
        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None and self.key_cache[i].shape[-2] > max_seq_len:
                self.key_cache[i] = self.key_cache[i][..., :max_seq_len, :].contiguous()
                self.value_cache[i] = self.value_cache[i][..., :max_seq_len, :].contiguous()

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy HuggingFace tuple-of-tuples past_key_values."""
        return tuple(
            (self.key_cache[i], self.value_cache[i])
            for i in range(len(self.key_cache))
            if self.key_cache[i] is not None
        )

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]]
    ) -> "SpecDynamicCache":
        """Construct SpecDynamicCache from legacy past_key_values tuple."""
        cache = cls()
        for layer_idx, (k, v) in enumerate(past_key_values):
            cache.update(k, v, layer_idx=layer_idx)
        return cache


class MultiLayerKVInjector:
    """Manages layer-wise SpecCacheEngine / PinnedKVPool slabs across num_layers.

    Coarsens DMA transfers or batches across layers so reconstructing a full multi-layer
    KV cache from pinned host memory into past_key_values (HuggingFace DynamicCache compatible
    tuple of (K_layer, V_layer)) minimizes Python loop overhead.
    Calls triton_fused_k_dequant_rope_shift for keys and dequantizes values.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | None = None,
        max_blocks: int = 128,
        chunk_size: int = 64,
    ) -> None:
        """Initialize MultiLayerKVInjector.

        Args:
            num_layers: Number of Transformer layers (e.g., 24, 28, or 32).
            num_kv_heads: Number of key-value attention heads (GQA/MHA).
            head_dim: Head dimension D (e.g., 64, 128).
            device: Target GPU/CPU device.
            max_blocks: Host pinned capacity in chunk blocks per layer.
            chunk_size: Number of tokens per chunk.
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.max_blocks = max_blocks

        # Layer-wise pinned memory pools
        self.layer_pools: list[PinnedKVPool] = [
            PinnedKVPool(
                max_blocks=max_blocks,
                chunk_size=chunk_size,
                num_heads=num_kv_heads,
                head_dim=head_dim,
            )
            for _ in range(num_layers)
        ]

    def format_to_model_layout(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [T, H, D] into [1, H, T, D] expected by HuggingFace attention.

        If already in [B, H, T, D], returns tensor unchanged.
        """
        if x.ndim == 3:
            return x.permute(1, 0, 2).unsqueeze(0).contiguous()
        elif x.ndim == 4:
            return x
        else:
            raise ValueError(f"Unexpected tensor ndim={x.ndim}, expected 3 or 4.")

    def format_to_engine_layout(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [1, H, T, D] into [T, H, D] expected by SpecCacheEngine.

        If already in [T, H, D], returns tensor unchanged.
        """
        if x.ndim == 4:
            return x.squeeze(0).permute(1, 0, 2).contiguous()
        elif x.ndim == 3:
            return x
        else:
            raise ValueError(f"Unexpected tensor ndim={x.ndim}, expected 3 or 4.")

    def apply_rope_shift(
        self,
        k_fp16: torch.Tensor,
        delta_p: int,
    ) -> torch.Tensor:
        """Apply relative RoPE rotation by Delta p to FP16 keys.

        Args:
            k_fp16: Key tensor of shape [T, H, D] or [1, H, T, D].
            delta_p: Position offset p_target - p_cached.

        Returns:
            Rotated key tensor in same layout and dtype.
        """
        if delta_p == 0:
            return k_fp16

        orig_ndim = k_fp16.ndim
        if orig_ndim == 4:
            t_tensor = self.format_to_engine_layout(k_fp16)
        else:
            t_tensor = k_fp16

        t, h, d = t_tensor.shape
        cos_d, sin_d = compute_rope_cos_sin_delta(
            torch.tensor([delta_p], device=t_tensor.device),
            head_dim=d,
            device=t_tensor.device,
        )

        cos_full = torch.cat([cos_d, cos_d], dim=-1)[:, None, :].to(torch.float32)
        sin_full = torch.cat([sin_d, sin_d], dim=-1)[:, None, :].to(torch.float32)
        k_f32 = t_tensor.to(torch.float32)
        rotated = (k_f32 * cos_full + rotate_half(k_f32) * sin_full).to(k_fp16.dtype)

        if orig_ndim == 4:
            return self.format_to_model_layout(rotated)
        return rotated

    def inject_layer_kv(
        self,
        cache: SpecDynamicCache,
        layer_idx: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        delta_p: int = 0,
    ) -> None:
        """Inject Key and Value states into specified cache layer."""
        k_formatted = self.format_to_model_layout(k_tensor)
        v_formatted = self.format_to_model_layout(v_tensor)

        if delta_p != 0:
            k_formatted = self.apply_rope_shift(k_formatted, delta_p)

        cache.update(k_formatted, v_formatted, layer_idx=layer_idx)

    def inject_multi_layer(
        self,
        cache: SpecDynamicCache,
        layer_k_list: Sequence[torch.Tensor],
        layer_v_list: Sequence[torch.Tensor],
        delta_p: int = 0,
    ) -> None:
        """Inject Key and Value states across all Transformer layers."""
        if len(layer_k_list) != len(layer_v_list):
            raise ValueError(
                f"Mismatch in layer counts: {len(layer_k_list)} keys vs {len(layer_v_list)} values"
            )

        for layer_idx, (k, v) in enumerate(zip(layer_k_list, layer_v_list, strict=True)):
            self.inject_layer_kv(
                cache=cache,
                layer_idx=layer_idx,
                k_tensor=k,
                v_tensor=v,
                delta_p=delta_p,
            )

    def reconstruct_layer_chunk(
        self,
        layer_idx: int,
        block_id: int,
        delta_p: int,
        transfer_stream: torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct FP16 Key (with RoPE phase shift) and Value for a single layer chunk."""
        pool = self.layer_pools[layer_idx]
        kq_host, ks_host, vq_host, vs_host = pool.get_block_views(block_id)

        # Asynchronous PCIe DMA from pinned host memory to GPU
        if self.device.type == "cuda" and transfer_stream is not None:
            with torch.cuda.stream(transfer_stream):
                kq_gpu = kq_host.to(self.device, non_blocking=True)
                ks_gpu = ks_host.to(self.device, non_blocking=True)
                vq_gpu = vq_host.to(self.device, non_blocking=True)
                vs_gpu = vs_host.to(self.device, non_blocking=True)
        else:
            kq_gpu = kq_host.to(self.device)
            ks_gpu = ks_host.to(self.device)
            vq_gpu = vq_host.to(self.device)
            vs_gpu = vs_host.to(self.device)

        # Compute cos and sin delta tables for relative RoPE phase shift
        try:
            cos_d, sin_d = compute_rope_cos_sin_delta(
                torch.tensor([delta_p], device=self.device),
                head_dim=self.head_dim,
                device=self.device,
            )

            # Dequantize and phase-shift Keys
            if self.device.type == "cuda" and TRITON_AVAILABLE:
                k_out = triton_fused_k_dequant_rope_shift(
                    kq_gpu, ks_gpu, cos_delta=cos_d, sin_delta=sin_d, delta_p=delta_p
                )
                v_out = dequant_value(vq_gpu, vs_gpu)
            else:
                k_out = pytorch_eager_dequant_rope_shift(
                    kq_gpu, ks_gpu, cos_delta=cos_d, sin_delta=sin_d, delta_p=delta_p
                )
                v_out = dequant_value_ref(vq_gpu, vs_gpu)

            logger.debug(
                "reconstruct_layer_chunk success: layer=%d block=%d delta_p=%d",
                layer_idx,
                block_id,
                delta_p,
            )
            return k_out, v_out
        except Exception as exc:
            logger.error(
                "reconstruct_layer_chunk failed: layer=%d block=%d delta_p=%d error=%s",
                layer_idx,
                block_id,
                delta_p,
                exc,
                exc_info=True,
            )
            raise

    def reconstruct_and_inject_multi_layer(
        self,
        cache: SpecDynamicCache,
        block_ids: list[int],
        delta_ps: list[int],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Coarsens DMA transfers across all layers and injects reconstructed KV cache.

        Minimizes Python loop overhead by pre-batching chunks across layers and
        directly populating SpecDynamicCache.

        Args:
            cache: SpecDynamicCache to inject reconstructed layers into.
            block_ids: List of cached block IDs to reconstruct.
            delta_ps: Position shifts (p_target - p_cached) corresponding to each block.

        Returns:
            HuggingFace-compatible past_key_values tuple of (K_layer, V_layer).
        """
        logger.info(
            "reconstruct_and_inject_multi_layer: blocks=%s delta_ps=%s num_layers=%d",
            block_ids,
            delta_ps,
            self.num_layers,
        )
        transfer_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

        for layer_idx in range(self.num_layers):
            layer_k_chunks: list[torch.Tensor] = []
            layer_v_chunks: list[torch.Tensor] = []

            for b_id, dp in zip(block_ids, delta_ps, strict=True):
                k_chunk, v_chunk = self.reconstruct_layer_chunk(
                    layer_idx=layer_idx,
                    block_id=b_id,
                    delta_p=dp,
                    transfer_stream=transfer_stream,
                )
                layer_k_chunks.append(k_chunk)
                layer_v_chunks.append(v_chunk)

            if layer_k_chunks:
                layer_k_full = torch.cat(layer_k_chunks, dim=0)
                layer_v_full = torch.cat(layer_v_chunks, dim=0)
                self.inject_layer_kv(
                    cache=cache,
                    layer_idx=layer_idx,
                    k_tensor=layer_k_full,
                    v_tensor=layer_v_full,
                    delta_p=0,  # Already shifted inside reconstruct_layer_chunk
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return cache.to_legacy_cache()


# Convenience backward compatibility alias
KVInjector = MultiLayerKVInjector
