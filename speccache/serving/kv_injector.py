"""Multi-layer DynamicCache and past_key_values injection with relative RoPE phase-shift.

Provides:
- SpecDynamicCache: High-performance Dynamic KV Cache compatible with HuggingFace DynamicCache.
- KVInjector: Multi-layer injector translating contiguous Phase 1/2 INT8/FP16 reconstructed
  chunks into model KV cache format ([batch_size, num_heads, seq_len, head_dim]).
"""

from typing import Any, List, Optional, Sequence, Tuple, Union
import torch

from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    pytorch_eager_dequant_rope_shift,
    rotate_half,
)

try:
    from speccache.kernels.triton_rope_kv import (
        TRITON_AVAILABLE,
        triton_fused_k_dequant_rope_shift,
    )
except ImportError:
    TRITON_AVAILABLE = False


class SpecDynamicCache:
    """High-performance Dynamic KV Cache compatible with HuggingFace DynamicCache interface.

    Maintains per-layer Key and Value tensors in layout:
    [batch_size, num_heads, seq_len, head_dim].
    Supports rollback/truncation for speculative draft verification.
    """

    def __init__(self) -> None:
        """Initialize empty Dynamic KV cache."""
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def __len__(self) -> int:
        """Number of layers registered in cache."""
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return max(self.get_seq_length(l) for l in range(len(self.key_cache)))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for layer_idx and return full cumulative (key, value) tensors.

        Args:
            key_states: Keys of shape [batch_size, num_heads, seq_len, head_dim].
            value_states: Values of shape [batch_size, num_heads, seq_len, head_dim].
            layer_idx: Transformer layer index.
            cache_kwargs: Additional kwargs for HuggingFace compatibility.

        Returns:
            Tuple of (cumulative_keys, cumulative_values) for the layer.
        """
        # Expand layer cache capacity if needed
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)  # type: ignore
            self.value_cache.append(None)  # type: ignore

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
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

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy HuggingFace tuple-of-tuples past_key_values."""
        return tuple(
            (self.key_cache[i], self.value_cache[i])
            for i in range(len(self.key_cache))
            if self.key_cache[i] is not None
        )

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> "SpecDynamicCache":
        """Construct SpecDynamicCache from legacy past_key_values tuple."""
        cache = cls()
        for layer_idx, (k, v) in enumerate(past_key_values):
            cache.update(k, v, layer_idx=layer_idx)
        return cache


class KVInjector:
    """Coordinates injection of SpecCache reconstructed KV blocks into LLM DynamicCache.

    Translates between serving tensor layout [num_tokens, num_kv_heads, head_dim]
    and standard HuggingFace layout [batch_size, num_kv_heads, num_tokens, head_dim].
    Applies relative RoPE phase shifting when injecting at shifted sequence offsets.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize KV Injector.

        Args:
            num_layers: Number of Transformer layers.
            num_kv_heads: Number of key-value attention heads (GQA/MHA).
            head_dim: Head dimension D (e.g., 64, 128).
            device: Target GPU/CPU device.
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def format_to_model_layout(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [T, H, D] into [1, H, T, D] expected by HuggingFace attention.

        If already in [B, H, T, D], returns tensor unchanged.
        """
        if x.ndim == 3:
            # [T, H, D] -> [1, H, T, D]
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
            # [1, H, T, D] -> [T, H, D]
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
        """Inject Key and Value states into specified cache layer.

        Args:
            cache: Target SpecDynamicCache.
            layer_idx: Layer index to populate.
            k_tensor: Key tensor [T, H, D] or [1, H, T, D].
            v_tensor: Value tensor [T, H, D] or [1, H, T, D].
            delta_p: Relative RoPE displacement to apply to keys if not pre-shifted.
        """
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
        """Inject Key and Value states across all Transformer layers.

        Args:
            cache: Target SpecDynamicCache.
            layer_k_list: Sequence of key tensors per layer.
            layer_v_list: Sequence of value tensors per layer.
            delta_p: Relative RoPE shift to apply if not already rotated.
        """
        if len(layer_k_list) != len(layer_v_list):
            raise ValueError(
                f"Mismatch in layer counts: {len(layer_k_list)} keys vs {len(layer_v_list)} values"
            )

        for layer_idx, (k, v) in enumerate(zip(layer_k_list, layer_v_list)):
            self.inject_layer_kv(
                cache=cache,
                layer_idx=layer_idx,
                k_tensor=k,
                v_tensor=v,
                delta_p=delta_p,
            )
