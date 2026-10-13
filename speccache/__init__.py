"""SpecCache: Hardware-aware semantic KV-cache engine.

Phase 1 provides fused INT8 dequantization and relative RoPE phase-shifting
kernels targeting high-throughput memory-bandwidth bound LLM serving pipelines.
"""

from speccache.kernels.reference import (
    compute_rope_cos_sin_delta,
    dequant_and_rope_key_compiled,
    dequant_and_rope_key_ref,
    dequant_value_compiled,
    dequant_value_ref,
    pytorch_compiled_dequant_rope_shift,
    pytorch_eager_dequant_rope_shift,
    quantize_kv_int8,
    quantize_symmetric_int8,
    rotate_half,
)
from speccache.kernels.triton_rope_kv import (
    dequant_and_rope_key,
    dequant_kv_cache,
    dequant_value,
    fused_k_dequant_rope_shift_kernel,
    triton_fused_k_dequant_rope_shift,
)

__all__ = [
    "compute_rope_cos_sin_delta",
    "dequant_and_rope_key",
    "dequant_and_rope_key_compiled",
    "dequant_and_rope_key_ref",
    "dequant_kv_cache",
    "dequant_value",
    "dequant_value_compiled",
    "dequant_value_ref",
    "fused_k_dequant_rope_shift_kernel",
    "pytorch_compiled_dequant_rope_shift",
    "pytorch_eager_dequant_rope_shift",
    "quantize_kv_int8",
    "quantize_symmetric_int8",
    "rotate_half",
    "triton_fused_k_dequant_rope_shift",
]
