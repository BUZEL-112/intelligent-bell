# SpecCache: Phase 1 Fused INT8 RoPE KV-Cache Engine

SpecCache is a hardware-aware semantic KV-cache engine designed for high-throughput LLM serving.
Phase 1 provides an autotuned OpenAI Triton kernel that fuses per-token-head INT8 KV-cache dequantization with in-place relative Rotary Position Embedding (RoPE) phase-shifting.

---

## 1. Mathematical Grounding & System Motivation

When reusing cached Key states from prior requests or earlier document chunks, cached keys were originally rotated at token positions $m$. In the new context, the reused chunk lands at positions $m + \Delta p$.

By the rotational group homomorphism of RoPE in $SO(2)$:
$$R(\theta, m + \Delta p) = R(\theta, \Delta p) R(\theta, m)$$

Relative rotation can be applied directly to already-rotated keys without knowing their original absolute positions $m$.

### Memory Bandwidth Bottleneck in Conventional Serving

In conventional un-fused pipelines, shifting cached keys incurs heavy High Bandwidth Memory (HBM) roundtrips:
1. Load INT8 quantized keys from HBM (1 byte/element)
2. Dequantize to FP16 and store intermediate buffer to HBM (2 bytes/element)
3. Load intermediate FP16 keys from HBM (2 bytes/element)
4. Apply RoPE phase-shift and store rotated FP16 keys to HBM (2 bytes/element)

Total memory traffic in un-fused pipelines:
$$\text{Memory Traffic}_{\text{un-fused}} \approx 1 + 2 + 2 + 2 = 7 \text{ bytes per element}$$

### Single-Pass SRAM Fusion in SpecCache

SpecCache eliminates all intermediate HBM roundtrips by fusing dequantization and relative rotation inside on-chip SRAM/registers:
1. Load INT8 keys ($q_{x1}, q_{x2}$) from HBM in contiguous coalesced blocks (1 byte/element)
2. Load FP16 scales ($s$) and FP16 relative rotary factors ($\cos, \sin$)
3. Dequantize and rotate in registers using FP32 arithmetic precision
4. Store rotated FP16 keys directly to the destination buffer (2 bytes/element)

Total memory traffic in SpecCache:
$$\text{Memory Traffic}_{\text{fused}} \approx 1 + 2 = 3 \text{ bytes per element}$$

Because the arithmetic intensity of this operation is $\approx 1.0 \text{ FLOP/Byte}$ (far below the ridge points of modern GPUs such as A100 at $\approx 9.5 \text{ FLOP/Byte}$ or H100 at $\approx 20 \text{ FLOP/Byte}$), memory traffic reduction translates directly to a theoretical $7 / 3 \approx 2.33\times$ speedup.

---

## 2. Tensor Layout & RoPE Pairing Convention

All tensors follow contiguous `[num_tokens, num_kv_heads, head_dim]` memory layout:
- `k_quant`: `torch.int8` of shape `[T, H, D]` (symmetric INT8 range $[-127, 127]$)
- `k_scale`: `torch.float16` of shape `[T, H, 1]` (per-token-head scale)
- `cos_delta`: `torch.float16` of shape `[1, D // 2]` (uniform scalar shift) or `[T, D // 2]` (per-token shifts)
- `sin_delta`: `torch.float16` of shape `[1, D // 2]` or `[T, D // 2]`
- `k_out`: `torch.float16` of shape `[T, H, D]`

### HuggingFace Llama / Qwen Half-Rotated Style
For head dimension $D$, pair index $i \in [0, D/2)$ with index $i + D/2$:
$$\begin{aligned}
x_1 &= q_{x1} \cdot s \\
x_2 &= q_{x2} \cdot s \\
\text{out}_1 &= x_1 \cos(\Delta p \cdot \theta_i) - x_2 \sin(\Delta p \cdot \theta_i) \\
\text{out}_2 &= x_2 \cos(\Delta p \cdot \theta_i) + x_1 \sin(\Delta p \cdot \theta_i)
\end{aligned}$$

---

## 3. Repository Structure

```text
speccache/
├── __init__.py             # Public API exports
├── kernels/
│   ├── __init__.py
│   ├── reference.py        # Eager PyTorch reference + torch.compile baseline
│   └── triton_rope_kv.py   # Autotuned Triton kernels (Key dequant+RoPE, Value dequant)
├── tests/
│   ├── __init__.py
│   └── test_correctness.py # Pytest suite across shapes, shifts, and GQA configs
└── benchmarks/
    ├── __init__.py
    └── bench_roofline.py   # Triton do_bench profiling, GB/s calculation, and plot
```

---

## 4. Google Colab Setup & Execution Guide

To test and benchmark in a Google Colab GPU environment (e.g., T4, L4, V100, or A100):

### Step 1: Clone Repository and Install Dependencies
In a Colab notebook cell:
```bash
!git clone https://github.com/<your-repo>/speccache.git
%cd speccache
!pip install -e .[dev]
```

### Step 2: Run Verification Test Suite
```bash
!pytest tests/test_correctness.py -v
```

This verifies:
- Mathematical group homomorphism: $R(p_1 + p_2) = R(p_2) R(p_1)$
- Identity rotation at $\Delta p = 0$
- Uniform scalar shifts and per-token shifts
- Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and Multi-Head Attention (MHA)
- Pre-allocated destination buffers
- Strict numerical tolerance against eager PyTorch ground truth

### Step 3: Run HBM Roofline Benchmark & Generate Plots
```bash
!python benchmarks/bench_roofline.py --num-heads 8 --head-dim 128 --output-plot benchmarks/roofline_results.png
```

To display the generated plot inline in Colab:
```python
from IPython.display import Image
Image("benchmarks/roofline_results.png")
```

---

## 5. Python API Usage

```python
import torch
from speccache import (
    compute_rope_cos_sin_delta,
    dequant_and_rope_key,
    dequant_value,
    quantize_symmetric_int8,
)

# Allocate KV tensors on GPU
num_tokens, num_heads, head_dim = 2048, 8, 128
device = torch.device("cuda")

# Quantize raw keys and values
raw_k = torch.randn((num_tokens, num_heads, head_dim), dtype=torch.float32, device=device)
k_quant, k_scale = quantize_symmetric_int8(raw_k)

# Compute relative RoPE tables for a chunk displacement of +64 tokens
delta_positions = torch.tensor([64], device=device)
cos_delta, sin_delta = compute_rope_cos_sin_delta(delta_positions, head_dim, device=device)

# Execute fused kernel with optional pre-allocated output buffer
k_out = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)
dequant_and_rope_key(k_quant, k_scale, cos_delta, sin_delta, out=k_out)
```
