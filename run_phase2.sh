#!/usr/bin/env bash
# SpecCache Phase 2: Automated Verification and PCIe DMA Benchmark Pipeline
set -e

echo "======================================================================"
echo "SpecCache Phase 2: Hybrid Radix-Semantic Engine & Async DMA Pipeline"
echo "======================================================================"

# Step 1: Install dependencies
echo "[1/3] Ensuring package and dependencies are installed..."
pip install -e .[dev]

# Step 2: Run Phase 2 test suite
echo "[2/3] Running Phase 2 test suite (PinnedPool, Radix, FAISS, Async DMA)..."
pytest tests/test_phase2_engine.py -v

# Step 3: Run Phase 2 PCIe DMA benchmark
echo "[3/3] Running Phase 2 PCIe DMA and kernel overlap benchmark..."
python benchmarks/bench_phase2_dma.py --output-plot benchmarks/phase2_dma_overlap.png

echo "======================================================================"
echo "Phase 2 complete. Benchmark plot saved to: benchmarks/phase2_dma_overlap.png"
echo "======================================================================"
