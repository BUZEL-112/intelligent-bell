#!/usr/bin/env bash
# SpecCache Phase 1: Automated Verification and Roofline Benchmark Pipeline
set -e

echo "======================================================================"
echo "SpecCache Phase 1: Verification & HBM Roofline Benchmark"
echo "======================================================================"

# Step 1: Install package in editable mode with dependencies
echo "[1/3] Ensuring package and dependencies are installed..."
pip install -e .[dev]

# Step 2: Execute rigorous pytest suite
echo "[2/3] Running numerical correctness test suite..."
pytest tests/test_correctness.py -v

# Step 3: Run roofline benchmark across GQA (Llama-3-8B) and MHA (70B) configs
echo "[3/3] Executing HBM roofline benchmark and generating dark-mode plot..."
python benchmarks/bench_roofline.py --output-plot benchmarks/phase1_roofline.png

echo "======================================================================"
echo "Phase 1 complete. Benchmark plot saved to: benchmarks/phase1_roofline.png"
echo "======================================================================"
