#!/usr/bin/env bash
# SpecCache Phase 3: Automated Verification and End-to-End LLM Serving Benchmark
set -e

echo "======================================================================"
echo "SpecCache Phase 3: End-to-End Serving & Speculative Verification"
echo "======================================================================"

# Step 1: Install dependencies
echo "[1/3] Ensuring package and dependencies are installed..."
pip install -e .[dev]

# Step 2: Run Phase 3 test suite
echo "[2/3] Running Phase 3 test suite (KVInjector, Speculative, Three-Path, SSE Server)..."
pytest tests/test_phase3_serving.py -v

# Step 3: Run Phase 3 End-to-End benchmark
echo "[3/3] Running Phase 3 End-to-End TTFT and Speculative Throughput benchmark..."
python benchmarks/bench_phase3_e2e.py --output-plot benchmarks/phase3_e2e_serving.png

echo "======================================================================"
echo "Phase 3 complete. Benchmark plot saved to: benchmarks/phase3_e2e_serving.png"
echo "======================================================================"
