.PHONY: all install test test-phase1 test-phase2 test-phase3 benchmark bench-phase1 bench-phase2 bench-phase3 clean

all: install test benchmark

install:
	pip install -e .[dev]

test: test-phase1 test-phase2 test-phase3

test-phase1:
	pytest tests/test_correctness.py -v

test-phase2:
	pytest tests/test_phase2_engine.py -v

test-phase3:
	pytest tests/test_phase3_serving.py -v

benchmark: bench-phase1 bench-phase2 bench-phase3

bench-phase1:
	python benchmarks/bench_roofline.py --output-plot benchmarks/phase1_roofline.png

bench-phase2:
	python benchmarks/bench_phase2_dma.py --output-plot benchmarks/phase2_dma_pipeline.png

bench-phase3:
	python benchmarks/bench_phase3_e2e.py --output-plot benchmarks/phase3_e2e_serving.png

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .triton build dist *.egg-info
