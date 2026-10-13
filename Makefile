.PHONY: all install test benchmark clean

all: install test benchmark

install:
	pip install -e .[dev]

test:
	pytest tests/test_correctness.py -v

benchmark:
	python benchmarks/bench_roofline.py --output-plot benchmarks/phase1_roofline.png

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .triton build dist *.egg-info
