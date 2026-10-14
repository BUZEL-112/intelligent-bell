# ==============================================================================
# SpecCache Production Multi-Stage Dockerfile
# Base: NVIDIA CUDA 12.4.1 with PyTorch, OpenAI Triton, and FastAPI SSE Serving
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependency Resolution
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    git \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY speccache/ ./speccache/

RUN pip install --no-cache-dir --no-deps -e .

# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV SPECCACHE_HOST=0.0.0.0
ENV SPECCACHE_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create unprivileged application user
RUN useradd -m -u 1000 -s /bin/bash speccache
WORKDIR /app

# Copy application code and assets
COPY --chown=speccache:speccache pyproject.toml README.md ./
COPY --chown=speccache:speccache speccache/ ./speccache/
COPY --chown=speccache:speccache benchmarks/ ./benchmarks/
COPY --chown=speccache:speccache tests/ ./tests/

USER speccache

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/stats || exit 1

ENTRYPOINT ["uvicorn", "speccache.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
