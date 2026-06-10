FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Install dependencies (stable layer - only rebuilds if pyproject.toml or uv.lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Reinstall PyTorch with B200 (sm_100) support using nightly build
RUN pip uninstall -y torch && \
    TORCH_CUDA_ARCH_LIST="5.0 6.0 7.0 7.5 8.0 8.6 9.0 10.0" \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/nightly/cu124

# Copy source code (changes frequently - uses cached dependencies)
COPY . .

ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["bash", "-c", "/workspace/.venv/bin/python main.py create -p parameters/preprocessing/feature_distance.yaml && /workspace/.venv/bin/python main.py grid --parameters parameters/GNN_2018_fuzzy.yaml"]
