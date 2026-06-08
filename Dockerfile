FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Install dependencies (stable layer - only rebuilds if pyproject.toml or uv.lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy source code (changes frequently - uses cached dependencies)
COPY . .

ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["bash", "-c", "/workspace/.venv/bin/python main.py create -p parameters/preprocessing/feature_distance.yaml && /workspace/.venv/bin/python main.py grid --parameters parameters/GNN_2018_fuzzy.yaml"]
