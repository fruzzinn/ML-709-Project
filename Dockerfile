# ML-709 Adversarial Agents Research - Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install dependencies
RUN uv sync

# Create directories
RUN mkdir -p experiments data/benchmarks

# Default command
CMD ["uv", "run", "python", "scripts/run_experiment.py", "--config", "configs/default.yaml"]
