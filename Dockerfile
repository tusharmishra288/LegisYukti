# Base Image
FROM python:3.12-slim-bookworm

# Tell uv to use the system python instead of looking for a venv
ENV UV_SYSTEM_PYTHON=1
# Ensure output is sent straight to logs
ENV PYTHONUNBUFFERED=1

# Add this near your other ENV lines in Dockerfile
ENV PGCHANNELBINDING=disable

# Build Argument (Defaults to 'local', set to 'cloud' for Hugging Face)
ARG TARGET_ENV=local

# Tell the OS where to find SSL certificates (Critical for Cloud SSL)
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 libglib2.0-0 build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Environment Variables
ENV UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/model_cache/fastembed

# Copy blueprints
COPY pyproject.toml uv.lock ./

# FLEXIBLE INSTALLATION LOGIC
# If local: Uses the 'cu118' index from your uv.lock
# If cloud: Overrides to the CPU-only index to save space and avoid CUDA errors
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGET_ENV" = "cloud" ]; then \
        echo "☁️ Building for Cloud (CPU)..." && \
        # Install CPU-only Torch first
        uv pip install --system --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
        # Install the rest of the project
        uv pip install --system . ; \
    else \
        echo "🎮 Building for Local (GPU)..." && \
        uv pip install --system . ; \
    fi

# Copy project files
COPY . .

# Ensure the docs directory exists and has the right permissions
RUN mkdir -p /app/docs && chmod -R 777 /app/docs

# Final Project Install (allow lockfile updates if needed)
# Using --no-dev ensures dependencies can be resolved even if uv.lock is out of date.
# Remove --frozen to avoid build failures due to transient PyPI wheel availability.
RUN uv sync --no-dev

# Hugging Face compatibility
EXPOSE 7860

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
