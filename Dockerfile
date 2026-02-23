# 1. Base Image
FROM python:3.12-slim-bookworm

# Tell uv to use the system python instead of looking for a venv
ENV UV_SYSTEM_PYTHON=1
# Ensure output is sent straight to logs
ENV PYTHONUNBUFFERED=1

# Add this near your other ENV lines in Dockerfile
ENV PGCHANNELBINDING=disable

# 2. Build Argument (Defaults to 'local', set to 'cloud' for Hugging Face)
ARG TARGET_ENV=local

# 2. Tell the OS where to find SSL certificates (Critical for Cloud SSL)
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# 3. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 libglib2.0-0 build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 4. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 5. Environment Variables
ENV UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/model_cache/fastembed

# 6. Copy blueprints
COPY pyproject.toml uv.lock ./

# 7. FLEXIBLE INSTALLATION LOGIC
# If local: Uses the 'cu118' index from your uv.lock (for your GTX 1050)
# If cloud: Overrides to the CPU-only index to save space and avoid CUDA errors
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGET_ENV" = "cloud" ]; then \
        echo "☁️ Building for Cloud (CPU)..." && \
        # 1. Install CPU-only Torch first
        uv pip install --system --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
        # 2. Install the rest of the project
        uv pip install --system . ; \
    else \
        echo "🎮 Building for Local (GPU)..." && \
        uv pip install --system . ; \
    fi

# 8. Copy project files
COPY . .

# 9. Final Project Install
RUN uv sync --frozen --no-dev

# 10. Hugging Face compatibility
EXPOSE 7860

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]