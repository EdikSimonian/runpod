# Official Ollama image — CUDA 12.4 + Ubuntu 22.04 + Ollama pre-installed.
# Saves ~500 MB vs. installing Ollama from scratch and is always up-to-date.
FROM ollama/ollama:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── Python + FastAPI stack ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages fastapi "uvicorn[standard]" httpx

# ── Ollama runtime configuration ─────────────────────────────────────────────
# Model served
ENV MODEL_NAME=gemma4:e2b

# Bind on all interfaces so FastAPI can reach Ollama
ENV OLLAMA_HOST=0.0.0.0:11434

# Keep model resident; never unload between requests
ENV OLLAMA_KEEP_ALIVE=-1

# Flash Attention — large speed / memory gain on Ampere+
ENV OLLAMA_FLASH_ATTENTION=1

# Q8 KV-cache: saves ~30 % VRAM versus FP16 with negligible quality loss
ENV OLLAMA_KV_CACHE_TYPE=q8_0

# One parallel slot is safest for a 27 B model on a single GPU
ENV OLLAMA_NUM_PARALLEL=1

# Never load a second model by accident
ENV OLLAMA_MAX_LOADED_MODELS=1

# Allow requests from the handler inside the same container
ENV OLLAMA_ORIGINS=*

# ── Model storage ─────────────────────────────────────────────────────────────
# Default: local path (no network volume). Override with a RunPod network
# volume mounted at /runpod-volume — start.sh handles the switch at runtime.
ENV OLLAMA_MODELS=/root/.ollama/models

# ── (Optional) Bake the model into the image at build time ───────────────────
# Uncomment the block below to pre-download during `docker build`.
# Produces a ~16 GB image but eliminates cold-start download time entirely.
# Requires Docker BuildKit + a GPU-enabled builder, OR use --network=host.
#
# RUN ollama serve & \
#     sleep 10 && \
#     ollama pull ${MODEL_NAME} && \
#     pkill ollama

EXPOSE 80

# ── Application files ────────────────────────────────────────────────────────
COPY start.sh /start.sh
COPY server.py /server.py
RUN chmod +x /start.sh

# ollama/ollama sets ENTRYPOINT=["/bin/ollama"] — override so our script runs
ENTRYPOINT []
CMD ["/start.sh"]
