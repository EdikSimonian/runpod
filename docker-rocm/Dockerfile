# MI300X / ROCm image — tracks a semi-stable vLLM 0.20 preview from AMD.
# If this tag rolls off Docker Hub, pick a newer one from:
#   https://hub.docker.com/r/rocm/vllm-dev/tags
# Why not `rocm/vllm:latest`? That stable tag is pinned at vLLM 0.16.0, which
# does not include the Qwen3.5 architecture needed by Qwen3.6-27B.
FROM rocm/vllm-dev:preview_releases_v0.20.0_20260422

# Base image already has vLLM, torch, transformers, flashinfer (ROCm build), etc.
# Only install what our handler needs on top, with --no-deps so pip doesn't try
# to reinstall torch / transformers from PyPI wheels.
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade --no-deps -r /requirements.txt

ARG MODEL_NAME=""
ARG BASE_PATH="/runpod-volume"

ENV MODEL_NAME=$MODEL_NAME \
    BASE_PATH=$BASE_PATH \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false

COPY src /src

CMD ["python3", "/src/handler.py"]
