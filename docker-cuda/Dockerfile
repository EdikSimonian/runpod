FROM nvidia/cuda:12.8.0-base-ubuntu22.04

RUN apt-get update -y && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN ldconfig /usr/local/cuda-12.8/compat/

# vLLM + FlashInfer built against CUDA 12.8 PyTorch wheels.
# Pinned to 0.19.1; bumping is a simple version bump here since our handler talks
# to vLLM's stable HTTP API rather than its Python internals.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "vllm[flashinfer]==0.19.1" --extra-index-url https://download.pytorch.org/whl/cu128

COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

# Build args carry through to runtime env for convenience; `handler.py` reads env.
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
