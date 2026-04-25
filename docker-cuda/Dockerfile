FROM nvidia/cuda:12.9.1-base-ubuntu22.04

RUN apt-get update -y && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN ldconfig /usr/local/cuda-12.9/compat/

RUN python3 -m pip install --upgrade pip

# Base image and torch wheel CUDA version MUST match what vLLM's wheel was
# compiled against. vLLM 0.19.1's official Dockerfile sets CUDA_VERSION=12.9.1
# (https://github.com/vllm-project/vllm/blob/v0.19.1/docker/Dockerfile#L25),
# so we mirror that. Using cu128 here would crash at engine init with
# `torch.AcceleratorError: CUDA error: the provided PTX was compiled with an
# unsupported toolchain` from Marlin during process_weights_after_loading,
# because vLLM's published wheel embeds 12.9-targeted PTX that 12.8 hosts
# can't decode.
#
# Trade: this image will not run on RTX 5090 / Blackwell consumer hosts whose
# drivers cap at CUDA 12.8. For 5090 we'd need to either (a) wait for vLLM
# to ship a cu128-compatible wheel, or (b) build vLLM from source against
# 12.8. Both out of scope.

# Three separate RUN steps so Docker pulls layers in parallel on cold start
# (vs one ~7 GB blob serialized over a single TCP stream). Pins sourced from
# https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/cuda.txt —
# bump together with the vLLM version below.

# Layer 1 — torch stack (~3 GB, biggest single chunk).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
      "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
      --extra-index-url https://download.pytorch.org/whl/cu129

# Layer 2 — FlashInfer kernels (~1.5 GB).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
      "flashinfer-python==0.6.6" "flashinfer-cubin==0.6.6"

# Layer 3 — vLLM and remaining transitive deps (~2 GB). torch + flashinfer
# are already satisfied from layers 1 and 2, so pip skips them.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install "vllm==0.19.1" \
      --extra-index-url https://download.pytorch.org/whl/cu129

COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

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
