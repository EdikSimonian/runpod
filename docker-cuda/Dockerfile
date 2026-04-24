FROM nvidia/cuda:12.8.0-base-ubuntu22.04

RUN apt-get update -y && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN ldconfig /usr/local/cuda-12.8/compat/

RUN python3 -m pip install --upgrade pip

# Split vLLM's heavyweight install across three separate RUN steps so Docker
# can pull them in parallel on first cold start. A single monolithic
# `pip install vllm[flashinfer]` produces one ~7.4 GB layer that serializes
# the pull; on a 20 MB/s host that's ~6 min wall time for one blob, since
# Docker doesn't parallelize *within* a layer.
#
# Three smaller concurrent layers (~3 GB + ~1.5 GB + ~2 GB) cut wall time
# closer to ~3 min on the same host.
#
# Pins sourced from https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/cuda.txt
# Bump all four (torch, torchvision, torchaudio, flashinfer) together with
# the vLLM version below if you upgrade.

# Layer 1 — torch stack (~3 GB, biggest single chunk).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
      "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
      --extra-index-url https://download.pytorch.org/whl/cu128

# Layer 2 — FlashInfer kernels (~1.5 GB).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
      "flashinfer-python==0.6.6" "flashinfer-cubin==0.6.6"

# Layer 3 — vLLM and remaining transitive deps (~2 GB). torch + flashinfer
# are already satisfied from layers 1 and 2, so pip skips them.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install "vllm==0.19.1" \
      --extra-index-url https://download.pytorch.org/whl/cu128

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
