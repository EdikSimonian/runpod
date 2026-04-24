# MI300X / ROCm image — tracks a semi-stable vLLM 0.20 preview from AMD.
# If this tag rolls off Docker Hub, pick a newer one from:
#   https://hub.docker.com/r/rocm/vllm-dev/tags
# or drop back to `rocm/vllm:latest` once AMD publishes a stable build with vLLM >= 0.17.
#
# Why not `rocm/vllm:latest`? That tag is pinned at vLLM 0.16.0, which does not
# include the Qwen3.5 model architecture. Qwen3.6-27B requires vLLM >= 0.17.0.
FROM rocm/vllm-dev:preview_releases_v0.20.0_20260422

# worker-vllm handler + runpod SDK deps.
# The base image already has vLLM, torch, transformers, flashinfer (ROCm build),
# numpy, etc. Installing on top of it can drag pip resolver into reinstalling
# torch — so we install only what's missing.
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade --no-deps \
      runpod ray pandas pyarrow huggingface-hub packaging pydantic pydantic-settings hf-transfer kernels \
    && python3 -m pip install --upgrade bitsandbytes typing-extensions

ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

ENV PYTHONPATH="/:/vllm-workspace"

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

CMD ["python3", "/src/handler.py"]
