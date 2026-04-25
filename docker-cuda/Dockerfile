# Use vLLM's official image as the base. This image is built and published by
# the vLLM project itself, so the CUDA toolkit, torch, FlashInfer, and vLLM
# wheels are guaranteed to be ABI/PTX-compatible with each other. Building our
# own CUDA + torch + vllm install (as we did before with cu128 / cu129) led to
# `cudaErrorUnsupportedPtxVersion` from Marlin during engine init because the
# wheel's compiled PTX didn't match our base toolkit. Using the upstream image
# eliminates that variable entirely.
#
# Bump tag together with any vLLM version upgrade; available tags at
# https://hub.docker.com/r/vllm/vllm-openai/tags
FROM vllm/vllm-openai:v0.19.1

# Layer our handler's deps on top of vLLM's pre-baked environment.
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

# vllm-openai's default ENTRYPOINT runs vLLM's api_server. We replace it with
# our RunPod handler, which spawns api_server as a subprocess itself — that's
# how we keep the OpenAI-compat HTTP behavior while still satisfying RunPod's
# serverless worker contract.
ENTRYPOINT []
CMD ["python3", "/src/handler.py"]
