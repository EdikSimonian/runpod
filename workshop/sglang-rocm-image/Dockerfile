## qwen-sglang:rocm720-aiter112-sgmain
##
## SGLang from main + aiter v0.1.12.post1 + flydsl 0.1.2 on top of the
## official rocm720-mi30x base. Built for AMD MI300X / ROCm 7.2 hosts.
## See ./README.md for the why behind each layer.

FROM lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x

# 1. flydsl 0.1.2 — the bundled 0.0.1.dev disables aiter's CK/HIP runtime,
#    which breaks aiter.fmoe at SGLang init time.
RUN pip install --no-cache-dir flydsl==0.1.2

# 2. aiter v0.1.12.post1 — bundled v0.1.11.post1 is missing aiter.dtypes and
#    aiter.dynamic_per_tensor_quant that SGLang's quantization import chain
#    requires unconditionally (via quark/schemes/quark_w4a4_mxfp4.py).
RUN cd /opt && \
    git clone --depth 1 --recursive --branch v0.1.12.post1 \
        https://github.com/ROCm/aiter.git && \
    cd aiter && \
    MAX_JOBS=8 python3 setup.py install && \
    cd / && rm -rf /opt/aiter/3rdparty

# 3. SGLang from main — pulls in PR #23062 (Qwen3.6-FP8 _make_packed_weight_loader
#    scale-broadcast fix) and other post-v0.5.10.post1 fixes for the
#    Qwen3_5GatedDeltaNet weight-loading path. The base image installs SGLang
#    as an editable package whose path-hook points at /sgl-workspace/sglang/python;
#    we replace that source tree with main and reinstall in editable mode so
#    the path hook still resolves.
ARG SGLANG_REF=main
RUN cd /sgl-workspace && \
    mv sglang sglang.v0.5.10.post1.bak && \
    git clone --depth 1 --branch "${SGLANG_REF}" \
        https://github.com/sgl-project/sglang.git && \
    cd sglang/python && \
    pip install --no-cache-dir --no-deps -e . && \
    echo "SGLang $(git -C /sgl-workspace/sglang rev-parse --short HEAD) installed from ${SGLANG_REF}" \
        > /sgl-workspace/SGLANG_VERSION.txt
