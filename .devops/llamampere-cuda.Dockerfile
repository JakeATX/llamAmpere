# llamAmpere llama-server image for NVIDIA Ampere cards (RTX 30-series, A-series).
#
# Build (from the repository root):
#   docker build -f .devops/llamampere-cuda.Dockerfile -t llamampere:server-cuda12 .
#   docker build -f .devops/llamampere-cuda.Dockerfile --build-arg CUDA_ARCH="80;86" -t llamampere:server-cuda12 .
#
# Run (see docs/docker.md for the full list of options):
#   docker run --gpus all --ulimit memlock=-1 --ipc=host -p 8080:8080 \
#     -v /path/to/models:/models -v llamampere-cache:/cache \
#     llamampere:server-cuda12 --preset 24dense -m /models/<model>.gguf
#
# The build matches the Linux release package: CUDA 12.4, g++-13 on Ubuntu 22.04, shared libraries with
# RUNPATH $ORIGIN, x86-64 AVX2/FMA/F16C CPU paths (no -march=native), no NCCL.

ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.4.1

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# 86 = RTX 30-series / A-series workstation cards; add 80 for A100 ("80;86").
ARG CUDA_ARCH=86
# Parallel compile jobs; the CUDA kernels need about 2 GB of RAM per job.
ARG BUILD_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc-13 g++-13 make ninja-build git ca-certificates curl python3 libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 ships CMake 3.22; use a current release from the official binaries.
ARG CMAKE_VERSION=3.31.6
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz \
      | tar -xz -C /opt && \
    ln -s /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake /usr/local/bin/cmake

ENV CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13

WORKDIR /app
COPY . .

RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
      -DGGML_CUDA_NCCL=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
      -DLLAMA_OPENSSL=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF && \
    cmake --build build -j ${BUILD_JOBS} --target llama-server llama-cli llama-mtmd-cli && \
    mkdir -p /out && \
    cp -P build/bin/llama-server build/bin/llama-cli build/bin/llama-mtmd-cli /out/ && \
    find build/bin -maxdepth 1 -name '*.so*' -exec cp -P {} /out/ \; && \
    cp LICENSE /out/ && \
    (git rev-parse HEAD > /out/COMMIT 2>/dev/null || true)

## Runtime image
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS server

ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
LABEL org.opencontainers.image.title="llamAmpere server" \
      org.opencontainers.image.description="llamAmpere llama-server for NVIDIA Ampere GPUs (CUDA 12)" \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.source="https://github.com/JakeATX/llamAmpere" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 libstdc++6 libssl3 curl ca-certificates && \
    apt-get purge -y software-properties-common gpg-agent && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Some hosts only mount the driver here.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH}

COPY --from=build /out/ /app/

# Prompt-cache disk tier and other caches go to /cache (mount a volume to keep them across restarts).
ENV XDG_CACHE_HOME=/cache \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080
RUN mkdir -p /cache /models
VOLUME ["/cache"]

WORKDIR /app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/llama-server"]
