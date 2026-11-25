# llamAmpere server image: wraps the prebuilt Linux release tar. Nothing is compiled here.
#
# The tar (llamampere-vX-linux-x86_64-cuda12-server.tar) already ships llama-server, the CUDA 12.4
# runtime, cuBLAS and its own GCC runtime, all found through RUNPATH $ORIGIN, so the base image only
# has to provide glibc 2.35+. The host needs an NVIDIA driver R550+ and the NVIDIA Container Toolkit.
#
# Build (the tar must be inside the build context):
#   docker build -f .devops/llamampere-server.Dockerfile \
#     --build-arg RELEASE_TAR=llamampere-v0.7.5-linux-x86_64-cuda12-server.tar -t llamampere .
# Run:
#   docker run --gpus all -v /path/to/models:/models -p 8080:8080 llamampere -m /models/model.gguf

ARG UBUNTU_VERSION=22.04

FROM docker.io/library/ubuntu:${UBUNTU_VERSION} AS unpack
ARG RELEASE_TAR
# ADD unpacks a local tar; the tar holds one top-level folder named after the release
ADD ${RELEASE_TAR} /unpack/
RUN set -eu; \
    test "$(ls /unpack | wc -l)" = 1; \
    mv /unpack/* /app; \
    test -x /app/llama-server; \
    cd /app && sha256sum --quiet -c SHA256SUMS

FROM docker.io/library/ubuntu:${UBUNTU_VERSION}
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

LABEL org.opencontainers.image.title="llamAmpere server" \
      org.opencontainers.image.description="llamAmpere llama-server for NVIDIA RTX 30-series (CUDA 12.4, sm_86)" \
      org.opencontainers.image.source="https://github.com/JakeATX/llamAmpere" \
      org.opencontainers.image.url="https://github.com/JakeATX/llamAmpere" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${APP_REVISION}" \
      org.opencontainers.image.licenses="MIT"

# ca-certificates: HTTPS model downloads (-hf); curl: the health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /models

COPY --from=unpack /app /app

# what the NVIDIA Container Toolkit mounts into the container
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080

WORKDIR /app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD [ "curl", "-fsS", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
