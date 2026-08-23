# syntax=docker/dockerfile:1.7
#
# ShipInfer images. One file, four stages, because the alternative is four Dockerfiles that
# disagree about the CUDA version six weeks from now.
#
#   base      CUDA runtime + Python + torch. What every other stage stands on.
#   builder   base's packages plus the CUDA *toolkit* and CMake; builds the fused kernels.
#   dev       builder plus the developer tooling. Source is bind-mounted, not copied.
#   runtime   base plus the built artefacts only. No compiler, no toolkit, no source tree.
#
# The split exists for a specific reason: `nvidia/cuda:*-devel` is ~6 GB because it carries
# nvcc and the full toolkit, and none of that is needed to *run* an inference server. The
# runtime image copies the compiled `.so` out of the builder and leaves the toolchain behind.
#
# Build:
#   docker build -f deploy/docker/Dockerfile --target dev     -t shipinfer:dev .
#   docker build -f deploy/docker/Dockerfile --target runtime -t shipinfer:runtime .
# or just `make build` / `make runtime` from the repository root.

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04
# Must match the *driver* on the host, not the newest release. A cu130 build of torch on a
# 12.6 driver reports `torch.cuda.is_available() == False` and an unhelpful "driver is too
# old" — the single most common way this project is misconfigured.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1

# =====================================================================================
# base — the shared runtime layer
# =====================================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS base

ARG TORCH_INDEX
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 22.04 ships Python 3.10, which is exactly the project's floor — no deadsnakes PPA
# and no source build, both of which are ways for an image to break on a Tuesday.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        libgl1 libglib2.0-0 \
        ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" && pip install --upgrade pip setuptools wheel

# torch first and on its own layer: it is ~2.5 GB and it changes far less often than the
# project's own dependencies, so it deserves a cache layer of its own.
RUN pip install --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# A non-root user, created here so every downstream stage inherits it. A server that runs
# as root inside a container is one container escape away from being a problem.
RUN groupadd --gid 1000 shipinfer \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash shipinfer

WORKDIR /workspace

# =====================================================================================
# builder — the CUDA toolkit and the fused kernels
# =====================================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG TORCH_INDEX
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential cmake ninja-build git \
        libgl1 libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" && pip install --upgrade pip setuptools wheel
RUN pip install --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

WORKDIR /workspace

# Dependencies before source, so editing a .py file does not invalidate the dependency
# layer. pyproject.toml is the only input that should ever do that.
COPY pyproject.toml README.md ./
RUN pip install -e ".[dev,cli,server]" || pip install ".[dev,cli,server]"

COPY . .

# Every architecture in the fleet, not `native`: an image built for the machine that
# happened to run the build is an image that silently fails on the next node. Override with
# --build-arg CUDA_ARCHITECTURES=86 for a fast local build.
ARG CUDA_ARCHITECTURES="70;75;80;86;89"
ARG BUILD_NATIVE=1
RUN if [ "$BUILD_NATIVE" = "1" ]; then \
        python scripts/build_native.py --arch "${CUDA_ARCHITECTURES}" ; \
    else \
        echo "skipping the native extension (BUILD_NATIVE=0)" ; \
    fi

# =====================================================================================
# dev — what a developer and CI actually work in
# =====================================================================================
FROM builder AS dev

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        vim less procps htop openssh-client clang-format \
    && rm -rf /var/lib/apt/lists/*

# gh, so a PR can be opened from inside the container the same way it is outside it.
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# The source tree is bind-mounted over /workspace by compose, so anything COPYed above is
# shadowed at run time. That is the point: edit on the host, run in the container.
ENV SHIPINFER_MODEL_REPOSITORY=/workspace/model_repository
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]

# =====================================================================================
# runtime — the shipping image
# =====================================================================================
FROM base AS runtime

# The virtualenv carries the editable install's path file, so the source has to come too;
# the compiled extension comes from the builder because this stage has no compiler.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/src /workspace/src
COPY --from=builder /workspace/pyproject.toml /workspace/README.md /workspace/
COPY model_repository /workspace/model_repository
COPY deploy/docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh && chown -R shipinfer:shipinfer /workspace

USER shipinfer

ENV SHIPINFER_MODEL_REPOSITORY=/workspace/model_repository \
    SHIPINFER_HTTP__ENABLED=true \
    SHIPINFER_HTTP__HOST=0.0.0.0 \
    SHIPINFER_HTTP__PORT=8000 \
    SHIPINFER_OBSERVABILITY__LOG_SINK=async

EXPOSE 8000

# Readiness, not liveness: the process answers long before the engines have deserialised,
# and a health check that passes then sends traffic into a wall.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/v2/health/ready || exit 1

# tini reaps the worker threads' children and forwards SIGTERM, which is what makes the
# server's drain-on-shutdown actually run instead of being killed mid-batch.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["serve", "--http"]
