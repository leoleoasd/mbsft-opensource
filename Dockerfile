ARG BASE_IMAGE=nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ENV LANG=en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential cmake python3-dev libnuma1 \
        locales openssh-server procps iproute2 && \
    locale-gen en_US.UTF-8 && \
    ssh-keygen -A && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# uv (Python/dependency manager) from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# s5cmd (S3 client, handy for syncing data/checkpoints) from its GitHub release
RUN curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz \
    | tar -xz -C /usr/local/bin s5cmd

# This image carries NO credentials. Provide tokens at runtime instead:
#   docker run -e HF_TOKEN=... -e WANDB_API_KEY=... <image>

COPY pyproject.toml /workdir/pyproject.toml
COPY uv.lock /workdir/uv.lock
COPY .python-version /workdir/.python-version
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN --mount=type=cache,id=uv-cache2,target=/root/.cache/uv <<'SH'
cd /workdir
uv sync --only-group build
SH

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv <<'SH'
source /workdir/.venv/bin/activate
J="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$J CTEST_PARALLEL_LEVEL=$J NPY_NUM_BUILD_JOBS=$J \
       CARGO_BUILD_JOBS=$J MAX_JOBS=$J MAKEFLAGS="-j$J -l$J" NINJAFLAGS="-j $J" \
       NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1
cd /workdir
uv sync
SH

RUN echo 'ulimit -n 65536' >> /root/.bashrc && \
    echo 'source /workdir/.venv/bin/activate' >> /root/.bashrc

COPY . /workdir

WORKDIR /workdir
