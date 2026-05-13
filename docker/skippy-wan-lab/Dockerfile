# syntax=docker/dockerfile:1.7

FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    git \
    build-essential \
    libdbus-1-dev \
    libssl-dev \
    ninja-build \
    pkg-config \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal

ENV PATH="/root/.cargo/bin:${PATH}"
ENV LLAMA_STAGE_BUILD_DIR=".deps/llama.cpp/build-stage-abi-static"
WORKDIR /src

RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    cargo install just --locked

COPY . .

RUN LLAMA_STAGE_USE_SCCACHE=0 just llama-build

RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/target \
    just skippy-wan-lab-build-bins && \
    mkdir -p /out && \
    cp /src/target/release/skippy-server /out/skippy-server && \
    cp /src/target/release/skippy-prompt /out/skippy-prompt && \
    cp /src/target/release/metrics-server /out/metrics-server && \
    cp /src/target/release/skippy-model-package /out/skippy-model-package

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    iproute2 \
    iputils-ping \
    jq \
    libdbus-1-3 \
    libgomp1 \
    libssl3 \
    netcat-openbsd \
    python3 \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/skippy-server /usr/local/bin/skippy-server
COPY --from=builder /out/skippy-prompt /usr/local/bin/skippy-prompt
COPY --from=builder /out/metrics-server /usr/local/bin/metrics-server
COPY --from=builder /out/skippy-model-package /usr/local/bin/skippy-model-package
COPY docker/skippy-wan-lab/entrypoint.sh /usr/local/bin/skippy-wan-lab

RUN chmod +x /usr/local/bin/skippy-wan-lab

ENV APP_ROLE=stage
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/skippy-wan-lab"]
