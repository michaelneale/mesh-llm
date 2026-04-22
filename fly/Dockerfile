# mesh-llm Fly.io console image
#
# Deploy:
#   fly deploy --config fly/console/fly.toml --dockerfile fly/Dockerfile
#   Set MESH_HEADLESS=1 in fly.toml [env] to run API-only (no web UI)

# syntax=docker/dockerfile:1.7

FROM node:24-alpine AS ui-builder
WORKDIR /app
COPY mesh-llm/ui/package.json mesh-llm/ui/package-lock.json ./
RUN --mount=type=cache,id=npm-cache,target=/root/.npm \
    npm ci
COPY mesh-llm/ui/ ./
RUN npm run build

FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git cmake ninja-build pkg-config libssl-dev libdbus-1-dev python3 build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY mesh-llm/Cargo.toml mesh-llm/Cargo.toml
COPY mesh-llm/build.rs mesh-llm/build.rs
COPY mesh-llm/plugin/Cargo.toml mesh-llm/plugin/Cargo.toml
COPY mesh-llm/plugin/build.rs mesh-llm/plugin/build.rs
COPY mesh-llm/proto/ mesh-llm/proto/
COPY mesh-llm/ mesh-llm/
COPY mesh-client/ mesh-client/
COPY mesh-api/ mesh-api/
COPY mesh-host-core/ mesh-host-core/
COPY mesh-api-ffi/ mesh-api-ffi/
COPY mesh-llm-test-harness/ mesh-llm-test-harness/
COPY tools/xtask/ tools/xtask/
COPY --from=ui-builder /app/dist mesh-llm/ui/dist
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p mesh-llm && \
    cp target/release/mesh-llm /usr/local/bin/mesh-llm

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/mesh-llm /usr/local/bin/mesh-llm
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV APP_MODE=console
EXPOSE 3131 9337
ENTRYPOINT ["/entrypoint.sh"]
