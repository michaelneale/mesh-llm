# mesh-llm Fly.io console image
#
# This is a self-contained copy of docker/Dockerfile.client, kept in fly/
# to preserve the `fly deploy --config fly/console/fly.toml --dockerfile fly/Dockerfile`
# invocation used by the existing deployment.
#
# If you modify this file, also update docker/Dockerfile.client and vice versa.
# They should be structurally identical to avoid drift between the Fly deploy
# and the published ghcr.io/mesh-llm/mesh-llm:client image.
#
# Deploy:
#   fly deploy --config fly/console/fly.toml --dockerfile fly/Dockerfile

# syntax=docker/dockerfile:1.7

FROM node:24-alpine AS ui-builder
WORKDIR /app
COPY mesh-llm/ui/package.json mesh-llm/ui/package-lock.json ./
RUN --mount=type=cache,id=npm-cache,target=/root/.npm \
    npm ci
COPY mesh-llm/ui/ ./
RUN npm run build
# Output: /app/dist

FROM debian:bookworm-slim AS rust-deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git cmake ninja-build pkg-config libssl-dev libdbus-1-dev python3 build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    cargo install cargo-chef --version 0.1.68 --locked
WORKDIR /src
COPY Cargo.toml ./
COPY Cargo.lock ./
COPY mesh-llm/Cargo.toml mesh-llm/Cargo.toml
COPY mesh-llm/build.rs mesh-llm/build.rs
COPY mesh-llm/plugin/Cargo.toml mesh-llm/plugin/Cargo.toml
COPY mesh-llm/plugin/build.rs mesh-llm/plugin/build.rs
COPY mesh-llm/proto/ mesh-llm/proto/
COPY mesh-llm/src/ mesh-llm/src/
COPY mesh-llm/plugin/src/ mesh-llm/plugin/src/
COPY tools/xtask/Cargo.toml tools/xtask/Cargo.toml
COPY tools/xtask/src/ tools/xtask/src/
RUN mkdir -p mesh-llm/ui/dist && echo '<html></html>' > mesh-llm/ui/dist/index.html
RUN cargo chef prepare --recipe-path recipe.json
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/target \
    cargo chef cook --release --locked --recipe-path recipe.json

FROM rust-deps AS rust-builder
WORKDIR /src
RUN rm -rf mesh-llm/ui/dist
COPY --from=ui-builder /app/dist mesh-llm/ui/dist
# Restore real build scripts (cargo-chef stubs them during cook)
COPY mesh-llm/build.rs mesh-llm/build.rs
COPY mesh-llm/plugin/build.rs mesh-llm/plugin/build.rs
COPY mesh-llm/src/ mesh-llm/src/
COPY mesh-llm/plugin/src/ mesh-llm/plugin/src/
COPY mesh-llm/skills/ mesh-llm/skills/
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p mesh-llm && \
    mkdir -p /out && \
    cp /src/target/release/mesh-llm /out/mesh-llm

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=rust-builder /out/mesh-llm /usr/local/bin/mesh-llm
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ARG CMD=console
ENV APP_MODE=${CMD}
EXPOSE 3131 9337
ENTRYPOINT ["/entrypoint.sh"]
