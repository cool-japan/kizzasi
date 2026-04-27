# Stage 1: Builder
FROM rust:1.89-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifest files first for layer caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Build all workspace crates in release mode, excluding Python bindings
# (kizzasi-python requires a Python interpreter at build time via PyO3)
RUN cargo build --release --workspace --exclude kizzasi-python

# Stage 2: Runtime
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy any compiled example binaries from the builder stage
COPY --from=builder /app/target/release/examples/ /app/examples/

# Create a non-root user for security
RUN useradd -m -u 1000 kizzasi \
    && chown -R kizzasi:kizzasi /app

USER kizzasi

# REST API port
EXPOSE 8080
# gRPC port
EXPOSE 50051

# Health check via REST endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default: print version info and keep container alive
# Override CMD or use docker-compose to run specific binaries/examples
CMD ["sh", "-c", "echo 'Kizzasi inference server ready (no main binary; mount and exec examples)'; sleep infinity"]
