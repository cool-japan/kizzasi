# Deployment and Serving

**Version 0.2.2** | Kizzasi AGSP — Network Adapters, Docker, and Container Orchestration

---

## 1. Overview

`kizzasi-inference` ships four feature-gated network adapters that expose the inference `Pipeline` over standard protocols. Each adapter wraps a `StreamingEngine`, handles wire framing and backpressure, and integrates with the common `NetworkAdapter` trait so they share a uniform `start / stop / is_running` lifecycle.

| Feature flag | Adapter types | Protocol | Primary use case |
|---|---|---|---|
| `rest` | `RestAdapter`, `RestServer` | HTTP/1.1 + HTTP/2 (axum) | Browser clients, one-off requests, curl-friendly testing |
| `grpc` | `GrpcAdapter`, `GrpcServer`, `InferenceService` | HTTP/2 gRPC (tonic) | High-throughput service-to-service, streaming RPCs |
| `websocket` | `WebSocketAdapter` | WebSocket (RFC 6455) | Real-time bidirectional streaming, browser clients |
| `mqtt` | `MqttAdapter` | MQTT 3.1.1 (rumqttc) | IoT edge devices, pub-sub fan-out, QoS delivery |
| `network` | all of the above | — | Enable all four with a single flag |

Enabling `rest` pulls in `axum` and `tower`. Enabling `grpc` pulls in `tonic` and `prost`. Enabling `websocket` or `mqtt` activates the `streaming` feature (which in turn activates `async` / tokio). All adapters compile only when their respective feature flag is set; none is included in the default build.

```toml
[dependencies]
kizzasi-inference = { version = "0.2", features = ["rest"] }
# or
kizzasi-inference = { version = "0.2", features = ["network"] }  # all adapters
```

---

## 2. REST Adapter

### 2.1 Feature Activation

```toml
kizzasi-inference = { version = "0.2", features = ["rest"] }
```

Dependencies pulled in: `axum`, `tower`, `tower-http` (CORS), `tokio`.

### 2.2 Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness and readiness probe; returns `{"status":"healthy","version":"..."}` |
| `POST` | `/infer` | Single-step or multi-step inference over JSON |
| `GET` | `/metrics` | Lightweight request-count and average-latency dump |

### 2.3 Request and Response Types

`POST /infer` accepts a JSON body of type `RestInferRequest`:

```rust
pub struct RestInferRequest {
    pub signal: Vec<f32>,          // required: input samples
    pub steps: Option<usize>,      // default 1
    pub temperature: Option<f32>,  // default 1.0 (greedy = 0.0)
}
```

On success it returns `RestInferResponse`:

```rust
pub struct RestInferResponse {
    pub prediction: Vec<f32>,  // output samples
    pub model_id: String,      // identifier of the serving model
    pub latency_ms: u64,       // wall-clock inference time
    pub steps_executed: usize, // number of autoregressive steps run
}
```

`GET /health` returns `HealthResponse`; `GET /metrics` returns `MetricsResponse` (totals and average latency tracked since server start).

### 2.4 Configuration

`RestConfig` carries four fields with the defaults shown below:

```rust
pub struct RestConfig {
    pub addr: String,              // default "0.0.0.0:8080"
    pub max_body_size: usize,      // default 1 MiB (1_048_576 bytes)
    pub request_timeout_ms: u64,   // default 30 000 ms
    pub cors_enabled: bool,        // default true (permissive CORS)
}
```

### 2.5 Usage

The `RestServer` wrapper handles OS shutdown signals automatically:

```rust
use kizzasi_inference::adapters::{RestConfig, RestServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = RestConfig::default();  // binds to 0.0.0.0:8080
    let server = RestServer::new(config);
    server.serve_until_shutdown().await
}
```

For custom shutdown logic (e.g., a tokio `oneshot` channel), use `RestAdapter` directly:

```rust
use kizzasi_inference::adapters::{RestAdapter, RestConfig};
use std::sync::Arc;

let config = RestConfig {
    addr: "127.0.0.1:9090".to_string(),
    ..Default::default()
};
let adapter = Arc::new(RestAdapter::new(config));

let (tx, rx) = tokio::sync::oneshot::channel::<()>();
adapter.serve_with_graceful_shutdown(async move { let _ = rx.await; }).await?;
// tx.send(()).ok();  // trigger shutdown from another task
```

### 2.6 Quick Test with curl

```bash
# Health check
curl http://localhost:8080/health

# Single-step inference
curl -s -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"signal":[0.5,0.3,-0.1],"steps":1}' | jq .

# Multi-step with temperature
curl -s -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"signal":[1.0],"steps":5,"temperature":0.8}' | jq .

# Metrics
curl http://localhost:8080/metrics
```

---

## 3. gRPC Adapter

### 3.1 Feature Activation

```toml
kizzasi-inference = { version = "0.2", features = ["grpc"] }
```

Dependencies: `tonic`, `prost`, `bytes`, `http`, `http-body`, `tower-service`. No `.proto` files or `build.rs` are required; service traits are hand-implemented using prost derive macros and a custom `ProstCodec`.

### 3.2 Service Definition

The gRPC service is named `kizzasi.inference.InferenceService` and exposes three RPC methods:

| Method | RPC type | Path |
|---|---|---|
| `Infer` | Unary | `/kizzasi.inference.InferenceService/Infer` |
| `InferStream` | Client-streaming | `/kizzasi.inference.InferenceService/InferStream` |
| `Health` | Unary | `/kizzasi.inference.InferenceService/Health` |

### 3.3 Message Types

```rust
// Request
pub struct InferenceRequest {
    pub request_id: String,
    pub input: Vec<f32>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
}

// Reply
pub struct InferenceReply {
    pub request_id: String,
    pub output: Vec<f32>,
    pub latency_ms: f64,
    pub num_tokens: u32,
}

// Health probe
pub struct HealthRequest {}
pub struct HealthReply { pub status: String, pub engine_state: String }
```

### 3.4 Configuration

```rust
pub struct GrpcServerConfig {
    pub addr: String,                   // default "127.0.0.1:50051"
    pub max_concurrent_streams: u32,    // default 256
    pub timeout_ms: u64,                // default 30 000
    pub max_frame_size: u32,            // default 16 384 bytes
}
```

### 3.5 Usage

```rust
use kizzasi_inference::adapters::grpc::{GrpcServer, GrpcServerConfig};
use kizzasi_inference::streaming::{StreamConfig, StreamingEngine};

let engine = StreamingEngine::new(StreamConfig::default())?;
let server = GrpcServer::from_config(GrpcServerConfig::default(), engine)?;

// Serve until runtime termination
server.serve_until_shutdown().await?;
```

Graceful shutdown with a signal:

```rust
server.serve_with_graceful_shutdown(async {
    tokio::signal::ctrl_c().await.ok();
}).await?;
```

---

## 4. WebSocket Adapter

### 4.1 Feature Activation

```toml
kizzasi-inference = { version = "0.2", features = ["websocket"] }
```

Activates the `streaming` feature (tokio + futures) and pulls in `tokio-tungstenite`.

### 4.2 Protocol

Each TCP connection is upgraded to WebSocket. The client sends JSON-encoded `InferenceMessage` frames (text or binary with the `msgpack` feature). The server replies with JSON-encoded `InferenceResponse` frames. The connection persists for the lifetime of the client session, allowing back-to-back inference calls without reconnection overhead.

### 4.3 Usage

```rust
use kizzasi_inference::adapters::WebSocketAdapter;
use kizzasi_inference::streaming::{StreamConfig, StreamingEngine};
use std::net::SocketAddr;

let engine = StreamingEngine::new(StreamConfig::default())?;
let addr: SocketAddr = "127.0.0.1:8081".parse()?;
let adapter = WebSocketAdapter::new(addr, engine);
adapter.serve().await?;
```

Message format (JSON text frames):

```bash
# Send an inference request
wscat -c ws://localhost:8081
> {"request_id":"req-1","input":[0.5,0.3],"config":{}}

# Receive response
< {"request_id":"req-1","output":[0.45,0.27],"latency_ms":0.8,"num_tokens":2}
```

---

## 5. MQTT Adapter

### 5.1 Feature Activation

```toml
kizzasi-inference = { version = "0.2", features = ["mqtt"] }
```

Activates the `streaming` feature and pulls in `rumqttc`.

### 5.2 Protocol

`MqttAdapter` subscribes to an *input topic* and publishes results to an *output topic*. Every message on the input topic is treated as a JSON-encoded `InferenceMessage`. After inference, a JSON-encoded `InferenceResponse` is published to the output topic with QoS `AtLeastOnce`. The adapter handles reconnection automatically (5-second back-off on connection errors).

### 5.3 Constructor

```rust
pub fn new(
    broker_url: &str,        // "mqtt://host:1883", "tcp://host:1883", or "host:1883"
    client_id: &str,         // unique client identifier
    input_topic: impl Into<String>,
    output_topic: impl Into<String>,
    engine: StreamingEngine,
) -> InferenceResult<Self>
```

### 5.4 Usage

```rust
use kizzasi_inference::adapters::MqttAdapter;
use kizzasi_inference::streaming::{StreamConfig, StreamingEngine};

let engine = StreamingEngine::new(StreamConfig::default())?;
let adapter = MqttAdapter::new(
    "mqtt://localhost:1883",
    "kizzasi-server",
    "agsp/infer/requests",
    "agsp/infer/responses",
    engine,
)?;

// Blocks, processing messages until adapter.stop() is called
adapter.run().await?;
```

Publishing an inference request from another client:

```bash
mosquitto_pub -h localhost -t agsp/infer/requests \
  -m '{"request_id":"iot-1","input":[0.1,0.2,0.3],"config":{}}'

mosquitto_sub -h localhost -t agsp/infer/responses
# Output: {"request_id":"iot-1","output":[...],"latency_ms":1.2,"num_tokens":3}
```

---

## 6. Docker

### 6.1 Dockerfile

The repository provides a two-stage `Dockerfile` at the workspace root.

**Stage 1 — builder** (`rust:1.89-slim`):

```dockerfile
FROM rust:1.89-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y pkg-config libssl-dev cmake \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
RUN cargo build --release --workspace --exclude kizzasi-python
```

**Stage 2 — runtime** (`debian:bookworm-slim`):

```dockerfile
FROM debian:bookworm-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y libssl3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/examples/ /app/examples/
RUN useradd -m -u 1000 kizzasi && chown -R kizzasi:kizzasi /app
USER kizzasi
EXPOSE 8080   # REST
EXPOSE 50051  # gRPC
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
CMD ["sh", "-c", "echo 'Kizzasi inference server ready (no main binary; mount and exec examples)'; sleep infinity"]
```

> **Important:** There is currently no default server binary. The `CMD` prints an informational message and sleeps indefinitely. To run an actual server, override `CMD` to point to one of the compiled example binaries, or mount your own binary into the container. See `crates/kizzasi-inference/examples/` for available examples that can be used as entry points.

### 6.2 Building and Running

```bash
# Build the image
docker build -t kizzasi:latest .

# Run with REST adapter example
docker run --rm -p 8080:8080 kizzasi:latest \
  /app/examples/basic_inference

# Override CMD to launch a specific example
docker run --rm -p 8080:8080 -p 50051:50051 kizzasi:latest \
  /app/examples/full_stack_agsp
```

### 6.3 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RUST_LOG` | `info` | Tracing log level (`trace`, `debug`, `info`, `warn`, `error`) |
| `KIZZASI_ADDR` | `0.0.0.0:8080` | Bind address passed to example binaries that respect it |

---

## 7. Docker Compose

The workspace includes a `docker-compose.yml` at the repository root that starts two services: the Kizzasi inference container and an Eclipse Mosquitto MQTT broker.

```yaml
version: '3.8'

services:
  kizzasi:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    image: kizzasi:latest
    ports:
      - "8080:8080"   # REST
      - "50051:50051" # gRPC
    environment:
      - RUST_LOG=info
      - KIZZASI_ADDR=0.0.0.0:8080
    volumes:
      - ./models:/app/models:ro  # pre-trained weights (read-only)
    depends_on:
      - mqtt-broker

  mqtt-broker:
    image: eclipse-mosquitto:2
    ports:
      - "1883:1883"  # MQTT
      - "9001:9001"  # MQTT over WebSockets
    volumes:
      - ./docker/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro
```

Starting the full stack:

```bash
# Build and start all services in the background
docker compose up --build -d

# Tail logs
docker compose logs -f kizzasi
docker compose logs -f mqtt-broker

# Health check
curl http://localhost:8080/health

# Stop everything
docker compose down
```

The `./models` directory on the host is mounted read-only at `/app/models` inside the container. Populate it with weight files before starting if your server binary reads from `/app/models`.

---

## 8. Choosing an Adapter

| Requirement | Recommended adapter |
|---|---|
| HTTP clients, REST tooling, curl | `rest` |
| High-throughput microservice calls | `grpc` |
| Browser WebSocket clients, streaming dashboards | `websocket` |
| IoT sensors, edge devices, pub-sub fan-out | `mqtt` |
| All protocols simultaneously | `network` (enables all four) |
| Embedded / no_std | See [Architecture Overview § 7.3](architecture_overview.md#73-embedded--nostd) |
| WebGPU / browser Wasm | See [Architecture Overview § 7.4](architecture_overview.md#74-webgpu) |

---

## 9. Further Reading

- [Architecture Overview](architecture_overview.md) — full crate dependency graph and core abstractions
- [Performance Tuning](performance_tuning.md) — SIMD acceleration, kernel fusion, batch throughput
- [Cross-Modal Integration](cross_modal_integration.md) — multi-modal pipelines and modality fusion
- [PyTorch Migration](pytorch_migration.md) — porting models from GGUF and SafeTensors checkpoints
- `crates/kizzasi-inference/examples/` — runnable server examples (REST, gRPC, WebSocket, MQTT)
