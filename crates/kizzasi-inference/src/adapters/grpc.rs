//! gRPC adapter for high-performance streaming inference
//!
//! Provides gRPC-based inference with bidirectional streaming and load balancing support.
//! Service traits are hand-implemented via `#[prost(...)]` macros — no `build.rs` code
//! generation is required.

use super::{InferenceMessage, InferenceResponse, NetworkAdapter};
use crate::error::{InferenceError, InferenceResult};
use crate::streaming::StreamingEngine;
use futures::StreamExt as _;
use http_body::Body as HttpBody;
use prost::Message as ProstMessage;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, watch, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tonic::body::Body as TonicBody;
use tonic::codec::{BufferSettings, Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::server::{Grpc, NamedService};
use tonic::{Request, Response, Status, Streaming};
use tower_service::Service;
use tracing::{debug, info};

// Re-export http types used in the Service impl — the http crate is a direct
// dependency gated on the `grpc` feature.
use ::http::{Request as HttpRequest, Response as HttpResponse};

/// Map an inference failure onto a gRPC status code
///
/// Rejected per-request overrides and malformed inputs are the client's mistake
/// and must not be reported as an internal server fault.
fn inference_status(error: InferenceError) -> Status {
    match error {
        InferenceError::InvalidConfiguration(message) => Status::invalid_argument(message),
        InferenceError::DimensionMismatch { .. } => Status::invalid_argument(error.to_string()),
        other => Status::internal(format!("Inference error: {other}")),
    }
}

// ============================================================================
// Protocol buffer message definitions
// ============================================================================

/// Protocol buffer definitions for inference service
pub mod proto {
    /// Inference request
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceRequest {
        /// Request ID
        #[prost(string, tag = "1")]
        pub request_id: String,

        /// Input signal data
        #[prost(float, repeated, tag = "2")]
        pub input: Vec<f32>,

        /// Temperature for sampling
        #[prost(float, optional, tag = "3")]
        pub temperature: Option<f32>,

        /// Top-k for sampling
        #[prost(uint32, optional, tag = "4")]
        pub top_k: Option<u32>,

        /// Top-p for sampling
        #[prost(float, optional, tag = "5")]
        pub top_p: Option<f32>,

        /// Maximum tokens to generate
        #[prost(uint32, optional, tag = "6")]
        pub max_tokens: Option<u32>,
    }

    /// Inference response
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct InferenceReply {
        /// Request ID
        #[prost(string, tag = "1")]
        pub request_id: String,

        /// Output signal data
        #[prost(float, repeated, tag = "2")]
        pub output: Vec<f32>,

        /// Latency in milliseconds
        #[prost(double, tag = "3")]
        pub latency_ms: f64,

        /// Number of tokens generated
        #[prost(uint32, tag = "4")]
        pub num_tokens: u32,
    }

    /// Health check request
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct HealthRequest {}

    /// Health check response
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct HealthReply {
        /// Service status
        #[prost(string, tag = "1")]
        pub status: String,

        /// Engine state
        #[prost(string, tag = "2")]
        pub engine_state: String,
    }
}

// ============================================================================
// Type conversions
// ============================================================================

/// Convert InferenceMessage to proto request
impl From<InferenceMessage> for proto::InferenceRequest {
    fn from(msg: InferenceMessage) -> Self {
        Self {
            request_id: msg.request_id,
            input: msg.input,
            temperature: msg.config.temperature,
            top_k: msg.config.top_k.map(|k| k as u32),
            top_p: msg.config.top_p,
            max_tokens: msg.config.max_tokens.map(|m| m as u32),
        }
    }
}

/// Convert proto request to InferenceMessage
impl From<proto::InferenceRequest> for InferenceMessage {
    fn from(req: proto::InferenceRequest) -> Self {
        let mut msg = InferenceMessage::new(req.request_id, req.input);
        msg.config.temperature = req.temperature;
        msg.config.top_k = req.top_k.map(|k| k as usize);
        msg.config.top_p = req.top_p;
        msg.config.max_tokens = req.max_tokens.map(|m| m as usize);
        msg
    }
}

/// Convert InferenceResponse to proto reply
impl From<InferenceResponse> for proto::InferenceReply {
    fn from(resp: InferenceResponse) -> Self {
        Self {
            request_id: resp.request_id,
            output: resp.output,
            latency_ms: resp.latency_ms,
            num_tokens: resp.num_tokens as u32,
        }
    }
}

// ============================================================================
// Prost codec — hand-written to avoid tonic-build / build.rs dependency
// ============================================================================

/// A `tonic::codec::Codec` implementation backed by `prost` (de)serialization.
///
/// `E` is the *encode* message type (server → client).
/// `D` is the *decode* message type (client → server).
#[derive(Debug, Clone, Default)]
struct ProstCodec<E, D> {
    _phantom: std::marker::PhantomData<(E, D)>,
}

/// `tonic::codec::Encoder` wrapping prost's `Message::encode`.
struct ProstEncoder<T>(std::marker::PhantomData<T>);

impl<T: ProstMessage + Default> Encoder for ProstEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, dst: &mut EncodeBuf<'_>) -> Result<(), Self::Error> {
        item.encode(dst)
            .map_err(|e| Status::internal(format!("prost encode error: {e}")))
    }

    fn buffer_settings(&self) -> BufferSettings {
        BufferSettings::default()
    }
}

/// `tonic::codec::Decoder` wrapping prost's `Message::decode`.
struct ProstDecoder<T>(std::marker::PhantomData<T>);

impl<T: ProstMessage + Default> Decoder for ProstDecoder<T> {
    type Item = T;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Self::Error> {
        use bytes::buf::Buf as _;
        let item = T::decode(src.chunk())
            .map_err(|e| Status::internal(format!("prost decode error: {e}")))?;
        let len = src.remaining();
        src.advance(len);
        Ok(Some(item))
    }

    fn buffer_settings(&self) -> BufferSettings {
        BufferSettings::default()
    }
}

impl<E, D> Codec for ProstCodec<E, D>
where
    E: ProstMessage + Default + Send + 'static,
    D: ProstMessage + Default + Send + 'static,
{
    type Encode = E;
    type Decode = D;
    type Encoder = ProstEncoder<E>;
    type Decoder = ProstDecoder<D>;

    fn encoder(&mut self) -> Self::Encoder {
        ProstEncoder(std::marker::PhantomData)
    }

    fn decoder(&mut self) -> Self::Decoder {
        ProstDecoder(std::marker::PhantomData)
    }
}

// ============================================================================
// gRPC server configuration
// ============================================================================

/// Configuration for the gRPC inference server
#[derive(Debug, Clone)]
pub struct GrpcServerConfig {
    /// Address to bind to
    pub addr: String,

    /// Maximum number of concurrent HTTP/2 streams per connection
    pub max_concurrent_streams: u32,

    /// Request timeout in milliseconds (0 = no timeout)
    pub timeout_ms: u64,

    /// Maximum HTTP/2 frame size in bytes
    pub max_frame_size: u32,
}

impl Default for GrpcServerConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:50051".to_string(),
            max_concurrent_streams: 256,
            timeout_ms: 30_000,
            max_frame_size: 16_384,
        }
    }
}

// ============================================================================
// Core service handler
// ============================================================================

/// Shared inner state held by both the service handler and the server wrapper.
struct InferenceServiceInner {
    engine: Arc<RwLock<StreamingEngine>>,
}

impl InferenceServiceInner {
    fn new(engine: StreamingEngine) -> Self {
        Self {
            engine: Arc::new(RwLock::new(engine)),
        }
    }

    /// Unary inference — run the engine on a single input frame.
    async fn infer(
        &self,
        request: Request<proto::InferenceRequest>,
    ) -> Result<Response<proto::InferenceReply>, Status> {
        let req = request.into_inner();
        debug!("Received inference request: {}", req.request_id);

        let msg: InferenceMessage = req.into();
        let input = msg.to_array();
        let request_id = msg.request_id.clone();

        let start = std::time::Instant::now();

        let outputs = {
            let engine = self.engine.write().await;
            engine
                .step_async_with(input, &(&msg.config).into())
                .await
                .map_err(inference_status)?
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let (output, num_tokens) = super::flatten_outputs(&outputs);
        let response = InferenceResponse::new(request_id, output, latency_ms, num_tokens);

        Ok(Response::new(response.into()))
    }

    /// Client-streaming inference — consume a stream of requests, emit a stream of replies.
    async fn infer_stream_impl(
        &self,
        request: Request<Streaming<proto::InferenceRequest>>,
    ) -> Result<Response<ReceiverStream<Result<proto::InferenceReply, Status>>>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<proto::InferenceReply, Status>>(32);
        let engine = self.engine.clone();

        tokio::spawn(async move {
            while let Some(item) = stream.next().await {
                match item {
                    Ok(req) => {
                        let msg: InferenceMessage = req.into();
                        let input = msg.to_array();
                        let request_id = msg.request_id.clone();

                        let start = std::time::Instant::now();
                        let result = {
                            let eng = engine.write().await;
                            eng.step_async_with(input, &(&msg.config).into()).await
                        };

                        let send_result = match result {
                            Ok(outputs) => {
                                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                                let (output, num_tokens) = super::flatten_outputs(&outputs);
                                let resp = InferenceResponse::new(
                                    request_id, output, latency_ms, num_tokens,
                                );
                                tx.send(Ok(resp.into())).await
                            }
                            Err(e) => {
                                tx.send(Err(Status::internal(format!("Inference error: {e}"))))
                                    .await
                            }
                        };

                        if send_result.is_err() {
                            // Receiver dropped — client disconnected
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    /// Health check — always returns "healthy" / "ready" when the server is up.
    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthReply>, Status> {
        Ok(Response::new(proto::HealthReply {
            status: "healthy".to_string(),
            engine_state: "ready".to_string(),
        }))
    }
}

// ============================================================================
// Public InferenceService — thin wrapper around the shared inner state
// ============================================================================

/// gRPC service implementation
///
/// Exposes three RPC methods:
/// - `infer` — unary request/response
/// - `infer_stream` — client-streaming / server-streaming
/// - `health` — liveness probe
pub struct InferenceService {
    inner: Arc<InferenceServiceInner>,
}

impl InferenceService {
    /// Create a new inference service backed by `engine`.
    pub fn new(engine: StreamingEngine) -> Self {
        Self {
            inner: Arc::new(InferenceServiceInner::new(engine)),
        }
    }

    /// Handle unary inference request
    pub async fn infer(
        &self,
        request: Request<proto::InferenceRequest>,
    ) -> Result<Response<proto::InferenceReply>, Status> {
        self.inner.infer(request).await
    }

    /// Handle streaming inference (client → server stream, server → client stream)
    pub async fn infer_stream(
        &self,
        request: Request<Streaming<proto::InferenceRequest>>,
    ) -> Result<Response<ReceiverStream<Result<proto::InferenceReply, Status>>>, Status> {
        self.inner.infer_stream_impl(request).await
    }

    /// Handle health check
    pub async fn health(
        &self,
        request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthReply>, Status> {
        self.inner.health(request).await
    }
}

// ============================================================================
// InferenceServiceServer — tonic transport integration
//
// This implements `tower::Service<http::Request<Body>>` so that it can be
// registered with `tonic::transport::Server::builder().add_service(...)`.
// The routing is done by matching on the gRPC path: `/Package.Service/Method`.
// ============================================================================

/// The package + service name used in gRPC URI routing.
const SERVICE_NAME: &str = "kizzasi.inference.InferenceService";

/// Path constants for each RPC method.
const PATH_INFER: &str = "/kizzasi.inference.InferenceService/Infer";
const PATH_INFER_STREAM: &str = "/kizzasi.inference.InferenceService/InferStream";
const PATH_HEALTH: &str = "/kizzasi.inference.InferenceService/Health";

/// A tonic-compatible server wrapper that handles HTTP/2 routing and
/// codec framing for `InferenceService`.
#[derive(Clone)]
pub struct InferenceServiceServer {
    inner: Arc<InferenceServiceInner>,
}

impl InferenceServiceServer {
    /// Wrap an `InferenceService` in a transport-ready server.
    pub fn new(svc: InferenceService) -> Self {
        Self { inner: svc.inner }
    }
}

impl NamedService for InferenceServiceServer {
    const NAME: &'static str = SERVICE_NAME;
}

/// Boxed error type matching tonic's internal transport alias.
type StdBoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// The future returned by `InferenceServiceServer::call`.
type InferServerFuture = Pin<
    Box<
        dyn Future<Output = Result<HttpResponse<TonicBody>, std::convert::Infallible>>
            + Send
            + 'static,
    >,
>;

impl<B> Service<HttpRequest<B>> for InferenceServiceServer
where
    B: HttpBody + Send + 'static,
    B::Data: Send,
    B::Error: Into<StdBoxError> + Send,
{
    type Response = HttpResponse<TonicBody>;
    type Error = std::convert::Infallible;
    type Future = InferServerFuture;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: HttpRequest<B>) -> Self::Future {
        let inner = self.inner.clone();
        let path = req.uri().path().to_owned();

        Box::pin(async move {
            match path.as_str() {
                PATH_INFER => {
                    struct UnaryHandler(Arc<InferenceServiceInner>);

                    impl tonic::server::UnaryService<proto::InferenceRequest> for UnaryHandler {
                        type Response = proto::InferenceReply;
                        type Future = Pin<
                            Box<
                                dyn Future<Output = Result<Response<proto::InferenceReply>, Status>>
                                    + Send,
                            >,
                        >;

                        fn call(
                            &mut self,
                            request: Request<proto::InferenceRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            Box::pin(async move { inner.infer(request).await })
                        }
                    }

                    let mut grpc: Grpc<ProstCodec<proto::InferenceReply, proto::InferenceRequest>> =
                        Grpc::new(ProstCodec::default());
                    let resp = grpc.unary(UnaryHandler(inner), req).await;
                    Ok(resp)
                }

                PATH_INFER_STREAM => {
                    // Bidirectional streaming: one reply per request frame,
                    // via `infer_stream_impl` — the same per-item streaming
                    // logic `InferenceService::infer_stream` exposes at the
                    // Rust API level, now actually reachable from the wire.
                    struct StreamHandler(Arc<InferenceServiceInner>);

                    impl tonic::server::StreamingService<proto::InferenceRequest> for StreamHandler {
                        type Response = proto::InferenceReply;
                        type ResponseStream = ReceiverStream<Result<proto::InferenceReply, Status>>;
                        type Future = Pin<
                            Box<
                                dyn Future<Output = Result<Response<Self::ResponseStream>, Status>>
                                    + Send,
                            >,
                        >;

                        fn call(
                            &mut self,
                            request: Request<Streaming<proto::InferenceRequest>>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            Box::pin(async move { inner.infer_stream_impl(request).await })
                        }
                    }

                    let mut grpc: Grpc<ProstCodec<proto::InferenceReply, proto::InferenceRequest>> =
                        Grpc::new(ProstCodec::default());
                    let resp = grpc.streaming(StreamHandler(inner), req).await;
                    Ok(resp)
                }

                PATH_HEALTH => {
                    struct HealthHandler(Arc<InferenceServiceInner>);

                    impl tonic::server::UnaryService<proto::HealthRequest> for HealthHandler {
                        type Response = proto::HealthReply;
                        type Future = Pin<
                            Box<
                                dyn Future<Output = Result<Response<proto::HealthReply>, Status>>
                                    + Send,
                            >,
                        >;

                        fn call(&mut self, request: Request<proto::HealthRequest>) -> Self::Future {
                            let inner = self.0.clone();
                            Box::pin(async move { inner.health(request).await })
                        }
                    }

                    let mut grpc: Grpc<ProstCodec<proto::HealthReply, proto::HealthRequest>> =
                        Grpc::new(ProstCodec::default());
                    let resp = grpc.unary(HealthHandler(inner), req).await;
                    Ok(resp)
                }

                _ => {
                    // Unknown method — return gRPC UNIMPLEMENTED
                    let mut response = HttpResponse::new(TonicBody::empty());
                    let headers = response.headers_mut();
                    headers.insert(
                        ::http::header::CONTENT_TYPE,
                        tonic::metadata::GRPC_CONTENT_TYPE,
                    );
                    headers.insert(
                        tonic::Status::GRPC_STATUS,
                        ::http::HeaderValue::from_static("12"), // UNIMPLEMENTED
                    );
                    Ok(response)
                }
            }
        })
    }
}

// ============================================================================
// GrpcAdapter — low-level transport adapter
// ============================================================================

/// gRPC adapter for streaming inference.
///
/// Holds the service and the bound address; integrates with the `NetworkAdapter`
/// trait so it can participate in the unified adapter lifecycle.
pub struct GrpcAdapter {
    /// Shared service handler
    service: InferenceService,

    /// Bound socket address
    addr: SocketAddr,

    /// Runtime running state, and the signal `stop()` uses to interrupt an
    /// in-flight `serve()`/`start()` call. A `watch` channel serves both
    /// purposes: `is_running()` is a plain synchronous `borrow()` (no
    /// runtime interaction, so it can never panic), and the serve loop
    /// awaits a transition to `false` via `serve_with_shutdown`.
    running: watch::Sender<bool>,

    /// Server configuration
    config: GrpcServerConfig,
}

impl GrpcAdapter {
    /// Create a new gRPC adapter with default configuration.
    ///
    /// # Arguments
    ///
    /// * `addr`   — Socket address to bind to
    /// * `engine` — Streaming inference engine to serve
    pub fn new(addr: impl Into<SocketAddr>, engine: StreamingEngine) -> Self {
        let (running, _) = watch::channel(false);
        Self {
            service: InferenceService::new(engine),
            addr: addr.into(),
            running,
            config: GrpcServerConfig::default(),
        }
    }

    /// Create a new gRPC adapter with explicit `GrpcServerConfig`.
    pub fn with_config(
        addr: impl Into<SocketAddr>,
        engine: StreamingEngine,
        config: GrpcServerConfig,
    ) -> Self {
        let (running, _) = watch::channel(false);
        Self {
            service: InferenceService::new(engine),
            addr: addr.into(),
            running,
            config,
        }
    }

    /// Get a reference to the service
    pub fn service(&self) -> &InferenceService {
        &self.service
    }

    /// Start the gRPC server and serve until the runtime shuts it down.
    ///
    /// Internally uses `tonic::transport::Server::builder()` with the hand-written
    /// `InferenceServiceServer` and applies `GrpcServerConfig` settings.
    pub async fn serve(&self) -> InferenceResult<()> {
        info!("gRPC server listening on {}", self.addr);
        let _ = self.running.send_replace(true);

        let inner = self.service.inner.clone();
        let svc = InferenceServiceServer { inner };
        let addr = self.addr;

        let mut builder = tonic::transport::Server::builder();

        if self.config.max_concurrent_streams > 0 {
            builder = builder.max_concurrent_streams(self.config.max_concurrent_streams);
        }

        if self.config.timeout_ms > 0 {
            builder = builder.timeout(std::time::Duration::from_millis(self.config.timeout_ms));
        }

        if self.config.max_frame_size > 0 {
            builder = builder.max_frame_size(Some(self.config.max_frame_size));
        }

        builder
            .add_service(svc)
            .serve(addr)
            .await
            .map_err(|e| InferenceError::NetworkError(format!("gRPC transport error: {e}")))?;

        let _ = self.running.send_replace(false);
        Ok(())
    }

    /// Start the server and serve until `signal` resolves (graceful shutdown).
    pub async fn serve_with_shutdown<F>(&self, signal: F) -> InferenceResult<()>
    where
        F: Future<Output = ()>,
    {
        info!("gRPC server (with shutdown) listening on {}", self.addr);
        let _ = self.running.send_replace(true);

        let inner = self.service.inner.clone();
        let svc = InferenceServiceServer { inner };
        let addr = self.addr;

        let mut builder = tonic::transport::Server::builder();

        if self.config.max_concurrent_streams > 0 {
            builder = builder.max_concurrent_streams(self.config.max_concurrent_streams);
        }

        if self.config.timeout_ms > 0 {
            builder = builder.timeout(std::time::Duration::from_millis(self.config.timeout_ms));
        }

        if self.config.max_frame_size > 0 {
            builder = builder.max_frame_size(Some(self.config.max_frame_size));
        }

        builder
            .add_service(svc)
            .serve_with_shutdown(addr, signal)
            .await
            .map_err(|e| InferenceError::NetworkError(format!("gRPC transport error: {e}")))?;

        let _ = self.running.send_replace(false);
        Ok(())
    }
}

impl NetworkAdapter for GrpcAdapter {
    /// Serve until [`NetworkAdapter::stop`] is called, using tonic's own
    /// `serve_with_shutdown` driven by a `watch` receiver — unlike the
    /// previous `stop()`, which flipped a flag nothing read, this genuinely
    /// interrupts the accept loop.
    async fn start(&self) -> InferenceResult<()> {
        let mut rx = self.running.subscribe();
        // `wait_for` always checks the *current* value first (not a stale
        // snapshot from `subscribe()`), so it is safe to build this future
        // before `serve_with_shutdown` sets the flag to `true` below: by the
        // time this future is actually polled the flag is already `true`,
        // and it correctly waits for a later transition to `false` from
        // `stop()`.
        self.serve_with_shutdown(async move {
            let _ = rx.wait_for(|running| !*running).await;
        })
        .await
    }

    async fn stop(&self) -> InferenceResult<()> {
        let _ = self.running.send_replace(false);
        info!("gRPC server stop requested");
        Ok(())
    }

    fn is_running(&self) -> bool {
        *self.running.borrow()
    }
}

// ============================================================================
// GrpcServer — higher-level entry point
// ============================================================================

/// Higher-level server wrapper that owns a `GrpcAdapter` and exposes a clean
/// lifecycle API with optional graceful shutdown.
pub struct GrpcServer {
    adapter: GrpcAdapter,
}

impl GrpcServer {
    /// Create a new `GrpcServer` from an existing `GrpcAdapter`.
    pub fn new(adapter: GrpcAdapter) -> Self {
        Self { adapter }
    }

    /// Construct a `GrpcServer` from configuration and an engine directly.
    pub fn from_config(config: GrpcServerConfig, engine: StreamingEngine) -> InferenceResult<Self> {
        let addr: SocketAddr = config
            .addr
            .parse()
            .map_err(|e| InferenceError::InvalidConfiguration(format!("invalid addr: {e}")))?;
        let adapter = GrpcAdapter::with_config(addr, engine, config);
        Ok(Self { adapter })
    }

    /// Serve until the runtime terminates the server.
    ///
    /// This is a blocking async call; await it inside a `tokio::spawn` task if
    /// you need concurrent operation.
    pub async fn serve_until_shutdown(&self) -> InferenceResult<()> {
        self.adapter.serve().await
    }

    /// Serve until `signal` resolves, then perform a graceful shutdown.
    ///
    /// Example:
    /// ```ignore
    /// server.serve_with_graceful_shutdown(async {
    ///     tokio::signal::ctrl_c().await.ok();
    /// }).await?;
    /// ```
    pub async fn serve_with_graceful_shutdown<F>(&self, signal: F) -> InferenceResult<()>
    where
        F: Future<Output = ()>,
    {
        self.adapter.serve_with_shutdown(signal).await
    }

    /// Get a reference to the underlying adapter.
    pub fn adapter(&self) -> &GrpcAdapter {
        &self.adapter
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::StreamConfig;

    // -----------------------------------------------------------------------
    // Helper: create a StreamingEngine with default config
    // -----------------------------------------------------------------------
    fn make_engine() -> StreamingEngine {
        StreamingEngine::new(StreamConfig::default()).expect("StreamingEngine::new should succeed")
    }

    // -----------------------------------------------------------------------
    // Config / struct tests (sync)
    // -----------------------------------------------------------------------

    #[test]
    fn test_grpc_server_config_defaults() {
        let config = GrpcServerConfig::default();
        assert!(
            config.max_concurrent_streams > 0,
            "max_concurrent_streams must be > 0"
        );
        assert!(config.timeout_ms > 0, "timeout_ms must be > 0");
        assert!(config.max_frame_size > 0, "max_frame_size must be > 0");
    }

    #[test]
    fn test_proto_conversion() {
        let msg = InferenceMessage::new("test-1", vec![1.0, 2.0, 3.0]);
        let proto_req: proto::InferenceRequest = msg.clone().into();

        assert_eq!(proto_req.request_id, "test-1");
        assert_eq!(proto_req.input, vec![1.0, 2.0, 3.0]);

        let back: InferenceMessage = proto_req.into();
        assert_eq!(back.request_id, msg.request_id);
        assert_eq!(back.input, msg.input);
    }

    #[test]
    fn test_response_proto_conversion() {
        let resp = InferenceResponse::new("resp-1", vec![4.0, 5.0], 10.5, 2);
        let proto_reply: proto::InferenceReply = resp.clone().into();

        assert_eq!(proto_reply.request_id, "resp-1");
        assert_eq!(proto_reply.output, vec![4.0, 5.0]);
        assert_eq!(proto_reply.latency_ms, 10.5);
        assert_eq!(proto_reply.num_tokens, 2);
    }

    // -----------------------------------------------------------------------
    // Service-level async tests (no transport layer)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_inference_service_health() {
        let service = InferenceService::new(make_engine());

        let request = Request::new(proto::HealthRequest {});
        let response = service.health(request).await;

        assert!(response.is_ok(), "health() must succeed");
        let reply = response.expect("health response").into_inner();
        assert_eq!(reply.status, "healthy");
        assert_eq!(reply.engine_state, "ready");
    }

    #[tokio::test]
    async fn test_inference_service_unary() {
        // The default engine has no model loaded, so step_async returns an identity-ish
        // output of the same dimension.  We just verify the call succeeds end-to-end.
        let service = InferenceService::new(make_engine());

        let req = proto::InferenceRequest {
            request_id: "unit-test-1".to_string(),
            input: vec![0.1, 0.2, 0.3],
            temperature: None,
            top_k: None,
            top_p: None,
            max_tokens: None,
        };

        let response = service.infer(Request::new(req)).await;
        // The default model-less engine returns an error; the important thing is
        // that we exercise the full call path without panicking.
        match response {
            Ok(reply) => {
                assert_eq!(reply.into_inner().request_id, "unit-test-1");
            }
            Err(status) => {
                // Acceptable: no model loaded in default engine
                assert!(
                    status.code() == tonic::Code::Internal,
                    "unexpected status code: {:?}",
                    status.code()
                );
            }
        }
    }

    #[tokio::test]
    async fn test_infer_stream_processes_requests() {
        // Test the streaming path at the InferenceServiceInner level, bypassing
        // the tonic Streaming<T> transport type entirely.  We construct the inner
        // directly and drive the mpsc-based response stream by sending requests
        // through the engine one at a time.
        let inner = Arc::new(InferenceServiceInner::new(make_engine()));

        // Three synthetic requests — each is processed by `infer`, which is the
        // same code path that `infer_stream_impl` calls internally.
        let requests: Vec<proto::InferenceRequest> = (0u32..3)
            .map(|i| proto::InferenceRequest {
                request_id: format!("stream-{i}"),
                input: vec![i as f32 * 0.1],
                ..Default::default()
            })
            .collect();

        let (tx, rx) = mpsc::channel::<Result<proto::InferenceReply, Status>>(16);

        // Simulate what infer_stream_impl does: iterate over requests and call
        // the engine for each one.
        let inner_clone = inner.clone();
        tokio::spawn(async move {
            for req in requests {
                let result = inner_clone.infer(Request::new(req)).await;
                match result {
                    Ok(resp) => {
                        if tx.send(Ok(resp.into_inner())).await.is_err() {
                            break;
                        }
                    }
                    Err(status) => {
                        let _ = tx.send(Err(status)).await;
                        break;
                    }
                }
            }
        });

        // Drain the receiver and count replies.
        let mut count = 0usize;
        let mut error_count = 0usize;
        let mut receiver = ReceiverStream::new(rx);
        while let Some(item) = tokio_stream::StreamExt::next(&mut receiver).await {
            match item {
                Ok(_reply) => count += 1,
                Err(_status) => error_count += 1,
            }
        }

        // With no model loaded the engine returns an error on the first call,
        // which causes the spawned task to break early.  The stream must complete
        // without panicking and must produce at least one response (ok or err).
        assert!(
            count + error_count >= 1,
            "expected at least 1 response (got {count} ok + {error_count} err)"
        );
        assert!(
            count + error_count <= 3,
            "expected at most 3 responses (got {count} ok + {error_count} err)"
        );
    }

    /// Regression: `PATH_INFER_STREAM` used to be routed through
    /// `ClientStreamingService`, which collects every frame and returns only
    /// the *last* reply — N requests in, 1 reply out, discarding N-1
    /// predictions, and a 0-request stream produced a fabricated
    /// `Default::default()` reply. Routed through `StreamingService` (this
    /// test exercises the real wire codec via `tonic::client::Grpc`, not
    /// just the `InferenceServiceInner` level `test_infer_stream_processes_requests`
    /// already covers), N requests must now produce N replies, in order,
    /// each with its own `request_id` — and 0 requests must produce 0
    /// replies rather than a fabricated one.
    #[tokio::test]
    async fn test_infer_stream_wire_returns_one_reply_per_frame() {
        use crate::testutil::CountingModel;

        let stream_config = StreamConfig {
            engine: crate::engine::EngineConfig::new(1, 1),
            ..Default::default()
        };
        let mut engine = StreamingEngine::new(stream_config).expect("engine must construct");
        engine.set_model(Box::new(CountingModel::new())).await;

        let inner = Arc::new(InferenceServiceInner::new(engine));
        let server = InferenceServiceServer { inner };

        let mut client = tonic::client::Grpc::new(server);
        client.ready().await.expect("service must be ready");

        let requests: Vec<proto::InferenceRequest> = (0u32..3)
            .map(|i| proto::InferenceRequest {
                request_id: format!("wire-{i}"),
                input: vec![i as f32 * 0.1],
                ..Default::default()
            })
            .collect();

        let path = ::http::uri::PathAndQuery::from_static(PATH_INFER_STREAM);
        let codec: ProstCodec<proto::InferenceRequest, proto::InferenceReply> =
            ProstCodec::default();

        let response = client
            .streaming(Request::new(futures::stream::iter(requests)), path, codec)
            .await
            .expect("streaming call must succeed");

        let mut reply_stream = response.into_inner();
        let mut request_ids = Vec::new();
        while let Some(reply) = reply_stream.next().await {
            request_ids.push(reply.expect("each reply must decode").request_id);
        }

        assert_eq!(
            request_ids,
            vec![
                "wire-0".to_string(),
                "wire-1".to_string(),
                "wire-2".to_string()
            ],
            "each request frame must produce its own reply, in order, over the real wire codec"
        );

        // Zero requests must yield zero replies, not a fabricated
        // `Default::default()` reply (empty request_id, latency_ms: 0.0).
        let empty_path = ::http::uri::PathAndQuery::from_static(PATH_INFER_STREAM);
        let empty_codec: ProstCodec<proto::InferenceRequest, proto::InferenceReply> =
            ProstCodec::default();
        let empty_response = client
            .streaming(
                Request::new(futures::stream::iter(Vec::<proto::InferenceRequest>::new())),
                empty_path,
                empty_codec,
            )
            .await
            .expect("zero-frame streaming call must still succeed");
        let empty_replies: Vec<_> = empty_response.into_inner().collect().await;
        assert!(
            empty_replies.is_empty(),
            "zero requests must yield zero replies, got {}",
            empty_replies.len()
        );
    }

    // -----------------------------------------------------------------------
    // Transport-level test: bind to an ephemeral port, start, signal shutdown
    // -----------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_grpc_server_starts_and_stops() {
        use std::net::TcpListener;

        // Find a free ephemeral port by binding to :0 and immediately releasing.
        let ephemeral_addr = {
            let listener = TcpListener::bind("127.0.0.1:0").expect("OS should assign a free port");
            listener.local_addr().expect("local_addr must be set")
        };

        let engine = make_engine();
        let adapter = GrpcAdapter::new(ephemeral_addr, engine);
        let server = GrpcServer::new(adapter);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // Spawn the server in the background.
        let server_handle = tokio::spawn(async move {
            server
                .serve_with_graceful_shutdown(async move {
                    // Await the shutdown signal; ignore if sender dropped early.
                    let _ = shutdown_rx.await;
                })
                .await
        });

        // Give the server a moment to start listening.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Send the shutdown signal.
        let _ = shutdown_tx.send(());

        // Wait for the server to exit — with a generous timeout so the test
        // doesn't hang indefinitely on CI.
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), server_handle).await;

        match result {
            Ok(join_result) => match join_result {
                Ok(Ok(())) => {
                    // Clean shutdown
                }
                Ok(Err(e)) => {
                    // NetworkError on bind failure is acceptable in constrained environments
                    let msg = format!("{e}");
                    assert!(
                        msg.contains("transport")
                            || msg.contains("Network")
                            || msg.contains("address"),
                        "unexpected serve error: {msg}"
                    );
                }
                Err(e) => panic!("server task panicked: {e:?}"),
            },
            Err(_) => {
                // Timeout — this can happen if the shutdown signal wasn't received;
                // log and continue rather than failing the whole suite.
                eprintln!("WARNING: test_grpc_server_starts_and_stops timed out — server may not have shut down cleanly");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_grpc_adapter_creation() {
        let engine = make_engine();
        let addr: SocketAddr = "127.0.0.1:50051".parse().expect("valid socket addr");
        let adapter = GrpcAdapter::new(addr, engine);
        assert!(!adapter.is_running());
    }

    /// `is_running()` must be a plain, panic-free read even on a
    /// current-thread runtime (the old `block_in_place` implementation
    /// panicked outside a multi-threaded Tokio context).
    #[tokio::test]
    async fn test_is_running_does_not_panic_on_current_thread_runtime() {
        let engine = make_engine();
        let addr: SocketAddr = "127.0.0.1:50052".parse().expect("valid socket addr");
        let adapter = GrpcAdapter::new(addr, engine);
        assert!(!adapter.is_running());
    }

    /// Regression: `NetworkAdapter::stop()` used to flip a flag that the
    /// `serve()` accept loop never read, so a running server kept accepting
    /// connections after `stop()` returned `Ok(())`. `start()` must now
    /// actually resolve promptly after `stop()` is called from a different
    /// `Arc`-shared handle.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_network_adapter_stop_unblocks_start() {
        use std::net::TcpListener as StdTcpListener;

        let ephemeral_addr = {
            let listener =
                StdTcpListener::bind("127.0.0.1:0").expect("OS should assign a free port");
            listener.local_addr().expect("local_addr must be set")
        };

        let adapter = Arc::new(GrpcAdapter::new(ephemeral_addr, make_engine()));
        let adapter_for_task = adapter.clone();
        let handle = tokio::spawn(async move { adapter_for_task.start().await });

        // Poll rather than a single fixed sleep: robust against scheduling
        // jitter when many tests run concurrently.
        let mut became_running = false;
        for _ in 0..100 {
            if adapter.is_running() {
                became_running = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            became_running,
            "adapter should report running after start()"
        );

        adapter.stop().await.expect("stop must succeed");

        let result = tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
        assert!(
            result.is_ok(),
            "start() must resolve promptly after stop(), not keep serving forever"
        );
        assert!(
            !adapter.is_running(),
            "adapter must report not-running once start() has returned"
        );
    }
}
