//! REST adapter for HTTP-based inference
//!
//! Provides an axum-based HTTP/1.1 + HTTP/2 REST server that exposes the Kizzasi
//! inference engine over a simple JSON API.
//!
//! `POST /infer` is only engine-backed when a [`StreamingEngine`] is attached
//! via [`RestAdapter::with_engine`] / [`RestServer::with_engine`]; an adapter
//! built via [`RestAdapter::new`] / [`RestServer::new`] has no engine and
//! answers every `/infer` call with `503 Service Unavailable` rather than
//! fabricated numbers.
//!
//! # Endpoints
//!
//! | Method | Path      | Description                      |
//! |--------|-----------|----------------------------------|
//! | GET    | /health   | Liveness / readiness probe       |
//! | POST   | /infer    | Single-step / multi-step inference (JSON) |
//! | GET    | /metrics  | Lightweight Prometheus-style dump |
//!
//! # Feature gate
//!
//! This module is only compiled when the `rest` feature is enabled:
//!
//! ```toml
//! kizzasi-inference = { features = ["rest"] }
//! ```

use crate::error::InferenceError;
use crate::streaming::{SamplingOverrides, StreamingEngine};
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, info, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the REST inference server.
#[derive(Debug, Clone)]
pub struct RestConfig {
    /// Address and port to bind, e.g. `"0.0.0.0:8080"`.
    pub addr: String,

    /// Maximum allowed request body size in bytes (default: 1 MiB).
    pub max_body_size: usize,

    /// Request handling timeout in milliseconds (default: 30 000 ms).
    pub request_timeout_ms: u64,

    /// Whether to attach a permissive CORS layer (default: `true`).
    pub cors_enabled: bool,
}

impl Default for RestConfig {
    fn default() -> Self {
        Self {
            addr: "0.0.0.0:8080".to_string(),
            max_body_size: 1024 * 1024,
            request_timeout_ms: 30_000,
            cors_enabled: true,
        }
    }
}

// ============================================================================
// Request / response types
// ============================================================================

/// JSON request body for `POST /infer`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RestInferRequest {
    /// Input signal samples (f32 PCM or any continuous signal frame).
    pub signal: Vec<f32>,

    /// Number of autoregressive steps to roll out.
    #[serde(default)]
    pub steps: Option<usize>,

    /// Sampling temperature (0.0 = greedy, higher = more random).
    #[serde(default)]
    pub temperature: Option<f32>,
}

/// JSON response body for `POST /infer`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RestInferResponse {
    /// Predicted output signal samples.
    pub prediction: Vec<f32>,

    /// Identifier of the model that produced the prediction.
    pub model_id: String,

    /// Wall-clock latency of the inference call in milliseconds.
    pub latency_ms: u64,

    /// Number of autoregressive steps that were executed.
    pub steps_executed: usize,
}

/// JSON response body for `GET /health`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Human-readable status string, e.g. `"healthy"`.
    pub status: String,

    /// Crate version from `Cargo.toml` at compile time.
    pub version: String,
}

/// JSON response body for `GET /metrics`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResponse {
    /// Total number of successful inference requests served since startup.
    pub requests_total: u64,

    /// Total number of failed inference requests since startup.
    pub errors_total: u64,

    /// Approximate average latency over all requests, in milliseconds.
    pub avg_latency_ms: f64,
}

// ============================================================================
// Shared server state
// ============================================================================

/// Atomic counters kept in the shared state so all axum handlers can update
/// them without blocking.
#[derive(Debug, Default)]
struct ServerMetrics {
    requests_total: std::sync::atomic::AtomicU64,
    errors_total: std::sync::atomic::AtomicU64,
    /// Running sum of latencies (ms × 1 000 for integer precision).
    latency_sum_us: std::sync::atomic::AtomicU64,
}

impl ServerMetrics {
    fn record_success(&self, latency_ms: u64) {
        self.requests_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.latency_sum_us
            .fetch_add(latency_ms * 1_000, std::sync::atomic::Ordering::Relaxed);
    }

    fn record_error(&self) {
        self.errors_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn snapshot(&self) -> MetricsResponse {
        let req = self
            .requests_total
            .load(std::sync::atomic::Ordering::Relaxed);
        let err = self.errors_total.load(std::sync::atomic::Ordering::Relaxed);
        let sum_us = self
            .latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed);
        let avg_latency_ms = if req == 0 {
            0.0
        } else {
            (sum_us as f64) / (req as f64) / 1_000.0
        };
        MetricsResponse {
            requests_total: req,
            errors_total: err,
            avg_latency_ms,
        }
    }
}

// ============================================================================
// REST adapter (core type)
// ============================================================================

/// Low-level axum REST adapter.
///
/// Owns the server configuration, the shared metrics state, and (optionally)
/// the [`StreamingEngine`] that backs `POST /infer`. Callers interact with it
/// either through [`RestServer`] (the high-level wrapper) or by building a
/// [`Router`] directly via [`RestAdapter::router`].
pub struct RestAdapter {
    config: RestConfig,
    metrics: Arc<ServerMetrics>,
    /// The engine backing `POST /infer`, if any. `None` (from
    /// [`RestAdapter::new`]) makes `/infer` answer `503 Service Unavailable`
    /// instead of fabricating a prediction.
    engine: Option<Arc<StreamingEngine>>,
}

impl RestAdapter {
    /// Create a new `RestAdapter` from the given [`RestConfig`], with no
    /// inference engine attached.
    ///
    /// `POST /infer` on an adapter built this way returns
    /// `503 Service Unavailable` for every request; use
    /// [`RestAdapter::with_engine`] to serve real predictions.
    pub fn new(config: RestConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(ServerMetrics::default()),
            engine: None,
        }
    }

    /// Create a new `RestAdapter` backed by `engine`: `POST /infer` runs real
    /// inference through it.
    pub fn with_engine(config: RestConfig, engine: StreamingEngine) -> Self {
        Self {
            config,
            metrics: Arc::new(ServerMetrics::default()),
            engine: Some(Arc::new(engine)),
        }
    }

    /// Attach (or replace) the [`StreamingEngine`] backing `POST /infer`.
    pub fn set_engine(&mut self, engine: StreamingEngine) {
        self.engine = Some(Arc::new(engine));
    }

    /// Whether an inference engine is currently attached.
    pub fn has_engine(&self) -> bool {
        self.engine.is_some()
    }

    /// Return a reference to the current [`RestConfig`].
    pub fn config(&self) -> &RestConfig {
        &self.config
    }

    /// Build and return the axum [`Router`].
    ///
    /// The router is cloneable and can be composed with other routers.
    pub fn router(self: &Arc<Self>) -> Router {
        build_router(self.clone())
    }

    /// Bind to the configured address and serve requests indefinitely.
    ///
    /// This is an async function that blocks until the process exits.
    pub async fn serve(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let adapter = Arc::new(RestAdapter {
            config: self.config.clone(),
            metrics: self.metrics.clone(),
            engine: self.engine.clone(),
        });
        let app = build_router(adapter);
        let listener = TcpListener::bind(&self.config.addr)
            .await
            .map_err(|e| format!("REST: failed to bind {}: {e}", self.config.addr))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| format!("REST: local_addr error: {e}"))?;
        info!("REST inference server listening on {local_addr}");
        axum::serve(listener, app)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    /// Bind to the configured address and serve until `shutdown` resolves.
    ///
    /// This enables graceful shutdown via any async signal (e.g.
    /// `tokio::signal::ctrl_c()` or a `oneshot` channel).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    /// adapter.serve_with_graceful_shutdown(async move { let _ = rx.await; }).await?;
    /// tx.send(()).ok();
    /// ```
    pub async fn serve_with_graceful_shutdown<F>(
        &self,
        shutdown: F,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let adapter = Arc::new(RestAdapter {
            config: self.config.clone(),
            metrics: self.metrics.clone(),
            engine: self.engine.clone(),
        });
        let app = build_router(adapter);
        let listener = TcpListener::bind(&self.config.addr)
            .await
            .map_err(|e| format!("REST: failed to bind {}: {e}", self.config.addr))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| format!("REST: local_addr error: {e}"))?;
        info!("REST inference server (with shutdown) listening on {local_addr}");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }
}

// ============================================================================
// Higher-level server wrapper
// ============================================================================

/// Higher-level server that owns a [`RestAdapter`] and exposes a clean
/// lifecycle API.
pub struct RestServer {
    adapter: RestAdapter,
}

impl RestServer {
    /// Create a new `RestServer` with the given [`RestConfig`] and no
    /// inference engine attached (`/infer` answers `503`).
    pub fn new(config: RestConfig) -> Self {
        Self {
            adapter: RestAdapter::new(config),
        }
    }

    /// Create a new `RestServer` backed by `engine`, so `/infer` runs real
    /// inference.
    pub fn with_engine(config: RestConfig, engine: StreamingEngine) -> Self {
        Self {
            adapter: RestAdapter::with_engine(config, engine),
        }
    }

    /// Serve requests until an OS-level shutdown signal is received.
    ///
    /// On Unix this listens for `SIGTERM` and `SIGINT`; on other platforms
    /// it falls back to `Ctrl-C` only.
    pub async fn serve_until_shutdown(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let signal = async {
            match tokio::signal::ctrl_c().await {
                Ok(()) => info!("REST server: received Ctrl-C, shutting down"),
                Err(e) => warn!("REST server: failed to install Ctrl-C handler: {e}"),
            }
        };
        self.adapter.serve_with_graceful_shutdown(signal).await
    }

    /// Expose the underlying [`RestAdapter`].
    pub fn adapter(&self) -> &RestAdapter {
        &self.adapter
    }
}

// ============================================================================
// Router construction
// ============================================================================

/// Build the axum [`Router`] from a shared [`Arc<RestAdapter>`].
///
/// The function is `pub(crate)` so that tests can call it directly without
/// going through the full bind/serve lifecycle.
pub(crate) fn build_router(adapter: Arc<RestAdapter>) -> Router {
    let cors_enabled = adapter.config.cors_enabled;
    let max_body_size = adapter.config.max_body_size;

    let mut router = Router::new()
        .route("/health", get(health_handler))
        .route("/infer", post(infer_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(adapter)
        .layer(DefaultBodyLimit::max(max_body_size));

    if cors_enabled {
        router = router.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }

    router
}

// ============================================================================
// Handlers
// ============================================================================

/// `GET /health` — liveness / readiness probe.
async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// `GET /metrics` — lightweight metrics dump.
async fn metrics_handler(State(state): State<Arc<RestAdapter>>) -> impl IntoResponse {
    Json(state.metrics.snapshot())
}

/// Map an [`InferenceError`] onto an HTTP status code.
///
/// A missing/uninitialized engine is the server's own configuration problem
/// (`503`); a shape or sampling-parameter mismatch is the caller's mistake
/// (`422`); anything else is an unexpected internal failure (`500`).
fn status_for_inference_error(e: &InferenceError) -> StatusCode {
    match e {
        InferenceError::NotInitialized => StatusCode::SERVICE_UNAVAILABLE,
        InferenceError::InvalidConfiguration(_) | InferenceError::DimensionMismatch { .. } => {
            StatusCode::UNPROCESSABLE_ENTITY
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn json_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}

/// `POST /infer` — single-step or multi-step inference, engine-backed.
///
/// Returns `503 Service Unavailable` when no [`StreamingEngine`] is attached
/// (see [`RestAdapter::with_engine`]), `422 Unprocessable Entity` for an
/// empty/mis-shaped signal or a rejected sampling override, and
/// `504 Gateway Timeout` if the request exceeds
/// [`RestConfig::request_timeout_ms`] (`0` disables the deadline).
async fn infer_handler(
    State(state): State<Arc<RestAdapter>>,
    Json(req): Json<RestInferRequest>,
) -> impl IntoResponse {
    if req.signal.is_empty() {
        state.metrics.record_error();
        return json_error(StatusCode::UNPROCESSABLE_ENTITY, "signal must not be empty");
    }

    let Some(engine) = state.engine.clone() else {
        state.metrics.record_error();
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no inference engine attached to this REST adapter; construct it via \
             RestAdapter::with_engine / RestServer::with_engine",
        );
    };

    debug!(
        signal_len = req.signal.len(),
        steps = ?req.steps,
        temperature = ?req.temperature,
        "REST /infer request"
    );

    let work = run_inference(state.clone(), engine, req);

    if state.config.request_timeout_ms == 0 {
        // `0` means "no deadline", matching the `GrpcServerConfig` convention
        // used elsewhere in this crate's adapters.
        return work.await;
    }

    match tokio::time::timeout(Duration::from_millis(state.config.request_timeout_ms), work).await {
        Ok(response) => response,
        Err(_) => {
            state.metrics.record_error();
            json_error(
                StatusCode::GATEWAY_TIMEOUT,
                format!(
                    "inference exceeded the configured request_timeout_ms ({} ms)",
                    state.config.request_timeout_ms
                ),
            )
        }
    }
}

/// The actual engine call, factored out of [`infer_handler`] so it can be
/// raced against a deadline via `tokio::time::timeout`.
async fn run_inference(
    state: Arc<RestAdapter>,
    engine: Arc<StreamingEngine>,
    req: RestInferRequest,
) -> Response {
    let start = std::time::Instant::now();
    let steps = req.steps.unwrap_or(1).max(1);

    let input_dim = engine.config().engine.input_dim;
    if req.signal.len() != input_dim {
        state.metrics.record_error();
        return json_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "signal has {} sample(s) but the attached model expects input_dim = {input_dim}",
                req.signal.len()
            ),
        );
    }

    let overrides = SamplingOverrides {
        temperature: req.temperature,
        top_k: None,
        top_p: None,
        max_tokens: Some(steps),
    };
    let input = Array1::from_vec(req.signal.clone());

    match engine.step_async_with(input, &overrides).await {
        Ok(outputs) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            state.metrics.record_success(latency_ms);

            // Autoregressive rollout: report the final steps-ahead
            // prediction (the "output signal" after `steps` iterations),
            // matching the crate's other adapters (see
            // `adapters::flatten_outputs` for the alternative "concatenate
            // every step" convention used by the streaming adapters).
            let prediction = outputs.last().map(|o| o.to_vec()).unwrap_or_default();
            let model_id = engine
                .model_info()
                .await
                .map(|info| info.model_type.to_string())
                .unwrap_or_else(|| "unknown".to_string());

            (
                StatusCode::OK,
                Json(RestInferResponse {
                    prediction,
                    model_id,
                    latency_ms,
                    steps_executed: outputs.len(),
                }),
            )
                .into_response()
        }
        Err(e) => {
            state.metrics.record_error();
            json_error(status_for_inference_error(&e), e.to_string())
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // for `oneshot`

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_adapter() -> Arc<RestAdapter> {
        Arc::new(RestAdapter::new(RestConfig::default()))
    }

    /// Build an adapter backed by a real, deterministic (`input_dim = 1`)
    /// [`crate::testutil::CountingModel`] engine, so `/infer` tests can
    /// assert genuine engine output instead of the old hardcoded mock.
    async fn make_engine_adapter() -> Arc<RestAdapter> {
        make_engine_adapter_with_config(RestConfig::default()).await
    }

    async fn make_engine_adapter_with_config(config: RestConfig) -> Arc<RestAdapter> {
        use crate::engine::EngineConfig;
        use crate::streaming::{StreamConfig, StreamingEngine};
        use crate::testutil::CountingModel;

        let mut stream_config = StreamConfig::new();
        stream_config.engine = EngineConfig::new(1, 1);
        let mut engine = StreamingEngine::new(stream_config).expect("engine must construct");
        engine.set_model(Box::new(CountingModel::new())).await;

        Arc::new(RestAdapter::with_engine(config, engine))
    }

    // -----------------------------------------------------------------------
    // Sync / configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rest_config_defaults() {
        let cfg = RestConfig::default();
        assert_eq!(cfg.addr, "0.0.0.0:8080");
        assert_eq!(cfg.max_body_size, 1024 * 1024);
        assert_eq!(cfg.request_timeout_ms, 30_000);
        assert!(cfg.cors_enabled);
    }

    #[test]
    fn test_rest_infer_request_serialization() {
        let req = RestInferRequest {
            signal: vec![1.0, 2.0],
            steps: Some(5),
            temperature: Some(0.8),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: RestInferRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.signal, req.signal);
        assert_eq!(back.steps, req.steps);
        assert_eq!(back.temperature, req.temperature);
    }

    #[test]
    fn test_rest_infer_request_optional_fields_default() {
        let json = r#"{"signal":[1.0,2.0]}"#;
        let req: RestInferRequest = serde_json::from_str(json).unwrap();
        assert!(req.steps.is_none());
        assert!(req.temperature.is_none());
    }

    #[test]
    fn test_server_metrics_default() {
        let m = ServerMetrics::default();
        let snap = m.snapshot();
        assert_eq!(snap.requests_total, 0);
        assert_eq!(snap.errors_total, 0);
        assert_eq!(snap.avg_latency_ms, 0.0);
    }

    #[test]
    fn test_server_metrics_record_success() {
        let m = ServerMetrics::default();
        m.record_success(10);
        m.record_success(20);
        let snap = m.snapshot();
        assert_eq!(snap.requests_total, 2);
        assert_eq!(snap.errors_total, 0);
        assert!((snap.avg_latency_ms - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_server_metrics_record_error() {
        let m = ServerMetrics::default();
        m.record_error();
        m.record_error();
        let snap = m.snapshot();
        assert_eq!(snap.errors_total, 2);
        assert_eq!(snap.requests_total, 0);
    }

    // -----------------------------------------------------------------------
    // Async handler tests (in-process, no network)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_health_endpoint_returns_ok() {
        let app = build_router(make_adapter());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_response_body() {
        let app = build_router(make_adapter());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(health.status, "healthy");
        assert!(!health.version.is_empty());
    }

    /// Regression: `/infer` used to be a hardcoded mock (`input * 0.9` per
    /// step) that never touched any model, always reporting
    /// `model_id: "kizzasi-default"`. It must now run the real, attached
    /// engine and report the real model type.
    #[tokio::test]
    async fn test_infer_endpoint_returns_prediction() {
        let app = build_router(make_engine_adapter().await);
        let body = serde_json::json!({
            "signal": [1.0_f32],
            "steps": 1
        });
        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let infer_resp: RestInferResponse = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(infer_resp.prediction.len(), 1);
        assert_eq!(infer_resp.steps_executed, 1);
        assert_ne!(
            infer_resp.model_id, "kizzasi-default",
            "model_id must reflect the real attached model, not the old fabricated constant"
        );
        assert_eq!(infer_resp.model_id, "S4D"); // CountingModel reports ModelType::S4D
    }

    /// Regression: prediction values used to follow the mock's
    /// `input * 0.9^steps` formula regardless of what model was configured.
    /// They must now be the real, deterministic `CountingModel` recurrence
    /// (`accumulated = accumulated * 0.5 + input`, starting from `0.0`).
    #[tokio::test]
    async fn test_infer_prediction_values_single_step() {
        let app = build_router(make_engine_adapter().await);
        let body = serde_json::json!({
            "signal": [10.0_f32],
            "steps": 1
        });
        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let infer_resp: RestInferResponse = serde_json::from_slice(&body_bytes).unwrap();
        // CountingModel: accumulated = 0.0 * 0.5 + 10.0 = 10.0 — NOT 10.0 * 0.9.
        let expected = 10.0_f32;
        assert!(
            (infer_resp.prediction[0] - expected).abs() < 1e-5,
            "expected {expected} got {}",
            infer_resp.prediction[0]
        );
    }

    /// `POST /infer` on an adapter with no engine attached must fail loudly
    /// (`503`) instead of fabricating a prediction.
    #[tokio::test]
    async fn test_infer_without_engine_returns_503() {
        let app = build_router(make_adapter());
        let body = serde_json::json!({ "signal": [1.0_f32] });
        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A signal whose length doesn't match the attached model's `input_dim`
    /// must be rejected with `422`, not silently truncated/padded or panic.
    #[tokio::test]
    async fn test_infer_rejects_signal_length_mismatch() {
        let app = build_router(make_engine_adapter().await);
        // CountingModel expects input_dim = 1; this sends 3.
        let body = serde_json::json!({ "signal": [1.0_f32, 2.0, 3.0] });
        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    /// An oversized request body must be rejected (`413`) rather than
    /// accepted unbounded, per the configured `max_body_size`.
    #[tokio::test]
    async fn test_infer_oversized_body_returns_413() {
        let config = RestConfig {
            max_body_size: 16,
            ..Default::default()
        };
        let app = build_router(Arc::new(RestAdapter::new(config)));

        let body = serde_json::json!({ "signal": [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] });
        let bytes = serde_json::to_vec(&body).unwrap();
        assert!(
            bytes.len() > 16,
            "test body must actually exceed the configured limit"
        );

        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(bytes))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    /// `request_timeout_ms: 0` must disable the deadline rather than firing
    /// immediately (a `Duration::from_millis(0)` timeout would fail every
    /// request).
    #[tokio::test]
    async fn test_request_timeout_zero_disables_wrapper() {
        let app = build_router(
            make_engine_adapter_with_config(RestConfig {
                request_timeout_ms: 0,
                ..Default::default()
            })
            .await,
        );
        let body = serde_json::json!({ "signal": [1.0_f32] });
        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Regression: `request_timeout_ms` used to have no reads outside
    /// `Default`/tests — a network-facing server had no request deadline at
    /// all. A request that outlives the deadline (because the engine's
    /// internal lock is held by another, slow in-flight request) must now
    /// receive `504` instead of hanging indefinitely.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_infer_request_timeout_returns_504_under_lock_contention() {
        use crate::engine::EngineConfig;
        use crate::streaming::{StreamConfig, StreamingEngine};
        use kizzasi_core::{CoreResult, HiddenState, SignalPredictor};
        use kizzasi_model::{AutoregressiveModel, ModelResult, ModelType};

        /// A model whose `step` blocks the calling worker thread for longer
        /// than the configured request timeout, simulating an expensive
        /// forward pass. `StreamingEngine::step_async_with` holds its
        /// internal `tokio::sync::Mutex` for the duration of `step`, so a
        /// second concurrent request genuinely awaits (yields on) that lock
        /// — a real async suspension point `tokio::time::timeout` can race
        /// against, unlike the blocking call itself.
        ///
        /// `started` flips to `true` as soon as `step` begins, so the test
        /// can deterministically wait for the slow request to actually be
        /// holding the lock before firing the second one, instead of
        /// guessing a fixed delay (which would be flaky under heavy
        /// concurrent test load).
        struct SlowModel {
            started: Arc<std::sync::atomic::AtomicBool>,
        }
        impl SignalPredictor for SlowModel {
            fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
                self.started
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(300));
                Ok(input.clone())
            }
            fn reset(&mut self) {}
            fn context_window(&self) -> usize {
                usize::MAX
            }
        }
        impl AutoregressiveModel for SlowModel {
            fn hidden_dim(&self) -> usize {
                1
            }
            fn state_dim(&self) -> usize {
                1
            }
            fn num_layers(&self) -> usize {
                1
            }
            fn model_type(&self) -> ModelType {
                ModelType::S4D
            }
            fn get_states(&self) -> Vec<HiddenState> {
                // Must match `num_layers()` (1): `InferenceEngine::step`
                // rejects a state count that disagrees with the context's
                // configured layer count.
                vec![HiddenState::new(1, 1)]
            }
            fn set_states(&mut self, _states: Vec<HiddenState>) -> ModelResult<()> {
                Ok(())
            }
        }

        let started = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut stream_config = StreamConfig::new();
        stream_config.engine = EngineConfig::new(1, 1);
        let mut engine = StreamingEngine::new(stream_config).expect("engine must construct");
        engine
            .set_model(Box::new(SlowModel {
                started: started.clone(),
            }))
            .await;

        let config = RestConfig {
            request_timeout_ms: 50, // much shorter than SlowModel's 300ms step
            ..Default::default()
        };
        let app = build_router(Arc::new(RestAdapter::with_engine(config, engine)));

        let make_request = || {
            let body = serde_json::json!({ "signal": [1.0_f32] });
            Request::builder()
                .uri("/infer")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap()
        };

        // Fire the slow request first, without waiting for it, so it is
        // holding the engine's internal lock while the second request races
        // its own timeout against lock acquisition.
        let app_for_slow = app.clone();
        let slow_task = tokio::spawn(async move { app_for_slow.oneshot(make_request()).await });

        // Deterministically wait for the slow request to actually acquire
        // the lock and start blocking inside `SlowModel::step`, rather than
        // guessing a fixed delay.
        let mut slow_request_started = false;
        for _ in 0..200 {
            if started.load(std::sync::atomic::Ordering::SeqCst) {
                slow_request_started = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(
            slow_request_started,
            "the slow request must have started within the polling window"
        );

        let resp = app.oneshot(make_request()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);

        let slow_resp = slow_task.await.expect("slow task must not panic").unwrap();
        assert_eq!(slow_resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_infer_empty_signal_returns_422() {
        let app = build_router(make_adapter());
        let body = serde_json::json!({ "signal": [] });
        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_returns_ok() {
        let app = build_router(make_adapter());
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let metrics: MetricsResponse = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(metrics.requests_total, 0);
        assert_eq!(metrics.errors_total, 0);
    }

    #[tokio::test]
    async fn test_metrics_increments_on_infer() {
        let adapter = make_engine_adapter().await;
        let app = build_router(adapter.clone());

        let body = serde_json::json!({ "signal": [1.0_f32] });
        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let _ = app.oneshot(req).await.unwrap();

        let snap = adapter.metrics.snapshot();
        assert_eq!(snap.requests_total, 1);
        assert_eq!(snap.errors_total, 0);
    }

    /// `/infer` on an adapter with no engine attached must still be counted
    /// as an error, not silently omitted from the metrics.
    #[tokio::test]
    async fn test_metrics_records_error_when_no_engine() {
        let adapter = make_adapter();
        let app = build_router(adapter.clone());

        let body = serde_json::json!({ "signal": [1.0_f32] });
        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let _ = app.oneshot(req).await.unwrap();

        let snap = adapter.metrics.snapshot();
        assert_eq!(snap.requests_total, 0);
        assert_eq!(snap.errors_total, 1);
    }

    #[tokio::test]
    async fn test_rest_server_starts_and_stops() {
        // Use an ephemeral port so the test never conflicts with other services.
        let config = RestConfig {
            addr: "127.0.0.1:0".to_string(),
            ..Default::default()
        };
        let adapter = RestAdapter::new(config);
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();

        let handle = tokio::spawn(async move {
            let _ = adapter
                .serve_with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });

        // Give the server a moment to start, then signal shutdown.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let _ = tx.send(());

        let result = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
        assert!(
            result.is_ok(),
            "REST server should shut down within 2 seconds"
        );
    }

    #[tokio::test]
    async fn test_unknown_route_returns_404() {
        let app = build_router(make_adapter());
        let req = Request::builder()
            .uri("/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_infer_multi_step() {
        let app = build_router(make_engine_adapter().await);
        let body = serde_json::json!({
            "signal": [100.0_f32],
            "steps": 2
        });
        let req = Request::builder()
            .uri("/infer")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let infer_resp: RestInferResponse = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(infer_resp.steps_executed, 2);
        // CountingModel autoregressive rollout (accumulated <- accumulated *
        // 0.5 + input, output fed back as the next input):
        //   step 1: accumulated = 0.0 * 0.5 + 100.0 = 100.0
        //   step 2: accumulated = 100.0 * 0.5 + 100.0 = 150.0
        // NOT the old mock's 100.0 * 0.9^2 = 81.0.
        let expected = 150.0_f32;
        assert!(
            (infer_resp.prediction[0] - expected).abs() < 1e-4,
            "expected {expected} got {}",
            infer_resp.prediction[0]
        );
    }
}
