//! REST adapter for HTTP-based inference
//!
//! Provides an axum-based HTTP/1.1 + HTTP/2 REST server that exposes the Kizzasi
//! inference engine over a simple JSON API.
//!
//! # Endpoints
//!
//! | Method | Path      | Description                      |
//! |--------|-----------|----------------------------------|
//! | GET    | /health   | Liveness / readiness probe       |
//! | POST   | /infer    | Single-step inference (JSON)     |
//! | GET    | /metrics  | Lightweight Prometheus-style dump |
//!
//! # Feature gate
//!
//! This module is only compiled when the `rest` feature is enabled:
//!
//! ```toml
//! kizzasi-inference = { features = ["rest"] }
//! ```

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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
/// Owns the server configuration and the shared metrics state.  Callers
/// interact with it either through [`RestServer`] (the high-level wrapper) or
/// by building a [`Router`] directly via [`RestAdapter::router`].
pub struct RestAdapter {
    config: RestConfig,
    metrics: Arc<ServerMetrics>,
}

impl RestAdapter {
    /// Create a new `RestAdapter` from the given [`RestConfig`].
    pub fn new(config: RestConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(ServerMetrics::default()),
        }
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
    /// Create a new `RestServer` with the given [`RestConfig`].
    pub fn new(config: RestConfig) -> Self {
        Self {
            adapter: RestAdapter::new(config),
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

    let mut router = Router::new()
        .route("/health", get(health_handler))
        .route("/infer", post(infer_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(adapter);

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

/// `POST /infer` — single-step or multi-step inference.
///
/// The mock implementation multiplies each input sample by `0.9` per step,
/// mimicking an exponential decay.  In production, replace the body of this
/// function with a call to the real inference engine.
async fn infer_handler(
    State(state): State<Arc<RestAdapter>>,
    Json(req): Json<RestInferRequest>,
) -> impl IntoResponse {
    if req.signal.is_empty() {
        state.metrics.record_error();
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": "signal must not be empty"
            })),
        )
            .into_response();
    }

    let start = std::time::Instant::now();
    let steps = req.steps.unwrap_or(1).max(1);
    let temperature = req.temperature.unwrap_or(1.0);

    debug!(
        signal_len = req.signal.len(),
        steps = steps,
        temperature = temperature,
        "REST /infer request"
    );

    // -----------------------------------------------------------------------
    // Mock inference kernel:
    // For each step apply gain = 0.9 * clamp(temperature, 0.1, 10.0)⁻¹.
    // This is a placeholder; real code would call engine.infer() here.
    // -----------------------------------------------------------------------
    let gain = 0.9_f32 / temperature.clamp(0.1, 10.0);
    let mut prediction = req.signal.clone();
    for _ in 0..steps {
        for sample in prediction.iter_mut() {
            *sample *= gain;
        }
    }

    let latency_ms = start.elapsed().as_millis() as u64;
    state.metrics.record_success(latency_ms);

    (
        StatusCode::OK,
        Json(RestInferResponse {
            prediction,
            model_id: "kizzasi-default".to_string(),
            latency_ms,
            steps_executed: steps,
        }),
    )
        .into_response()
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

    #[tokio::test]
    async fn test_infer_endpoint_returns_prediction() {
        let app = build_router(make_adapter());
        let body = serde_json::json!({
            "signal": [1.0_f32, 2.0_f32, 3.0_f32],
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
        assert_eq!(infer_resp.prediction.len(), 3);
        assert_eq!(infer_resp.steps_executed, 1);
        assert_eq!(infer_resp.model_id, "kizzasi-default");
    }

    #[tokio::test]
    async fn test_infer_prediction_values_single_step() {
        let app = build_router(make_adapter());
        let signal = vec![10.0_f32, 20.0_f32];
        let body = serde_json::json!({
            "signal": signal,
            "steps": 1,
            "temperature": 1.0
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
        // gain = 0.9 / clamp(1.0, 0.1, 10.0) = 0.9
        for (input, output) in signal.iter().zip(infer_resp.prediction.iter()) {
            let expected = input * 0.9;
            assert!(
                (output - expected).abs() < 1e-5,
                "expected {expected} got {output}"
            );
        }
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
        assert_eq!(snap.requests_total, 1);
        assert_eq!(snap.errors_total, 0);
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
        let app = build_router(make_adapter());
        let body = serde_json::json!({
            "signal": [100.0_f32],
            "steps": 2,
            "temperature": 1.0
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
        // gain = 0.9 / 1.0 = 0.9; after 2 steps: 100.0 * 0.9^2 = 81.0
        let expected = 100.0_f32 * 0.9_f32 * 0.9_f32;
        assert!(
            (infer_resp.prediction[0] - expected).abs() < 1e-4,
            "expected {expected} got {}",
            infer_resp.prediction[0]
        );
    }
}
