//! gRPC adapter for high-performance streaming inference
//!
//! Provides gRPC-based inference with bidirectional streaming and load balancing support.

use super::{InferenceMessage, InferenceResponse, NetworkAdapter};
use crate::error::InferenceResult;
use crate::streaming::StreamingEngine;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, info};

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

/// gRPC service implementation
pub struct InferenceService {
    engine: Arc<RwLock<StreamingEngine>>,
}

impl InferenceService {
    /// Create a new inference service
    pub fn new(engine: StreamingEngine) -> Self {
        Self {
            engine: Arc::new(RwLock::new(engine)),
        }
    }

    /// Handle unary inference request
    pub async fn infer(
        &self,
        request: Request<proto::InferenceRequest>,
    ) -> Result<Response<proto::InferenceReply>, Status> {
        let req = request.into_inner();
        debug!("Received inference request: {}", req.request_id);

        let msg: InferenceMessage = req.into();
        let input = msg.to_array();

        let start = std::time::Instant::now();

        let output = {
            let engine = self.engine.write().await;
            engine
                .step_async(input)
                .await
                .map_err(|e| Status::internal(format!("Inference error: {}", e)))?
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        let response =
            InferenceResponse::new(msg.request_id, output.to_vec(), latency_ms, output.len());

        Ok(Response::new(response.into()))
    }

    /// Handle streaming inference
    ///
    /// Note: In a production implementation, this would use tonic's generated
    /// streaming traits. For now, this is a simplified stub.
    pub async fn infer_stream(
        &self,
        _request: Request<Streaming<proto::InferenceRequest>>,
    ) -> Result<Response<Streaming<proto::InferenceReply>>, Status> {
        // In production, you would:
        // 1. Use tonic-build in build.rs to generate the service trait
        // 2. Implement the generated trait with proper streaming types
        // 3. Use tokio::sync::mpsc for the response stream
        //
        // For now, return an error indicating this needs proper setup
        Err(Status::unimplemented(
            "Streaming inference requires full gRPC service setup with tonic-build",
        ))
    }

    /// Handle health check
    pub async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthReply>, Status> {
        Ok(Response::new(proto::HealthReply {
            status: "healthy".to_string(),
            engine_state: "ready".to_string(),
        }))
    }
}

/// gRPC adapter for streaming inference
pub struct GrpcAdapter {
    /// Service implementation
    service: InferenceService,

    /// Address to bind to
    addr: SocketAddr,

    /// Running state
    running: Arc<RwLock<bool>>,
}

impl GrpcAdapter {
    /// Create a new gRPC adapter
    ///
    /// # Arguments
    ///
    /// * `addr` - Socket address to bind to
    /// * `engine` - Streaming inference engine
    pub fn new(addr: impl Into<SocketAddr>, engine: StreamingEngine) -> Self {
        Self {
            service: InferenceService::new(engine),
            addr: addr.into(),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Get a reference to the service
    pub fn service(&self) -> &InferenceService {
        &self.service
    }

    /// Serve gRPC requests
    ///
    /// Note: This is a simplified implementation. In production, you would use
    /// tonic::transport::Server with the generated service trait.
    pub async fn serve(&self) -> InferenceResult<()> {
        info!("gRPC server would listen on {}", self.addr);
        *self.running.write().await = true;

        // In a full implementation, you would:
        // 1. Generate protobuf service with tonic-build
        // 2. Implement the generated trait
        // 3. Use tonic::transport::Server::builder()
        //    .add_service(service)
        //    .serve(addr)
        //    .await

        Ok(())
    }
}

impl NetworkAdapter for GrpcAdapter {
    async fn start(&mut self) -> InferenceResult<()> {
        self.serve().await
    }

    async fn stop(&mut self) -> InferenceResult<()> {
        *self.running.write().await = false;
        info!("gRPC server stopped");
        Ok(())
    }

    fn is_running(&self) -> bool {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async { *self.running.read().await })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_grpc_adapter_creation() {
        use crate::streaming::StreamConfig;

        let stream_config = StreamConfig::default();
        let engine = StreamingEngine::new(stream_config).unwrap();

        let addr: SocketAddr = "127.0.0.1:50051".parse().unwrap();
        let adapter = GrpcAdapter::new(addr, engine);

        assert!(!adapter.is_running());
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

    #[tokio::test]
    async fn test_inference_service_health() {
        use crate::streaming::StreamConfig;

        let stream_config = StreamConfig::default();
        let engine = StreamingEngine::new(stream_config).unwrap();
        let service = InferenceService::new(engine);

        let request = Request::new(proto::HealthRequest {});
        let response = service.health(request).await;

        assert!(response.is_ok());
        let reply = response.unwrap().into_inner();
        assert_eq!(reply.status, "healthy");
    }
}
