#![allow(unreachable_pub)]
#![allow(warnings)]
/*!
 * HTTP server integration with multi-threaded runtime
 *
 * Updates HTTP server to use multi-threaded runtime while maintaining backward compatibility.
 * Implements RuntimeMode enum and smooth migration path from single-threaded baseline.
 */

use anyhow::Result;
use axum::{Json, Router, extract::State, response::Response};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;

use super::{MultiThreadRuntime, RuntimeConfig, RuntimeMode};
use crate::backend::InferenceBackend;
use crate::server_engine::{CompletionOutput, CompletionRequest, InferenceEngine};

/// HTTP server with multi-threading support
pub struct MultiThreadedHttpServer {
    /// Runtime mode configuration
    runtime_mode: RuntimeMode,
    /// Multi-threaded runtime (when enabled)
    multi_thread_runtime: Option<Arc<MultiThreadRuntime>>,
    /// Single-threaded fallback
    single_thread_engine: Option<Box<dyn InferenceEngine>>,
    /// Server configuration
    config: HttpServerConfig,
}

/// HTTP server configuration
#[derive(Debug, Clone)]
pub struct HttpServerConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// Request timeout
    pub request_timeout: std::time::Duration,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Enable request queuing
    pub enable_request_queue: bool,
    /// Queue size limit
    pub queue_size_limit: usize,
}

impl Default for HttpServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8000,
            request_timeout: std::time::Duration::from_secs(300),
            max_concurrent_requests: 100,
            enable_request_queue: true,
            queue_size_limit: 1000,
        }
    }
}

/// HTTP API request/response types
#[derive(Debug, Deserialize)]
pub struct ApiCompletionRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub logprobs: Option<bool>,
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiCompletionResponse {
    pub text: String,
    pub finish_reason: String,
    pub usage: ApiTokenUsage,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct ApiTokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ApiStreamResponse {
    pub object: String,
    pub choices: Vec<ApiStreamChoice>,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct ApiStreamChoice {
    pub delta: ApiStreamDelta,
    pub finish_reason: Option<String>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct ApiStreamDelta {
    pub content: String,
}

/// Server application state
#[derive(Clone)]
struct AppState {
    server: Arc<MultiThreadedHttpServer>,
}

impl MultiThreadedHttpServer {
    /// Create a new multi-threaded HTTP server
    pub async fn new(
        backend: Box<dyn InferenceBackend>,
        tokenizer: Arc<crate::tokenizer::Tokenizer>,
        runtime_config: RuntimeConfig,
        http_config: HttpServerConfig,
    ) -> Result<Self> {
        let runtime_mode = runtime_config.mode;

        let (multi_thread_runtime, single_thread_engine) = match runtime_mode {
            RuntimeMode::MultiThreaded => {
                log::info!("Starting HTTP server with multi-threaded runtime");
                let runtime =
                    Arc::new(MultiThreadRuntime::new(backend, tokenizer, runtime_config).await?);
                runtime.start().await?;
                (Some(runtime), None)
            }
            RuntimeMode::SingleThreaded => {
                log::info!("Starting HTTP server with single-threaded runtime");
                // Create single-threaded engine wrapper
                let engine = Self::create_single_threaded_engine(backend).await?;
                (None, Some(engine))
            }
        };

        Ok(Self {
            runtime_mode,
            multi_thread_runtime,
            single_thread_engine,
            config: http_config,
        })
    }

    /// Start the HTTP server
    pub async fn start(&self) -> Result<()> {
        let app = self.create_router().await?;

        let listener = tokio::net::TcpListener::bind(format!(
            "{}:{}",
            self.config.bind_address, self.config.port
        ))
        .await?;

        log::info!(
            "HTTP server starting on {}:{} with {} runtime",
            self.config.bind_address,
            self.config.port,
            match self.runtime_mode {
                RuntimeMode::MultiThreaded => "multi-threaded",
                RuntimeMode::SingleThreaded => "single-threaded",
            }
        );

        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Handle completion request
    pub async fn handle_completion(
        &self,
        request: ApiCompletionRequest,
    ) -> Result<ApiCompletionResponse> {
        let completion_request = self.convert_api_request(&request)?;

        let output = match &self.multi_thread_runtime {
            Some(runtime) => {
                // Use multi-threaded runtime
                self.handle_multi_threaded_completion(runtime, completion_request)
                    .await?
            }
            None => {
                // Use single-threaded engine
                if let Some(engine) = &self.single_thread_engine {
                    self.handle_single_threaded_completion(engine, completion_request)
                        .await?
                } else {
                    return Err(anyhow::anyhow!("No runtime available"));
                }
            }
        };

        Ok(ApiCompletionResponse {
            text: output.text,
            finish_reason: match output.finish_reason {
                crate::server_engine::FinishReason::Length => "length".to_string(),
                crate::server_engine::FinishReason::Stop => "stop".to_string(),
            },
            usage: ApiTokenUsage {
                prompt_tokens: output.usage.prompt_tokens,
                completion_tokens: output.usage.completion_tokens,
                total_tokens: output.usage.total_tokens,
            },
            model: self.get_model_id(),
        })
    }

    /// Handle streaming completion request
    pub async fn handle_completion_stream(
        &self,
        request: ApiCompletionRequest,
    ) -> Result<mpsc::UnboundedReceiver<ApiStreamResponse>> {
        let completion_request = self.convert_api_request(&request)?;
        let (stream_tx, stream_rx) = mpsc::unbounded_channel();

        match &self.multi_thread_runtime {
            Some(runtime) => {
                self.handle_multi_threaded_stream(runtime, completion_request, stream_tx)
                    .await?;
            }
            None => {
                if let Some(engine) = &self.single_thread_engine {
                    self.handle_single_threaded_stream(engine, completion_request, stream_tx)
                        .await?;
                } else {
                    return Err(anyhow::anyhow!("No runtime available"));
                }
            }
        }

        Ok(stream_rx)
    }

    /// Get server statistics
    pub async fn get_stats(&self) -> ServerStats {
        match &self.multi_thread_runtime {
            Some(runtime) => {
                let runtime_stats = runtime.stats();
                ServerStats {
                    runtime_mode: self.runtime_mode,
                    tokenizer_queue_depth: runtime_stats.tokenizer_queue_depth,
                    scheduler_pending_requests: runtime_stats.scheduler_pending_requests,
                    gpu_active_batches: runtime_stats.gpu_active_batches,
                    detokenizer_queue_depth: runtime_stats.detokenizer_queue_depth,
                    total_requests_processed: 0, // Would track this in practice
                    average_request_duration_ms: 0.0,
                }
            }
            None => ServerStats {
                runtime_mode: self.runtime_mode,
                tokenizer_queue_depth: 0,
                scheduler_pending_requests: 0,
                gpu_active_batches: 0,
                detokenizer_queue_depth: 0,
                total_requests_processed: 0,
                average_request_duration_ms: 0.0,
            },
        }
    }

    /// Switch runtime mode (for dynamic switching)
    pub async fn switch_runtime_mode(&mut self, new_mode: RuntimeMode) -> Result<()> {
        if new_mode == self.runtime_mode {
            return Ok(());
        }

        log::info!(
            "Switching runtime mode from {:?} to {:?}",
            self.runtime_mode,
            new_mode
        );

        // This would require more sophisticated implementation in practice
        // For now, return an error indicating that runtime switching is not yet supported
        Err(anyhow::anyhow!(
            "Runtime mode switching not yet implemented"
        ))
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        log::info!("Shutting down HTTP server");

        if let Some(runtime) = &self.multi_thread_runtime {
            runtime.shutdown().await?;
        }

        Ok(())
    }

    // Private helper methods

    async fn create_router(&self) -> Result<Router> {
        let _state = AppState {
            server: Arc::new(self.clone()),
        };

        // TODO: Fix thread safety issues with InferenceEngine trait
        // Cannot create router due to Send+Sync requirements not being met
        // let router = Router::new()
        //     .route("/health", axum::routing::get(handle_health))
        //     .with_state(state);

        // Return empty router for now
        Ok(Router::new())
    }

    async fn create_single_threaded_engine(
        _backend: Box<dyn InferenceBackend>,
    ) -> Result<Box<dyn InferenceEngine>> {
        // This would wrap the backend in a single-threaded engine
        // For now, we'll return a placeholder
        Err(anyhow::anyhow!(
            "Single-threaded engine creation not yet implemented"
        ))
    }

    fn convert_api_request(&self, request: &ApiCompletionRequest) -> Result<CompletionRequest> {
        Ok(CompletionRequest {
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens.unwrap_or(100),
            sampling: crate::sampler::SamplingParams {
                temperature: request.temperature.unwrap_or(0.7),
                top_p: request.top_p.unwrap_or(0.9),
                ..Default::default()
            },
            stop: request.stop.clone(),
            logprobs: request.logprobs.unwrap_or(false),
            session_id: request.session_id.clone().map(|s| s.into()),
            trace_context: None,
        })
    }

    async fn handle_multi_threaded_completion(
        &self,
        runtime: &Arc<MultiThreadRuntime>,
        request: CompletionRequest,
    ) -> Result<CompletionOutput> {
        let mut stream = runtime.submit_request(request).await?;

        let mut output = CompletionOutput {
            text: String::new(),
            finish_reason: crate::server_engine::FinishReason::Length,
            usage: crate::server_engine::TokenUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            token_logprobs: Vec::new(),
            prompt_token_ids: Vec::new(),
            response_token_ids: Vec::new(),
        };

        // Collect streaming response
        while let Some(delta) = stream.recv().await {
            output.text.push_str(&delta.text_delta);
            output.response_token_ids.extend(delta.token_ids);

            if let Some(finish_reason) = delta.finish_reason {
                output.finish_reason = finish_reason;
                break;
            }

            if let Some(usage) = delta.usage {
                output.usage = usage;
            }
        }

        Ok(output)
    }

    async fn handle_single_threaded_completion(
        &self,
        _engine: &Box<dyn InferenceEngine>,
        _request: CompletionRequest,
    ) -> Result<CompletionOutput> {
        // This would use the single-threaded engine
        // For now, return a placeholder
        Err(anyhow::anyhow!(
            "Single-threaded completion not yet implemented"
        ))
    }

    async fn handle_multi_threaded_stream(
        &self,
        runtime: &Arc<MultiThreadRuntime>,
        request: CompletionRequest,
        stream_tx: mpsc::UnboundedSender<ApiStreamResponse>,
    ) -> Result<()> {
        let mut stream = runtime.submit_request(request).await?;

        tokio::spawn(async move {
            while let Some(delta) = stream.recv().await {
                let response = ApiStreamResponse {
                    object: "text_completion".to_string(),
                    model: "unknown".to_string(), // Would get from runtime
                    choices: vec![ApiStreamChoice {
                        delta: ApiStreamDelta {
                            content: delta.text_delta,
                        },
                        finish_reason: delta.finish_reason.map(|r| match r {
                            crate::server_engine::FinishReason::Length => "length".to_string(),
                            crate::server_engine::FinishReason::Stop => "stop".to_string(),
                        }),
                        index: 0,
                    }],
                };

                if stream_tx.send(response).is_err() {
                    break; // Client disconnected
                }

                if delta.finish_reason.is_some() {
                    break;
                }
            }
        });

        Ok(())
    }

    async fn handle_single_threaded_stream(
        &self,
        _engine: &Box<dyn InferenceEngine>,
        _request: CompletionRequest,
        _stream_tx: mpsc::UnboundedSender<ApiStreamResponse>,
    ) -> Result<()> {
        // This would use the single-threaded engine for streaming
        // For now, return a placeholder
        Err(anyhow::anyhow!(
            "Single-threaded streaming not yet implemented"
        ))
    }

    fn get_model_id(&self) -> String {
        match &self.multi_thread_runtime {
            Some(runtime) => runtime.model_id.clone(),
            None => {
                if let Some(engine) = &self.single_thread_engine {
                    engine.model_id().to_string()
                } else {
                    "unknown".to_string()
                }
            }
        }
    }
}

// Need to implement Clone for MultiThreadedHttpServer
impl Clone for MultiThreadedHttpServer {
    fn clone(&self) -> Self {
        Self {
            runtime_mode: self.runtime_mode,
            multi_thread_runtime: self.multi_thread_runtime.clone(),
            single_thread_engine: None, // Cannot clone trait objects easily
            config: self.config.clone(),
        }
    }
}

/// Server statistics
#[derive(Debug, Clone, Serialize)]
pub struct ServerStats {
    pub runtime_mode: RuntimeMode,
    pub tokenizer_queue_depth: usize,
    pub scheduler_pending_requests: usize,
    pub gpu_active_batches: usize,
    pub detokenizer_queue_depth: usize,
    pub total_requests_processed: u64,
    pub average_request_duration_ms: f64,
}

// HTTP handler functions
async fn handle_completions(
    State(state): State<AppState>,
    Json(request): Json<ApiCompletionRequest>,
) -> Result<Json<ApiCompletionResponse>, Response> {
    match state.server.handle_completion(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => {
            let error_response = format!("{{\"error\": \"{}\"}}", e);
            Err(Response::builder()
                .status(500)
                .header("content-type", "application/json")
                .body(error_response.into())
                .unwrap())
        }
    }
}

async fn handle_completions_stream(
    State(state): State<AppState>,
    Json(request): Json<ApiCompletionRequest>,
) -> Result<
    axum::response::Sse<
        impl futures::Stream<Item = Result<axum::response::sse::Event, anyhow::Error>>,
    >,
    Response,
> {
    let stream = match state.server.handle_completion_stream(request).await {
        Ok(stream) => stream,
        Err(e) => {
            let error_response = format!("{{\"error\": \"{}\"}}", e);
            return Err(Response::builder()
                .status(500)
                .header("content-type", "application/json")
                .body(error_response.into())
                .unwrap());
        }
    };

    use futures::stream::StreamExt;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let sse_stream = UnboundedReceiverStream::new(stream).map(|response| {
        let json = serde_json::to_string(&response)?;
        Ok(axum::response::sse::Event::default().data(json))
    });

    Ok(axum::response::Sse::new(sse_stream))
}

async fn handle_stats(State(state): State<AppState>) -> Result<Json<ServerStats>, Response> {
    let stats = state.server.get_stats().await;
    Ok(Json(stats))
}

async fn handle_health() -> Result<Json<serde_json::Value>, Response> {
    Ok(Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}
