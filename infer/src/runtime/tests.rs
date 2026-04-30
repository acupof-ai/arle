/*!
 * Comprehensive test suite for multi-threaded runtime
 *
 * Thread safety validation: race condition detection, load testing with concurrent requests,
 * stress testing for channel backpressure, and correctness verification
 */

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::Barrier;
    use tokio::time::timeout;

    use crate::runtime::{
        MultiThreadRuntime, RuntimeConfig, RuntimeMode, ThreadingConfig,
        error_recovery::{ErrorCategory, ErrorRecoveryConfig, ErrorRecoveryCoordinator},
        memory_pool_coordinator::{MemoryPoolConfig, MemoryPoolCoordinator},
        thread_safe_radix_cache::{RadixCacheConfig, ThreadSafeRadixCache},
    };

    /// Mock backend for testing
    #[derive(Clone)]
    struct MockBackend {
        model_id: String,
        delay: Duration,
        should_fail: bool,
    }

    impl MockBackend {
        fn new(model_id: &str, delay: Duration) -> Self {
            Self {
                model_id: model_id.to_string(),
                delay,
                should_fail: false,
            }
        }

        fn with_failure(mut self) -> Self {
            self.should_fail = true;
            self
        }
    }

    impl crate::backend::InferenceBackend for MockBackend {
        fn load(&mut self, _model_path: &std::path::Path) -> anyhow::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            prompt: &str,
            _params: &crate::sampler::SamplingParams,
        ) -> anyhow::Result<crate::backend::GenerateResult> {
            if self.should_fail {
                return Err(anyhow::anyhow!("Mock backend failure"));
            }

            // Simulate processing time
            std::thread::sleep(self.delay);

            Ok(crate::backend::GenerateResult {
                text: format!("Mock response to: {}", prompt),
                prompt_tokens: prompt.split_whitespace().count(),
                completion_tokens: 10,
                finish_reason: "stop".to_string(),
                ttft_ms: self.delay.as_secs_f64() * 1000.0,
                prompt_tps: 1000.0,
                generation_tps: 50.0,
                total_time_ms: self.delay.as_secs_f64() * 1000.0,
            })
        }

        fn name(&self) -> &'static str {
            "mock"
        }

        fn tokenize(&self, text: &str) -> anyhow::Result<Vec<u32>> {
            // Simple mock tokenization
            Ok(text.chars().map(|c| c as u32).collect())
        }
    }

    /// Mock tokenizer for testing
    struct MockTokenizer;

    impl crate::tokenizer::Tokenizer for MockTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
            // Simple mock: each character becomes a token
            Ok(text.chars().map(|c| c as u32).collect())
        }

        fn decode(&self, tokens: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            // Simple mock: each token becomes a character
            Ok(tokens.iter().map(|&t| t as u8 as char).collect())
        }

        fn vocab_size(&self) -> usize {
            65536
        }
    }

    /// Create test runtime configuration
    fn create_test_config() -> RuntimeConfig {
        RuntimeConfig {
            mode: RuntimeMode::MultiThreaded,
            threading: ThreadingConfig {
                tokenizer_workers: 2,
                detokenizer_workers: 2,
                gpu_workers_per_device: 1,
                scheduler_tick_ms: 1,
            },
            ..RuntimeConfig::development()
        }
    }

    /// Create test runtime
    async fn create_test_runtime() -> anyhow::Result<MultiThreadRuntime> {
        let backend = Box::new(MockBackend::new("test-model", Duration::from_millis(10)));
        let tokenizer = Arc::new(MockTokenizer);
        let config = create_test_config();

        MultiThreadRuntime::new(backend, tokenizer, config).await
    }

    #[tokio::test]
    async fn test_runtime_basic_functionality() -> anyhow::Result<()> {
        let runtime = create_test_runtime().await?;
        runtime.start().await?;

        let request = crate::server_engine::CompletionRequest {
            prompt: "Hello world".to_string(),
            max_tokens: 10,
            sampling: crate::sampler::SamplingParams::default(),
            stop: None,
            logprobs: false,
            session_id: None,
            trace_context: None,
        };

        let mut stream = runtime.submit_request(request).await?;

        // Should receive at least one response
        let response = timeout(Duration::from_secs(5), stream.recv()).await??;
        assert!(response.is_some());

        runtime.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_requests() -> anyhow::Result<()> {
        let runtime = Arc::new(create_test_runtime().await?);
        runtime.start().await?;

        let num_requests = 10;
        let mut handles = Vec::new();

        // Submit concurrent requests
        for i in 0..num_requests {
            let runtime = runtime.clone();
            let handle = tokio::spawn(async move {
                let request = crate::server_engine::CompletionRequest {
                    prompt: format!("Request {}", i),
                    max_tokens: 5,
                    sampling: crate::sampler::SamplingParams::default(),
                    stop: None,
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                };

                let mut stream = runtime.submit_request(request).await?;

                // Collect all responses
                let mut responses = Vec::new();
                while let Some(response) = stream.recv().await {
                    responses.push(response);
                    if response.finish_reason.is_some() {
                        break;
                    }
                }

                Ok::<Vec<_>, anyhow::Error>(responses)
            });

            handles.push(handle);
        }

        // Wait for all requests to complete
        let start = Instant::now();
        let results: anyhow::Result<Vec<_>> = futures::future::try_join_all(handles)
            .await
            .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
            .into_iter()
            .collect();

        let duration = start.elapsed();
        println!("Processed {} requests in {:?}", num_requests, duration);

        let results = results?;
        assert_eq!(results.len(), num_requests);

        // Verify all requests got responses
        for result in &results {
            assert!(
                !result.is_empty(),
                "Request should have received at least one response"
            );
        }

        runtime.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_channel_backpressure() -> anyhow::Result<()> {
        // Create runtime with small channel buffers
        let mut config = create_test_config();
        config.channel_config.buffer_sizes.tokenizer_buffer = 2;
        config.channel_config.buffer_sizes.scheduler_buffer = 2;

        let backend = Box::new(MockBackend::new("test-model", Duration::from_millis(100))); // Slow backend
        let tokenizer = Arc::new(MockTokenizer);

        let runtime = Arc::new(MultiThreadRuntime::new(backend, tokenizer, config).await?);
        runtime.start().await?;

        let num_requests = 20; // More than buffer capacity
        let barrier = Arc::new(Barrier::new(num_requests));
        let mut handles = Vec::new();

        // Submit many requests simultaneously to test backpressure
        for i in 0..num_requests {
            let runtime = runtime.clone();
            let barrier = barrier.clone();

            let handle = tokio::spawn(async move {
                // Wait for all tasks to be ready
                barrier.wait().await;

                let request = crate::server_engine::CompletionRequest {
                    prompt: format!("Backpressure test {}", i),
                    max_tokens: 5,
                    sampling: crate::sampler::SamplingParams::default(),
                    stop: None,
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                };

                runtime.submit_request(request).await
            });

            handles.push(handle);
        }

        // All requests should eventually succeed (no deadlocks)
        let start = Instant::now();
        let results: anyhow::Result<Vec<_>> = futures::future::try_join_all(handles)
            .await
            .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
            .into_iter()
            .collect();

        let duration = start.elapsed();
        println!("Handled backpressure test in {:?}", duration);

        let results = results?;
        assert_eq!(results.len(), num_requests);

        runtime.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_error_recovery() -> anyhow::Result<()> {
        let mut error_config = ErrorRecoveryConfig::default();
        error_config.enable_auto_recovery = true;

        let mut coordinator = ErrorRecoveryCoordinator::new(error_config);
        coordinator.start().await?;

        // Simulate various error scenarios
        let test_errors = vec![
            (anyhow::anyhow!("Worker panic"), ErrorCategory::WorkerPanic),
            (anyhow::anyhow!("GPU error"), ErrorCategory::GpuError),
            (
                anyhow::anyhow!("Memory allocation failed"),
                ErrorCategory::MemoryError,
            ),
        ];

        for (error, category) in test_errors {
            coordinator.report_error(error, category, None).await?;
        }

        // Check system health degradation
        tokio::time::sleep(Duration::from_millis(100)).await;
        let health = coordinator.system_health().await;
        println!("System health after errors: {:?}", health);

        let stats = coordinator.error_stats().await;
        assert!(stats.total_errors > 0);

        coordinator.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_memory_pool_coordination() -> anyhow::Result<()> {
        let config = MemoryPoolConfig {
            kv_cache_fraction: 0.5,
            host_memory_bytes: 1024 * 1024 * 1024, // 1GB
            default_worker_quota_fraction: 0.1,
            enable_rebalancing: true,
            rebalancing_interval: Duration::from_millis(100),
            quota_enforcement: super::memory_pool_coordinator::QuotaEnforcementMode::Soft,
        };

        let coordinator = MemoryPoolCoordinator::new(config).await?;
        coordinator.start().await?;

        // Test allocation and deallocation
        let allocation1 = coordinator
            .allocate(
                super::memory_pool_coordinator::PoolId::HostMemory,
                1024,
                0, // worker_id
            )
            .await?;

        let allocation2 = coordinator
            .allocate(
                super::memory_pool_coordinator::PoolId::HostMemory,
                2048,
                1, // worker_id
            )
            .await?;

        // Check stats
        let stats = coordinator.stats().await;
        assert!(stats.total_allocated_bytes > 0);

        // Test deallocation
        coordinator.deallocate(allocation1).await?;
        coordinator.deallocate(allocation2).await?;

        coordinator.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_radix_cache_thread_safety() -> anyhow::Result<()> {
        let config = RadixCacheConfig::default();
        let cache = Arc::new(ThreadSafeRadixCache::new(config));

        let num_threads = 10;
        let operations_per_thread = 100;
        let mut handles = Vec::new();

        // Test concurrent access
        for thread_id in 0..num_threads {
            let cache = cache.clone();

            let handle = tokio::spawn(async move {
                for i in 0..operations_per_thread {
                    let tokens = vec![thread_id as u32, i];

                    // Lookup (should not find initially)
                    let lookup_result = cache.lookup(&tokens).await?;
                    assert_eq!(lookup_result.matched_tokens, 0);

                    // Insert
                    let blocks = vec![super::thread_safe_radix_cache::CacheBlock {
                        block_id: (thread_id * 1000 + i) as u32,
                        token_offset: 0,
                        token_count: tokens.len(),
                    }];
                    cache.insert(&tokens, blocks).await?;

                    // Lookup again (should find now)
                    let lookup_result = cache.lookup(&tokens).await?;
                    assert_eq!(lookup_result.matched_tokens, tokens.len());
                }

                Ok::<(), anyhow::Error>(())
            });

            handles.push(handle);
        }

        // Wait for all operations to complete
        let results: anyhow::Result<Vec<_>> = futures::future::try_join_all(handles)
            .await
            .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
            .into_iter()
            .collect();

        results?;

        // Check final cache stats
        let stats = cache.stats().await;
        println!("Cache stats: {:?}", stats);
        assert!(stats.total_nodes > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_runtime_shutdown_gracefully() -> anyhow::Result<()> {
        let runtime = Arc::new(create_test_runtime().await?);
        runtime.start().await?;

        // Start some background requests
        let runtime_clone = runtime.clone();
        let background_task = tokio::spawn(async move {
            for i in 0..5 {
                let request = crate::server_engine::CompletionRequest {
                    prompt: format!("Background request {}", i),
                    max_tokens: 10,
                    sampling: crate::sampler::SamplingParams::default(),
                    stop: None,
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                };

                if let Ok(mut stream) = runtime_clone.submit_request(request).await {
                    while let Some(_) = stream.recv().await {
                        // Process response
                    }
                }

                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });

        // Let some requests start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Shutdown should complete even with active requests
        let shutdown_start = Instant::now();
        runtime.shutdown().await?;
        let shutdown_duration = shutdown_start.elapsed();

        println!("Graceful shutdown took: {:?}", shutdown_duration);
        assert!(
            shutdown_duration < Duration::from_secs(5),
            "Shutdown took too long"
        );

        // Background task should complete or be cancelled
        let _ = background_task.await;

        Ok(())
    }

    #[tokio::test]
    async fn test_load_spike_handling() -> anyhow::Result<()> {
        let runtime = Arc::new(create_test_runtime().await?);
        runtime.start().await?;

        // Simulate a load spike
        let spike_requests = 50;
        let mut wave1_handles = Vec::new();
        let mut wave2_handles = Vec::new();

        // First wave of requests
        for i in 0..spike_requests {
            let runtime = runtime.clone();
            let handle = tokio::spawn(async move {
                let request = crate::server_engine::CompletionRequest {
                    prompt: format!("Wave1 request {}", i),
                    max_tokens: 5,
                    sampling: crate::sampler::SamplingParams::default(),
                    stop: None,
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                };

                runtime.submit_request(request).await
            });
            wave1_handles.push(handle);
        }

        // Second wave after a short delay
        tokio::time::sleep(Duration::from_millis(50)).await;

        for i in 0..spike_requests {
            let runtime = runtime.clone();
            let handle = tokio::spawn(async move {
                let request = crate::server_engine::CompletionRequest {
                    prompt: format!("Wave2 request {}", i),
                    max_tokens: 5,
                    sampling: crate::sampler::SamplingParams::default(),
                    stop: None,
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                };

                runtime.submit_request(request).await
            });
            wave2_handles.push(handle);
        }

        // Measure response times
        let start = Instant::now();

        let wave1_results: anyhow::Result<Vec<_>> = futures::future::try_join_all(wave1_handles)
            .await
            .map_err(|e| anyhow::anyhow!("Wave1 error: {}", e))?
            .into_iter()
            .collect();

        let wave2_results: anyhow::Result<Vec<_>> = futures::future::try_join_all(wave2_handles)
            .await
            .map_err(|e| anyhow::anyhow!("Wave2 error: {}", e))?
            .into_iter()
            .collect();

        let total_duration = start.elapsed();

        wave1_results?;
        wave2_results?;

        println!(
            "Load spike test: {} requests in {:?}",
            spike_requests * 2,
            total_duration
        );

        // Check runtime stats
        let stats = runtime.stats();
        println!("Final runtime stats: {:?}", stats);

        runtime.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_memory_pressure_handling() -> anyhow::Result<()> {
        // Create runtime with limited memory
        let mut config = create_test_config();
        config.memory.kv_cache_fraction = 0.1; // Very small cache

        let backend = Box::new(MockBackend::new("test-model", Duration::from_millis(10)));
        let tokenizer = Arc::new(MockTokenizer);

        let runtime = Arc::new(MultiThreadRuntime::new(backend, tokenizer, config).await?);
        runtime.start().await?;

        // Generate requests that should cause memory pressure
        let large_requests = 20;
        let mut handles = Vec::new();

        for i in 0..large_requests {
            let runtime = runtime.clone();
            let handle = tokio::spawn(async move {
                let request = crate::server_engine::CompletionRequest {
                    prompt: format!(
                        "Large request {} with lots of text to cause memory pressure and trigger eviction",
                        i
                    ),
                    max_tokens: 100,
                    sampling: crate::sampler::SamplingParams::default(),
                    stop: None,
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                };

                runtime.submit_request(request).await
            });
            handles.push(handle);
        }

        // All requests should still complete successfully
        let results: anyhow::Result<Vec<_>> = futures::future::try_join_all(handles)
            .await
            .map_err(|e| anyhow::anyhow!("Memory pressure test error: {}", e))?
            .into_iter()
            .collect();

        results?;

        runtime.shutdown().await?;
        Ok(())
    }

    /// Property-based test for thread safety
    #[tokio::test]
    async fn property_test_thread_safety() -> anyhow::Result<()> {
        use rand::Rng;

        let runtime = Arc::new(create_test_runtime().await?);
        runtime.start().await?;

        let num_workers = 20;
        let operations_per_worker = 50;
        let mut handles = Vec::new();

        for worker_id in 0..num_workers {
            let runtime = runtime.clone();

            let handle = tokio::spawn(async move {
                let mut rng = rand::thread_rng();
                let mut successful_operations = 0;

                for _ in 0..operations_per_worker {
                    // Random request parameters
                    let prompt_length: usize = rng.gen_range(1..100);
                    let max_tokens: usize = rng.gen_range(1..50);
                    let delay: u64 = rng.gen_range(0..10);

                    tokio::time::sleep(Duration::from_millis(delay)).await;

                    let request = crate::server_engine::CompletionRequest {
                        prompt: "a".repeat(prompt_length),
                        max_tokens,
                        sampling: crate::sampler::SamplingParams::default(),
                        stop: None,
                        logprobs: false,
                        session_id: Some(format!("worker-{}", worker_id)),
                        trace_context: None,
                    };

                    match runtime.submit_request(request).await {
                        Ok(mut stream) => {
                            // Try to receive at least one response
                            if let Ok(Some(_)) =
                                timeout(Duration::from_secs(1), stream.recv()).await
                            {
                                successful_operations += 1;
                            }
                        }
                        Err(_) => {
                            // Some failures are acceptable under load
                        }
                    }
                }

                successful_operations
            });

            handles.push(handle);
        }

        let results: anyhow::Result<Vec<_>> = futures::future::try_join_all(handles)
            .await
            .map_err(|e| anyhow::anyhow!("Property test error: {}", e))?
            .into_iter()
            .collect();

        let total_successful: usize = results?.into_iter().sum();
        let total_operations = num_workers * operations_per_worker;
        let success_rate = total_successful as f64 / total_operations as f64;

        println!(
            "Property test: {}/{} operations successful ({:.2}% success rate)",
            total_successful,
            total_operations,
            success_rate * 100.0
        );

        // We expect at least 80% success rate under random load
        assert!(
            success_rate >= 0.8,
            "Success rate too low: {:.2}%",
            success_rate * 100.0
        );

        runtime.shutdown().await?;
        Ok(())
    }
}
