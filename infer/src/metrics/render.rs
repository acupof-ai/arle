use std::fmt::Write as FmtWrite;
use std::sync::atomic::Ordering;

use super::ServerMetrics;

impl ServerMetrics {
    // -----------------------------------------------------------------------
    // Prometheus text format rendering
    // -----------------------------------------------------------------------

    /// Render all metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let model = &self.inner.model_id;
        let labels = if model.is_empty() {
            String::new()
        } else {
            format!("model=\"{model}\",")
        };

        let mut out = String::new();

        // Counters
        out.push_str("# HELP infer_requests_total Total completed inference requests.\n");
        out.push_str("# TYPE infer_requests_total counter\n");
        writeln!(
            out,
            "infer_requests_total{{{labels}}} {}",
            self.inner.requests_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_tokens_generated_total Total output tokens generated.\n");
        out.push_str("# TYPE infer_tokens_generated_total counter\n");
        writeln!(
            out,
            "infer_tokens_generated_total{{{labels}}} {}",
            self.inner.tokens_generated_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_tokens_prompt_total Total prompt tokens processed.\n");
        out.push_str("# TYPE infer_tokens_prompt_total counter\n");
        writeln!(
            out,
            "infer_tokens_prompt_total{{{labels}}} {}",
            self.inner.tokens_prompt_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_requests_failed_total Total failed inference requests.\n");
        out.push_str("# TYPE infer_requests_failed_total counter\n");
        writeln!(
            out,
            "infer_requests_failed_total{{{labels}}} {}",
            self.inner.requests_failed_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_prefix_lookups_total Total prefix-cache lookups.\n");
        out.push_str("# TYPE infer_prefix_lookups_total counter\n");
        writeln!(
            out,
            "infer_prefix_lookups_total{{{labels}}} {}",
            self.inner.prefix_lookups_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_prefix_hits_total Total reusable prefix-cache hits.\n");
        out.push_str("# TYPE infer_prefix_hits_total counter\n");
        writeln!(
            out,
            "infer_prefix_hits_total{{{labels}}} {}",
            self.inner.prefix_hits_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_prefix_hit_rate Reusable prefix-cache hit rate [0,1].\n");
        out.push_str("# TYPE infer_prefix_hit_rate gauge\n");
        writeln!(
            out,
            "infer_prefix_hit_rate{{{labels}}} {:.4}",
            self.prefix_hit_rate()
        )
        .unwrap();

        out.push_str("# HELP infer_prefix_reused_tokens_total Prefix tokens skipped by reuse.\n");
        out.push_str("# TYPE infer_prefix_reused_tokens_total counter\n");
        writeln!(
            out,
            "infer_prefix_reused_tokens_total{{{labels}}} {}",
            self.inner
                .prefix_reused_tokens_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_prefix_lookup_prompt_tokens_total Prompt tokens seen by prefix lookup.\n",
        );
        out.push_str("# TYPE infer_prefix_lookup_prompt_tokens_total counter\n");
        writeln!(
            out,
            "infer_prefix_lookup_prompt_tokens_total{{{labels}}} {}",
            self.inner
                .prefix_lookup_prompt_tokens_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_prefix_skip_rate Fraction of prompt tokens skipped by prefix reuse [0,1].\n",
        );
        out.push_str("# TYPE infer_prefix_skip_rate gauge\n");
        writeln!(
            out,
            "infer_prefix_skip_rate{{{labels}}} {:.4}",
            self.prefix_skip_rate()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_tier_fetch_staged_host_blocks_total Request-weighted staged blocks found in T1.\n",
        );
        out.push_str("# TYPE infer_tier_fetch_staged_host_blocks_total counter\n");
        writeln!(
            out,
            "infer_tier_fetch_staged_host_blocks_total{{{labels}}} {}",
            self.tier_fetch_staged_host_blocks_total()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_tier_fetch_staged_disk_blocks_total Request-weighted staged blocks found in T2.\n",
        );
        out.push_str("# TYPE infer_tier_fetch_staged_disk_blocks_total counter\n");
        writeln!(
            out,
            "infer_tier_fetch_staged_disk_blocks_total{{{labels}}} {}",
            self.tier_fetch_staged_disk_blocks_total()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_tier_fetch_staged_remote_blocks_total Request-weighted staged blocks found in T3.\n",
        );
        out.push_str("# TYPE infer_tier_fetch_staged_remote_blocks_total counter\n");
        writeln!(
            out,
            "infer_tier_fetch_staged_remote_blocks_total{{{labels}}} {}",
            self.tier_fetch_staged_remote_blocks_total()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_tier_fetch_promoted_blocks_total Staged blocks promoted back into T0.\n",
        );
        out.push_str("# TYPE infer_tier_fetch_promoted_blocks_total counter\n");
        writeln!(
            out,
            "infer_tier_fetch_promoted_blocks_total{{{labels}}} {}",
            self.tier_fetch_promoted_blocks_total()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_tier_fetch_fallback_total Staged-prefix fallbacks back to cold prefill.\n",
        );
        out.push_str("# TYPE infer_tier_fetch_fallback_total counter\n");
        writeln!(
            out,
            "infer_tier_fetch_fallback_total{{{labels}}} {}",
            self.tier_fetch_fallback_total()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_tier_fetch_recall_rate Promoted staged blocks divided by staged blocks [0,1].\n",
        );
        out.push_str("# TYPE infer_tier_fetch_recall_rate gauge\n");
        writeln!(
            out,
            "infer_tier_fetch_recall_rate{{{labels}}} {:.4}",
            self.tier_fetch_recall_rate()
        )
        .unwrap();

        // DFlash speculative decode counters
        out.push_str(
            "# HELP infer_dflash_blocks_total Total DFlash speculative blocks executed.\n",
        );
        out.push_str("# TYPE infer_dflash_blocks_total counter\n");
        writeln!(
            out,
            "infer_dflash_blocks_total{{{labels}}} {}",
            self.inner.dflash_blocks_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_dflash_accepted_tokens_total Total tokens accepted from DFlash speculative blocks.\n");
        out.push_str("# TYPE infer_dflash_accepted_tokens_total counter\n");
        writeln!(
            out,
            "infer_dflash_accepted_tokens_total{{{labels}}} {}",
            self.inner
                .dflash_accepted_tokens_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_dflash_acceptance_rate DFlash acceptance rate: fraction of generated tokens from draft [0,1].\n",
        );
        out.push_str("# TYPE infer_dflash_acceptance_rate gauge\n");
        writeln!(
            out,
            "infer_dflash_acceptance_rate{{{labels}}} {:.4}",
            self.dflash_acceptance_rate()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_dflash_utilization DFlash speculative capacity utilization [0,1].\n",
        );
        out.push_str("# TYPE infer_dflash_utilization gauge\n");
        writeln!(
            out,
            "infer_dflash_utilization{{{labels}}} {:.4}",
            self.dflash_utilization()
        )
        .unwrap();

        out.push_str("# HELP infer_metal_decode_batches_total Metal decode batches executed on a batched GPU path.\n");
        out.push_str("# TYPE infer_metal_decode_batches_total counter\n");
        writeln!(
            out,
            "infer_metal_decode_batches_total{{{labels}}} {}",
            self.inner
                .metal_decode_batches_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_metal_decode_batched_rows_total Metal decode rows executed on a batched GPU path.\n");
        out.push_str("# TYPE infer_metal_decode_batched_rows_total counter\n");
        writeln!(
            out,
            "infer_metal_decode_batched_rows_total{{{labels}}} {}",
            self.inner
                .metal_decode_batched_rows_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_metal_decode_scalar_rows_total Metal decode rows executed by the scalar per-request path.\n");
        out.push_str("# TYPE infer_metal_decode_scalar_rows_total counter\n");
        writeln!(
            out,
            "infer_metal_decode_scalar_rows_total{{{labels}}} {}",
            self.inner
                .metal_decode_scalar_rows_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_metal_decode_batch_fallback_rows_total Metal decode rows scheduled together but forced to scalar fallback.\n");
        out.push_str("# TYPE infer_metal_decode_batch_fallback_rows_total counter\n");
        writeln!(
            out,
            "infer_metal_decode_batch_fallback_rows_total{{{labels}}} {}",
            self.inner
                .metal_decode_batch_fallback_rows_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_metal_qwen35_packed_decode_batches_total Qwen3.5 packed decode batches executed.\n");
        out.push_str("# TYPE infer_metal_qwen35_packed_decode_batches_total counter\n");
        writeln!(
            out,
            "infer_metal_qwen35_packed_decode_batches_total{{{labels}}} {}",
            self.inner
                .metal_qwen35_packed_decode_batches_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_metal_qwen35_packed_decode_rows_total Qwen3.5 packed decode rows executed.\n");
        out.push_str("# TYPE infer_metal_qwen35_packed_decode_rows_total counter\n");
        writeln!(
            out,
            "infer_metal_qwen35_packed_decode_rows_total{{{labels}}} {}",
            self.inner
                .metal_qwen35_packed_decode_rows_total
                .load(Ordering::Relaxed)
        )
        .unwrap();

        // Gauges
        out.push_str("# HELP infer_requests_active Currently running requests.\n");
        out.push_str("# TYPE infer_requests_active gauge\n");
        writeln!(
            out,
            "infer_requests_active{{{labels}}} {}",
            self.inner.requests_active.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_requests_waiting Requests waiting in queue.\n");
        out.push_str("# TYPE infer_requests_waiting gauge\n");
        writeln!(
            out,
            "infer_requests_waiting{{{labels}}} {}",
            self.inner.requests_waiting.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_running_batch Requests currently held in the running decode batch.\n",
        );
        out.push_str("# TYPE infer_scheduler_running_batch gauge\n");
        writeln!(
            out,
            "infer_scheduler_running_batch{{{labels}}} {}",
            self.inner.scheduler_running_batch.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_prefill_queue Requests currently queued for prefill continuation.\n",
        );
        out.push_str("# TYPE infer_scheduler_prefill_queue gauge\n");
        writeln!(
            out,
            "infer_scheduler_prefill_queue{{{labels}}} {}",
            self.inner.scheduler_prefill_queue.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_scheduled_rows Rows scheduled in the most recent scheduler tick.\n",
        );
        out.push_str("# TYPE infer_scheduler_scheduled_rows gauge\n");
        writeln!(
            out,
            "infer_scheduler_scheduled_rows{{{labels}}} {}",
            self.inner.scheduler_scheduled_rows.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_scheduled_decode_rows Decode rows scheduled in the most recent scheduler tick.\n",
        );
        out.push_str("# TYPE infer_scheduler_scheduled_decode_rows gauge\n");
        writeln!(
            out,
            "infer_scheduler_scheduled_decode_rows{{{labels}}} {}",
            self.inner
                .scheduler_scheduled_decode_rows
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_scheduled_prefill_rows Prefill rows scheduled in the most recent scheduler tick.\n",
        );
        out.push_str("# TYPE infer_scheduler_scheduled_prefill_rows gauge\n");
        writeln!(
            out,
            "infer_scheduler_scheduled_prefill_rows{{{labels}}} {}",
            self.inner
                .scheduler_scheduled_prefill_rows
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_decode_tokens Decode tokens advanced in the most recent scheduler tick.\n",
        );
        out.push_str("# TYPE infer_scheduler_decode_tokens gauge\n");
        writeln!(
            out,
            "infer_scheduler_decode_tokens{{{labels}}} {}",
            self.inner.scheduler_decode_tokens.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_prefill_tokens Prefill tokens advanced in the most recent scheduler tick.\n",
        );
        out.push_str("# TYPE infer_scheduler_prefill_tokens gauge\n");
        writeln!(
            out,
            "infer_scheduler_prefill_tokens{{{labels}}} {}",
            self.inner.scheduler_prefill_tokens.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_batch_width Total GPU batch width in the most recent scheduler tick.\n",
        );
        out.push_str("# TYPE infer_scheduler_batch_width gauge\n");
        writeln!(
            out,
            "infer_scheduler_batch_width{{{labels}}} {}",
            self.inner.scheduler_batch_width.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_scheduler_step_last_seconds Most recent end-to-end scheduler tick latency.\n",
        );
        out.push_str("# TYPE infer_scheduler_step_last_seconds gauge\n");
        writeln!(
            out,
            "infer_scheduler_step_last_seconds{{{labels}}} {:.6}",
            self.scheduler_step_last_seconds()
        )
        .unwrap();

        if let Some((
            phase_admission_us,
            phase_prefill_us,
            phase_decode_us,
            phase_emit_us,
            phase_total_us,
        )) = self.scheduler_step_phase_us()
        {
            for (name, help, value) in [
                (
                    "infer_scheduler_step_phase_admission_microseconds",
                    "EMA scheduler tick admission phase duration.",
                    phase_admission_us,
                ),
                (
                    "infer_scheduler_step_phase_prefill_microseconds",
                    "EMA scheduler tick prefill phase duration.",
                    phase_prefill_us,
                ),
                (
                    "infer_scheduler_step_phase_decode_microseconds",
                    "EMA scheduler tick decode phase duration.",
                    phase_decode_us,
                ),
                (
                    "infer_scheduler_step_phase_emit_microseconds",
                    "EMA scheduler tick emit phase duration.",
                    phase_emit_us,
                ),
                (
                    "infer_scheduler_step_phase_total_microseconds",
                    "EMA scheduler tick total duration.",
                    phase_total_us,
                ),
            ] {
                writeln!(out, "# HELP {name} {help}").unwrap();
                writeln!(out, "# TYPE {name} gauge").unwrap();
                writeln!(out, "{name}{{{labels}}} {value}").unwrap();
            }
        }

        out.push_str("# HELP infer_kv_coordinator_queue_capacity Coordinator queue capacity shared by staged KV fetch/store work.\n");
        out.push_str("# TYPE infer_kv_coordinator_queue_capacity gauge\n");
        writeln!(
            out,
            "infer_kv_coordinator_queue_capacity{{{labels}}} {}",
            self.inner
                .kv_coordinator_queue_capacity
                .load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_kv_fetch_queue_depth In-flight staged KV fetch tickets.\n");
        out.push_str("# TYPE infer_kv_fetch_queue_depth gauge\n");
        writeln!(
            out,
            "infer_kv_fetch_queue_depth{{{labels}}} {}",
            self.inner.kv_fetch_queue_depth.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_kv_fetch_waiters Requests currently waiting on staged KV fetch completion.\n");
        out.push_str("# TYPE infer_kv_fetch_waiters gauge\n");
        writeln!(
            out,
            "infer_kv_fetch_waiters{{{labels}}} {}",
            self.inner.kv_fetch_waiters.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_kv_store_queue_depth In-flight staged KV spill/store tickets.\n",
        );
        out.push_str("# TYPE infer_kv_store_queue_depth gauge\n");
        writeln!(
            out,
            "infer_kv_store_queue_depth{{{labels}}} {}",
            self.inner.kv_store_queue_depth.load(Ordering::Relaxed)
        )
        .unwrap();
        out.push_str(
            "# HELP infer_kv_store_submitted_total Submitted staged KV spill/store tickets.\n",
        );
        out.push_str("# TYPE infer_kv_store_submitted_total counter\n");
        writeln!(
            out,
            "infer_kv_store_submitted_total{{{labels}}} {}",
            self.inner.kv_store_submitted_total.load(Ordering::Relaxed)
        )
        .unwrap();
        out.push_str(
            "# HELP infer_kv_store_completed_total Completed staged KV spill/store tickets.\n",
        );
        out.push_str("# TYPE infer_kv_store_completed_total counter\n");
        writeln!(
            out,
            "infer_kv_store_completed_total{{{labels}}} {}",
            self.inner.kv_store_completed_total.load(Ordering::Relaxed)
        )
        .unwrap();
        out.push_str("# HELP infer_kv_store_failed_total Failed staged KV spill/store tickets.\n");
        out.push_str("# TYPE infer_kv_store_failed_total counter\n");
        writeln!(
            out,
            "infer_kv_store_failed_total{{{labels}}} {}",
            self.inner.kv_store_failed_total.load(Ordering::Relaxed)
        )
        .unwrap();
        out.push_str(
            "# HELP infer_kv_store_rejected_total Rejected staged KV spill/store tickets.\n",
        );
        out.push_str("# TYPE infer_kv_store_rejected_total counter\n");
        writeln!(
            out,
            "infer_kv_store_rejected_total{{{labels}}} {}",
            self.inner.kv_store_rejected_total.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_kv_fetch_backpressure Staged KV fetch queue backpressure flag (0 or 1).\n");
        out.push_str("# TYPE infer_kv_fetch_backpressure gauge\n");
        writeln!(
            out,
            "infer_kv_fetch_backpressure{{{labels}}} {}",
            self.inner.kv_fetch_backpressure.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_kv_store_backpressure Staged KV store queue backpressure flag (0 or 1).\n");
        out.push_str("# TYPE infer_kv_store_backpressure gauge\n");
        writeln!(
            out,
            "infer_kv_store_backpressure{{{labels}}} {}",
            self.inner.kv_store_backpressure.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str(
            "# HELP infer_tier_fetch_wait_seconds Oldest outstanding staged KV fetch wait.\n",
        );
        out.push_str("# TYPE infer_tier_fetch_wait_seconds gauge\n");
        writeln!(
            out,
            "infer_tier_fetch_wait_seconds{{{labels}}} {:.6}",
            self.tier_fetch_wait_seconds()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_tier_store_wait_seconds Oldest outstanding staged KV store wait.\n",
        );
        out.push_str("# TYPE infer_tier_store_wait_seconds gauge\n");
        writeln!(
            out,
            "infer_tier_store_wait_seconds{{{labels}}} {:.6}",
            self.tier_store_wait_seconds()
        )
        .unwrap();

        let total = self.inner.kv_gpu_blocks_total.load(Ordering::Relaxed);
        let free = self.inner.kv_gpu_blocks_free.load(Ordering::Relaxed);
        let utilization = if total == 0 {
            0.0
        } else {
            (total - free) as f64 / total as f64
        };

        out.push_str("# HELP infer_kv_gpu_utilization GPU KV cache utilization [0,1].\n");
        out.push_str("# TYPE infer_kv_gpu_utilization gauge\n");
        writeln!(out, "infer_kv_gpu_utilization{{{labels}}} {utilization:.4}").unwrap();

        out.push_str("# HELP infer_kv_gpu_blocks_free Free GPU KV cache blocks.\n");
        out.push_str("# TYPE infer_kv_gpu_blocks_free gauge\n");
        writeln!(out, "infer_kv_gpu_blocks_free{{{labels}}} {free}").unwrap();

        out.push_str("# HELP infer_kv_gpu_blocks_total Total GPU KV cache blocks.\n");
        out.push_str("# TYPE infer_kv_gpu_blocks_total gauge\n");
        writeln!(out, "infer_kv_gpu_blocks_total{{{labels}}} {total}").unwrap();

        out.push_str("# HELP infer_memory_active_bytes Active MLX allocator memory in bytes.\n");
        out.push_str("# TYPE infer_memory_active_bytes gauge\n");
        writeln!(
            out,
            "infer_memory_active_bytes{{{labels}}} {}",
            self.inner.memory_active_bytes.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_memory_peak_bytes Peak MLX allocator memory in bytes.\n");
        out.push_str("# TYPE infer_memory_peak_bytes gauge\n");
        writeln!(
            out,
            "infer_memory_peak_bytes{{{labels}}} {}",
            self.inner.memory_peak_bytes.load(Ordering::Relaxed)
        )
        .unwrap();

        out.push_str("# HELP infer_memory_cache_bytes Cached MLX allocator memory in bytes.\n");
        out.push_str("# TYPE infer_memory_cache_bytes gauge\n");
        writeln!(
            out,
            "infer_memory_cache_bytes{{{labels}}} {}",
            self.inner.memory_cache_bytes.load(Ordering::Relaxed)
        )
        .unwrap();

        // Histograms
        if let Ok(h) = self.inner.histograms.lock() {
            out.push_str("# HELP infer_queue_wait_seconds Submit-to-admit queue latency.\n");
            out.push_str("# TYPE infer_queue_wait_seconds histogram\n");
            out.push_str(&h.queue_wait.render("infer_queue_wait_seconds", &labels));

            out.push_str("# HELP infer_active_ttft_seconds Admit-to-first-token latency.\n");
            out.push_str("# TYPE infer_active_ttft_seconds histogram\n");
            out.push_str(&h.active_ttft.render("infer_active_ttft_seconds", &labels));

            out.push_str("# HELP infer_ttft_seconds Time to first token latency.\n");
            out.push_str("# TYPE infer_ttft_seconds histogram\n");
            out.push_str(&h.ttft.render("infer_ttft_seconds", &labels));

            out.push_str("# HELP infer_tpot_seconds Time per output token latency.\n");
            out.push_str("# TYPE infer_tpot_seconds histogram\n");
            out.push_str(&h.tpot.render("infer_tpot_seconds", &labels));

            out.push_str("# HELP infer_service_seconds First-token-to-finish service latency.\n");
            out.push_str("# TYPE infer_service_seconds histogram\n");
            out.push_str(&h.service.render("infer_service_seconds", &labels));

            out.push_str("# HELP infer_e2e_seconds End-to-end request latency.\n");
            out.push_str("# TYPE infer_e2e_seconds histogram\n");
            out.push_str(&h.e2e.render("infer_e2e_seconds", &labels));

            out.push_str(
                "# HELP infer_scheduler_step_seconds End-to-end scheduler tick latency.\n",
            );
            out.push_str("# TYPE infer_scheduler_step_seconds histogram\n");
            out.push_str(
                &h.scheduler_step
                    .render("infer_scheduler_step_seconds", &labels),
            );
        }

        out
    }

    /// Render a simple human-readable summary (for `/v1/stats` or logging).
    pub fn render_summary(&self) -> String {
        let histograms = self.inner.histograms.lock().ok();
        let ttft_p50 = histograms
            .as_ref()
            .and_then(|h| h.ttft.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let queue_p50 = histograms
            .as_ref()
            .and_then(|h| h.queue_wait.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let active_ttft_p50 = histograms
            .as_ref()
            .and_then(|h| h.active_ttft.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let ttft_p99 = histograms
            .as_ref()
            .and_then(|h| h.ttft.percentile(0.99))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let tpot_p50 = histograms
            .as_ref()
            .and_then(|h| h.tpot.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let step_p50 = histograms
            .as_ref()
            .and_then(|h| h.scheduler_step.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let service_p50 = histograms
            .as_ref()
            .and_then(|h| h.service.percentile(0.50))
            .map_or_else(|| "—".to_string(), |v| format!("{:.1}ms", v * 1000.0));
        let active_mb =
            self.inner.memory_active_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let peak_mb =
            self.inner.memory_peak_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let cache_mb =
            self.inner.memory_cache_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let phase_suffix = self.scheduler_step_phase_us().map_or_else(
            || " step_phase_us=unavailable".to_string(),
            |(admission_us, prefill_us, decode_us, emit_us, total_us)| {
                format!(
                    " step_phase_us=adm:{},prefill:{},decode:{},emit:{},total:{}",
                    admission_us, prefill_us, decode_us, emit_us, total_us
                )
            },
        );

        let dflash_blocks = self.inner.dflash_blocks_total.load(Ordering::Relaxed);
        let dflash_suffix = if dflash_blocks > 0 {
            format!(
                " dflash_blocks={} dflash_accept={:.1}% util={:.1}%",
                dflash_blocks,
                self.dflash_acceptance_rate() * 100.0,
                self.dflash_utilization() * 100.0,
            )
        } else {
            String::new()
        };
        let metal_decode_suffix = format!(
            " metal_decode=batch:{}/{},scalar:{},fallback:{},qwen35_packed:{}/{}",
            self.inner
                .metal_decode_batches_total
                .load(Ordering::Relaxed),
            self.inner
                .metal_decode_batched_rows_total
                .load(Ordering::Relaxed),
            self.inner
                .metal_decode_scalar_rows_total
                .load(Ordering::Relaxed),
            self.inner
                .metal_decode_batch_fallback_rows_total
                .load(Ordering::Relaxed),
            self.inner
                .metal_qwen35_packed_decode_batches_total
                .load(Ordering::Relaxed),
            self.inner
                .metal_qwen35_packed_decode_rows_total
                .load(Ordering::Relaxed),
        );
        let queue_capacity = self.kv_coordinator_queue_capacity();
        let coordinator_suffix = if queue_capacity > 0 {
            format!(
                " kv_fetch_q={}/{} kv_fetch_waiters={} kv_store_q={}/{} kv_store=sub:{},done:{},fail:{},rej:{} kv_bp=fetch:{},store:{}",
                self.kv_fetch_queue_depth(),
                queue_capacity,
                self.kv_fetch_waiters(),
                self.kv_store_queue_depth(),
                queue_capacity,
                self.kv_store_submitted_total(),
                self.kv_store_completed_total(),
                self.kv_store_failed_total(),
                self.kv_store_rejected_total(),
                u8::from(self.kv_fetch_backpressure()),
                u8::from(self.kv_store_backpressure()),
            )
        } else {
            String::new()
        };
        let staged_blocks = self.tier_fetch_staged_blocks_total();
        let tier_suffix = if staged_blocks > 0 || self.tier_fetch_fallback_total() > 0 {
            format!(
                " prefix_skip_rate={:.1}% tier_recall={:.1}% tier_src=h:{}/d:{}/r:{} tier_promoted={} tier_fallback={}",
                self.prefix_skip_rate() * 100.0,
                self.tier_fetch_recall_rate() * 100.0,
                self.tier_fetch_staged_host_blocks_total(),
                self.tier_fetch_staged_disk_blocks_total(),
                self.tier_fetch_staged_remote_blocks_total(),
                self.tier_fetch_promoted_blocks_total(),
                self.tier_fetch_fallback_total(),
            )
        } else {
            format!(" prefix_skip_rate={:.1}%", self.prefix_skip_rate() * 100.0)
        };

        format!(
            "requests={} active={} waiting={} scheduled={} decode_rows={} prefill_rows={} running_batch={} prefill_queue={} batch_width={} decode_tokens={} prefill_tokens={} tokens_out={} step_last={:.1}ms step_p50={}{} tier_fetch_wait={:.1}ms tier_store_wait={:.1}ms kv_util={:.1}% prefix_hit_rate={:.1}% active_mem={:.1}MB peak_mem={:.1}MB cache_mem={:.1}MB queue_p50={} active_ttft_p50={} ttft_p50={} ttft_p99={} service_p50={} tpot_p50={}{}{}{}{}",
            self.requests_total(),
            self.requests_active(),
            self.requests_waiting(),
            self.scheduler_scheduled_rows(),
            self.scheduler_scheduled_decode_rows(),
            self.scheduler_scheduled_prefill_rows(),
            self.scheduler_running_batch(),
            self.scheduler_prefill_queue(),
            self.scheduler_batch_width(),
            self.scheduler_decode_tokens(),
            self.scheduler_prefill_tokens(),
            self.tokens_generated_total(),
            self.scheduler_step_last_seconds() * 1000.0,
            step_p50,
            phase_suffix,
            self.tier_fetch_wait_seconds() * 1000.0,
            self.tier_store_wait_seconds() * 1000.0,
            self.kv_gpu_utilization() * 100.0,
            self.prefix_hit_rate() * 100.0,
            active_mb,
            peak_mb,
            cache_mb,
            queue_p50,
            active_ttft_p50,
            ttft_p50,
            ttft_p99,
            service_p50,
            tpot_p50,
            metal_decode_suffix,
            dflash_suffix,
            tier_suffix,
            coordinator_suffix,
        )
    }
}
