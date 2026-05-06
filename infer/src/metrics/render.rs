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
            "# HELP infer_prefix_request_hit_rate Prefix-cache hit rate for the most recent lookup [0,1].\n",
        );
        out.push_str("# TYPE infer_prefix_request_hit_rate gauge\n");
        writeln!(
            out,
            "infer_prefix_request_hit_rate{{{labels}}} {:.4}",
            self.prefix_request_hit_rate()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_prefix_request_skip_rate Prefix-token skip rate for the most recent lookup [0,1].\n",
        );
        out.push_str("# TYPE infer_prefix_request_skip_rate gauge\n");
        writeln!(
            out,
            "infer_prefix_request_skip_rate{{{labels}}} {:.4}",
            self.prefix_request_skip_rate()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_session_affinity_hit_total Session-tagged requests that reused a prefix.\n",
        );
        out.push_str("# TYPE infer_session_affinity_hit_total counter\n");
        writeln!(
            out,
            "infer_session_affinity_hit_total{{{labels}}} {}",
            self.session_affinity_hit_total()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_session_affinity_miss_total Session-tagged requests without prefix reuse.\n",
        );
        out.push_str("# TYPE infer_session_affinity_miss_total counter\n");
        writeln!(
            out,
            "infer_session_affinity_miss_total{{{labels}}} {}",
            self.session_affinity_miss_total()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_session_slot_pressure_evictions_hard_total Inactive session slots evicted under hard pressure.\n",
        );
        out.push_str("# TYPE infer_session_slot_pressure_evictions_hard_total counter\n");
        writeln!(
            out,
            "infer_session_slot_pressure_evictions_hard_total{{{labels}}} {}",
            self.session_slot_pressure_evictions_hard()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_matched_prefix_tokens Matched prefix tokens for the most recent lookup.\n",
        );
        out.push_str("# TYPE infer_matched_prefix_tokens gauge\n");
        writeln!(
            out,
            "infer_matched_prefix_tokens{{{labels}}} {}",
            self.matched_prefix_tokens()
        )
        .unwrap();

        out.push_str(
            "# HELP infer_resume_prefill_tokens Effective prefill tokens for the most recent lookup.\n",
        );
        out.push_str("# TYPE infer_resume_prefill_tokens gauge\n");
        writeln!(
            out,
            "infer_resume_prefill_tokens{{{labels}}} {}",
            self.resume_prefill_tokens()
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
        if let Some((phase_cleanup_us, loop_total_us)) = self.scheduler_loop_phase_us() {
            for (name, help, value) in [
                (
                    "infer_scheduler_step_cleanup_microseconds",
                    "EMA scheduler cleanup duration after GPU step dispatch/readback.",
                    phase_cleanup_us,
                ),
                (
                    "infer_scheduler_loop_total_microseconds",
                    "EMA full scheduler loop duration including cleanup.",
                    loop_total_us,
                ),
            ] {
                writeln!(out, "# HELP {name} {help}").unwrap();
                writeln!(out, "# TYPE {name} gauge").unwrap();
                writeln!(out, "{name}{{{labels}}} {value}").unwrap();
            }
        }

        out.push_str("# HELP infer_scheduler_plan_total Scheduler ticks by selected plan label.\n");
        out.push_str("# TYPE infer_scheduler_plan_total counter\n");
        let (plan_idle, plan_decode, plan_prefill, plan_split, plan_mixed) =
            self.scheduler_plan_totals();
        for (plan, value) in [
            ("idle", plan_idle),
            ("decode", plan_decode),
            ("prefill", plan_prefill),
            ("split", plan_split),
            ("mixed", plan_mixed),
        ] {
            writeln!(
                out,
                "infer_scheduler_plan_total{{{labels}plan=\"{plan}\",}} {value}"
            )
            .unwrap();
        }

        out.push_str("# HELP infer_spec_draft_tokens_total Draft tokens proposed by Phase 2 speculative decode.\n");
        out.push_str("# TYPE infer_spec_draft_tokens_total counter\n");
        writeln!(
            out,
            "infer_spec_draft_tokens_total{{{labels}}} {}",
            self.spec_draft_tokens_total()
        )
        .unwrap();
        out.push_str("# HELP infer_spec_verified_tokens_total Draft tokens checked by the target verifier.\n");
        out.push_str("# TYPE infer_spec_verified_tokens_total counter\n");
        writeln!(
            out,
            "infer_spec_verified_tokens_total{{{labels}}} {}",
            self.spec_verified_tokens_total()
        )
        .unwrap();
        out.push_str(
            "# HELP infer_spec_accepted_tokens_total Draft tokens accepted by the verifier.\n",
        );
        out.push_str("# TYPE infer_spec_accepted_tokens_total counter\n");
        writeln!(
            out,
            "infer_spec_accepted_tokens_total{{{labels}}} {}",
            self.spec_accepted_tokens_total()
        )
        .unwrap();
        out.push_str("# HELP infer_spec_sparse_view_empty_total Sparse self-spec decode rows that could not build a sparse KV view.\n");
        out.push_str("# TYPE infer_spec_sparse_view_empty_total counter\n");
        writeln!(
            out,
            "infer_spec_sparse_view_empty_total{{{labels}}} {}",
            self.spec_sparse_view_empty_total()
        )
        .unwrap();
        out.push_str("# HELP infer_spec_acceptance_rate Aggregate speculative accepted / verified token ratio [0,1].\n");
        out.push_str("# TYPE infer_spec_acceptance_rate gauge\n");
        writeln!(
            out,
            "infer_spec_acceptance_rate{{{labels}}} {:.6}",
            self.spec_acceptance_rate()
        )
        .unwrap();

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

            out.push_str(
                "# HELP infer_spec_step_latency_us Phase 2 speculative decode step latency.\n",
            );
            out.push_str("# TYPE infer_spec_step_latency_us histogram\n");
            out.push_str(
                &h.spec_step_latency_us
                    .render("infer_spec_step_latency_us", &labels),
            );
        }

        out
    }

    /// Project the rolling `ServerMetrics` counters into the
    /// backend-agnostic `EngineTelemetry` snapshot used by
    /// `InferenceEngine::telemetry()` and the HTTP `/v1/stats` engine_*
    /// fields. CUDA and Metal both write into the same `ServerMetrics`
    /// instance — this method is the single projection point.
    ///
    /// Tier hit rates are keyed by canonical labels (`"T0"` / `"T1"` /
    /// `"T2"` / `"T3"`). Backends that do not populate a tier (e.g. Metal
    /// has unified memory, no T1) simply leave the corresponding counter
    /// at 0 and the entry is omitted.
    pub fn snapshot_engine_telemetry(&self) -> crate::server_engine::EngineTelemetry {
        let (ttft_us, itl_p50_us, itl_p99_us) = self
            .inner
            .histograms
            .lock()
            .ok()
            .map(|h| {
                let ttft = h.ttft.percentile(0.50).map(|s| s * 1_000_000.0);
                let itl_p50 = h.tpot.percentile(0.50).map(|s| s * 1_000_000.0);
                let itl_p99 = h.tpot.percentile(0.99).map(|s| s * 1_000_000.0);
                (ttft, itl_p50, itl_p99)
            })
            .unwrap_or((None, None, None));

        self.snapshot_engine_telemetry_with(ttft_us, itl_p50_us, itl_p99_us)
    }

    /// Pure helper: build an [`EngineTelemetry`] when the caller already
    /// has the histogram-derived percentiles. Callers MUST NOT hold
    /// `self.inner.histograms` locked across this method (the public
    /// `snapshot_engine_telemetry` shim does the lock + drop for the
    /// common case). The split exists so paths like `render_summary`,
    /// which already lock `histograms` to extract their own percentiles,
    /// can build the telemetry without re-locking the same non-reentrant
    /// `std::sync::Mutex` (would deadlock).
    pub fn snapshot_engine_telemetry_with(
        &self,
        ttft_us: Option<f64>,
        itl_p50_us: Option<f64>,
        itl_p99_us: Option<f64>,
    ) -> crate::server_engine::EngineTelemetry {
        use std::collections::HashMap;
        use std::time::{SystemTime, UNIX_EPOCH};

        let active = self.requests_active();
        let waiting = self.requests_waiting();

        // batch_occupancy: fraction of GPU KV blocks in use. Both
        // backends populate `kv_gpu_blocks_{free,total}` from their own
        // pool views — `kv_gpu_utilization()` is the canonical fraction.
        let batch_occupancy = self.kv_gpu_utilization().clamp(0.0, 1.0);
        let model_arch = self
            .inner
            .model_arch
            .lock()
            .ok()
            .and_then(|summary| summary.clone());

        // KV-tier hit rates by canonical label. T0 = active GPU pool,
        // T1 = host pinned, T2 = local disk, T3 = remote. Backends
        // without a particular tier leave its counter at zero; we
        // still emit "T0" because every backend has an active pool.
        let mut kv_tier_hit_rates: HashMap<String, f64> = HashMap::new();
        let staged_total = self.tier_fetch_staged_blocks_total();
        // T0 hit = prefix reuse (pages already resident in GPU pool).
        kv_tier_hit_rates.insert("T0".to_string(), self.prefix_hit_rate());
        if staged_total > 0 {
            let host = self.tier_fetch_staged_host_blocks_total();
            let disk = self.tier_fetch_staged_disk_blocks_total();
            let remote = self.tier_fetch_staged_remote_blocks_total();
            let denom = staged_total as f64;
            kv_tier_hit_rates.insert("T1".to_string(), host as f64 / denom);
            kv_tier_hit_rates.insert("T2".to_string(), disk as f64 / denom);
            kv_tier_hit_rates.insert("T3".to_string(), remote as f64 / denom);
        }

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // None until any draft tokens have been verified — the accumulator
        // starts at 0 and would otherwise emit a misleading 1.0 (the
        // optimistic-start convention used by `AcceptanceTracker::current_rate`,
        // which is per-request and resets, vs the global ppm gauge which is
        // monotonic across all requests).
        let spec_acceptance_rate =
            (self.spec_verified_tokens_total() > 0).then(|| self.spec_acceptance_rate());

        crate::server_engine::EngineTelemetry {
            ttft_us,
            itl_p50_us,
            itl_p99_us,
            queue_depth: u32::try_from(waiting).unwrap_or(u32::MAX),
            active_requests: u32::try_from(active).unwrap_or(u32::MAX),
            batch_occupancy,
            model_arch,
            kv_tier_hit_rates,
            spec_acceptance_rate,
            timestamp_ms,
        }
    }

    /// Render structured stats for `/v1/stats?format=json`.
    pub fn render_stats_json(&self) -> serde_json::Value {
        let latest = self
            .inner
            .latest_request_cache
            .lock()
            .ok()
            .map(|stats| stats.clone())
            .unwrap_or_default();
        let mut session_entries: Vec<(String, super::SessionCacheStats)> = self
            .inner
            .session_cache
            .lock()
            .ok()
            .map(|sessions| {
                sessions
                    .iter()
                    .map(|(session_id, stats)| (session_id.clone(), stats.clone()))
                    .collect()
            })
            .unwrap_or_default();
        session_entries.sort_by(|left, right| left.0.cmp(&right.0));
        let mut sessions = serde_json::Map::new();
        for (session_id, stats) in session_entries {
            sessions.insert(
                session_id,
                serde_json::json!({
                    "prefix_lookups_total": stats.prefix_lookups_total,
                    "prefix_hits_total": stats.prefix_hits_total,
                    "prefix_hit_rate": stats.prefix_hit_rate(),
                    "prefix_skip_rate": stats.prefix_skip_rate(),
                    "prefix_reused_tokens_total": stats.prefix_reused_tokens_total,
                    "prefix_lookup_prompt_tokens_total": stats.prefix_lookup_prompt_tokens_total,
                    "session_affinity_hit": stats.session_affinity_hit,
                    "session_affinity_miss": stats.session_affinity_miss,
                    "matched_prefix_tokens_total": stats.matched_prefix_tokens_total,
                    "resume_prefill_tokens_total": stats.resume_prefill_tokens_total,
                    "matched_prefix_tokens": stats.last_matched_prefix_tokens,
                    "resume_prefill_tokens": stats.last_resume_prefill_tokens,
                }),
            );
        }

        // M1 unified engine telemetry: project the rolling counters
        // through `snapshot_engine_telemetry()` and emit each field
        // under an `engine_*` flat-key prefix. New keys only — every
        // legacy field above stays exactly as before so bench scripts
        // and dashboards do not break.
        let telemetry = self.snapshot_engine_telemetry();
        let engine_kv_tier_hit_rates = serde_json::Value::Object(
            telemetry
                .kv_tier_hit_rates
                .iter()
                .map(|(tier, rate)| (tier.clone(), serde_json::json!(rate)))
                .collect(),
        );

        serde_json::json!({
            "requests": self.requests_total(),
            "active": self.requests_active(),
            "waiting": self.requests_waiting(),
            "tokens_out": self.tokens_generated_total(),
            "kv_util": self.kv_gpu_utilization(),
            "prefix_hit_rate": self.prefix_hit_rate(),
            "prefix_skip_rate": self.prefix_skip_rate(),
            "session_affinity_hit": self.session_affinity_hit_total(),
            "session_affinity_miss": self.session_affinity_miss_total(),
            "session_slot_pressure_evictions_hard": self.session_slot_pressure_evictions_hard(),
            "matched_prefix_tokens": self.matched_prefix_tokens(),
            "resume_prefill_tokens": self.resume_prefill_tokens(),
            "last_request": {
                "session_id": latest.session_id,
                "prefix_hit_rate": latest.prefix_hit_rate(),
                "prefix_skip_rate": latest.prefix_skip_rate(),
                "prompt_tokens": latest.prompt_tokens,
                "matched_prefix_tokens": latest.matched_prefix_tokens,
                "resume_prefill_tokens": latest.resume_prefill_tokens,
            },
            "sessions": sessions,
            "engine_ttft_us": telemetry.ttft_us,
            "engine_itl_p50_us": telemetry.itl_p50_us,
            "engine_itl_p99_us": telemetry.itl_p99_us,
            "engine_queue_depth": telemetry.queue_depth,
            "engine_active_requests": telemetry.active_requests,
            "engine_batch_occupancy": telemetry.batch_occupancy,
            "engine_model_arch": telemetry.model_arch,
            "engine_kv_tier_hit_rates": engine_kv_tier_hit_rates,
            "engine_spec_acceptance_rate": telemetry.spec_acceptance_rate,
            "engine_timestamp_ms": telemetry.timestamp_ms,
        })
    }

    /// Render a simple human-readable summary (for `/v1/stats` or logging).
    pub fn render_summary(&self) -> String {
        // Take the histogram lock once, extract every percentile we need
        // (both the human-readable `…ms` strings and the engine-level
        // µs values that feed `snapshot_engine_telemetry_with`), then
        // drop the guard before doing anything else. `std::sync::Mutex`
        // is non-reentrant, so the engine-telemetry projection further
        // down MUST run with this guard released — see
        // `snapshot_engine_telemetry_with`.
        let (
            ttft_p50,
            queue_p50,
            active_ttft_p50,
            ttft_p99,
            tpot_p50,
            step_p50,
            spec_step_latency_count,
            service_p50,
            engine_ttft_us,
            engine_itl_p50_us,
            engine_itl_p99_us,
        ) = {
            let histograms = self.inner.histograms.lock().ok();
            let fmt_ms = |v: Option<f64>| -> String {
                v.map_or_else(|| "—".to_string(), |val| format!("{:.1}ms", val * 1000.0))
            };
            let ttft_p50_secs = histograms.as_ref().and_then(|h| h.ttft.percentile(0.50));
            let ttft_p99_secs = histograms.as_ref().and_then(|h| h.ttft.percentile(0.99));
            let tpot_p50_secs = histograms.as_ref().and_then(|h| h.tpot.percentile(0.50));
            let tpot_p99_secs = histograms.as_ref().and_then(|h| h.tpot.percentile(0.99));
            let queue_p50_secs = histograms
                .as_ref()
                .and_then(|h| h.queue_wait.percentile(0.50));
            let active_ttft_p50_secs = histograms
                .as_ref()
                .and_then(|h| h.active_ttft.percentile(0.50));
            let step_p50_secs = histograms
                .as_ref()
                .and_then(|h| h.scheduler_step.percentile(0.50));
            let service_p50_secs = histograms.as_ref().and_then(|h| h.service.percentile(0.50));
            let spec_step_latency_count = histograms
                .as_ref()
                .map_or(0, |h| h.spec_step_latency_us.count());
            (
                fmt_ms(ttft_p50_secs),
                fmt_ms(queue_p50_secs),
                fmt_ms(active_ttft_p50_secs),
                fmt_ms(ttft_p99_secs),
                fmt_ms(tpot_p50_secs),
                fmt_ms(step_p50_secs),
                spec_step_latency_count,
                fmt_ms(service_p50_secs),
                ttft_p50_secs.map(|s| s * 1_000_000.0),
                tpot_p50_secs.map(|s| s * 1_000_000.0),
                tpot_p99_secs.map(|s| s * 1_000_000.0),
            )
            // `histograms` MutexGuard dropped here.
        };
        let active_mb =
            self.inner.memory_active_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let peak_mb =
            self.inner.memory_peak_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let cache_mb =
            self.inner.memory_cache_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let phase_suffix = self.scheduler_step_phase_us().map_or_else(
            || " step_phase_us=unavailable".to_string(),
            |(admission_us, prefill_us, decode_us, emit_us, total_us)| {
                let cleanup_suffix = self.scheduler_loop_phase_us().map_or_else(
                    String::new,
                    |(cleanup_us, loop_total_us)| {
                        format!(",cleanup:{cleanup_us},loop_total:{loop_total_us}")
                    },
                );
                format!(
                    " step_phase_us=adm:{},prefill:{},decode:{},emit:{},total:{}{}",
                    admission_us, prefill_us, decode_us, emit_us, total_us, cleanup_suffix
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
        let (plan_idle, plan_decode, plan_prefill, plan_split, plan_mixed) =
            self.scheduler_plan_totals();
        let plan_suffix = format!(
            " plan_label=idle:{plan_idle},decode:{plan_decode},prefill:{plan_prefill},split:{plan_split},mixed:{plan_mixed}"
        );
        let spec_suffix = format!(
            " spec=draft:{},verified:{},accepted:{},empty_sparse_views:{},accept_rate:{:.1}%,step_latency_count:{}",
            self.spec_draft_tokens_total(),
            self.spec_verified_tokens_total(),
            self.spec_accepted_tokens_total(),
            self.spec_sparse_view_empty_total(),
            self.spec_acceptance_rate() * 100.0,
            spec_step_latency_count,
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
        let agent_cache_suffix = format!(
            " prefix_request_hit_rate={:.1}% prefix_request_skip_rate={:.1}% session_affinity_hit={} session_affinity_miss={} session_slot_pressure_evictions_hard={} matched_prefix_tokens={} resume_prefill_tokens={}",
            self.prefix_request_hit_rate() * 100.0,
            self.prefix_request_skip_rate() * 100.0,
            self.session_affinity_hit_total(),
            self.session_affinity_miss_total(),
            self.session_slot_pressure_evictions_hard(),
            self.matched_prefix_tokens(),
            self.resume_prefill_tokens(),
        );

        // M1 unified engine telemetry tail. Keys are flat key=value
        // pairs the bench script's regex parser already understands;
        // missing percentiles render as `na` (and the parser maps that
        // to `None`). Tier hit rates emit as `engine_kv_tier_hit_T0=…`.
        // Use the no-lock helper because we already extracted the
        // histogram percentiles above (and even though we dropped the
        // guard, going through `snapshot_engine_telemetry()` would
        // re-acquire it for no reason).
        let telemetry = self.snapshot_engine_telemetry_with(
            engine_ttft_us,
            engine_itl_p50_us,
            engine_itl_p99_us,
        );
        let fmt_opt = |v: Option<f64>| -> String {
            v.map_or_else(|| "na".to_string(), |val| format!("{val:.1}"))
        };
        let mut tier_entries: Vec<(String, f64)> = telemetry
            .kv_tier_hit_rates
            .iter()
            .map(|(t, r)| (t.clone(), *r))
            .collect();
        tier_entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut engine_tier_suffix = String::new();
        for (tier, rate) in &tier_entries {
            let _ = write!(engine_tier_suffix, " engine_kv_tier_hit_{tier}={rate:.4}");
        }
        let engine_telemetry_suffix = format!(
            " engine_ttft_us={} engine_itl_p50_us={} engine_itl_p99_us={} engine_queue_depth={} engine_active_requests={} engine_batch_occupancy={:.4} engine_timestamp_ms={}{}",
            fmt_opt(telemetry.ttft_us),
            fmt_opt(telemetry.itl_p50_us),
            fmt_opt(telemetry.itl_p99_us),
            telemetry.queue_depth,
            telemetry.active_requests,
            telemetry.batch_occupancy,
            telemetry.timestamp_ms,
            engine_tier_suffix,
        );

        format!(
            "requests={} active={} waiting={} scheduled={} decode_rows={} prefill_rows={} running_batch={} prefill_queue={} batch_width={} decode_tokens={} prefill_tokens={} tokens_out={} step_last={:.1}ms step_p50={}{}{}{} tier_fetch_wait={:.1}ms tier_store_wait={:.1}ms kv_util={:.1}% prefix_hit_rate={:.1}% active_mem={:.1}MB peak_mem={:.1}MB cache_mem={:.1}MB queue_p50={} active_ttft_p50={} ttft_p50={} ttft_p99={} service_p50={} tpot_p50={}{}{}{}{}{}{}",
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
            plan_suffix,
            spec_suffix,
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
            agent_cache_suffix,
            coordinator_suffix,
            engine_telemetry_suffix,
        )
    }
}
