//! Prometheus text exposition for `GET /metrics`.
//!
//! Renders the same [`CounterSnapshot`] the engine loop republishes each tick
//! (and `/v1/stats` serves as JSON) in the Prometheus text format. Host-side
//! and backend-neutral: no engine or executor coupling beyond the snapshot
//! struct, and nothing here touches the request hot path — the handler reads
//! the latest published snapshot exactly like the stats route.

use crate::execution::CounterSnapshot;

/// Content type for the Prometheus text exposition format.
pub(crate) const PROMETHEUS_CONTENT_TYPE: &str = "text/plain; version=0.0.4";

/// Render `counters` as Prometheus text exposition, labelled with the served
/// model id (the vLLM-style `model_name` label, so multi-instance scrapes can
/// tell deployments apart).
pub(crate) fn render_prometheus(counters: &CounterSnapshot, model: &str) -> String {
    let labels = format!("{{model_name=\"{}\"}}", escape_label_value(model));
    let prefix = &counters.prefix_cache;
    let kv_system = &counters.kv_system;
    let mut out = String::with_capacity(4096);
    let mut push = |name: &str, kind: &str, help: &str, value: u64| {
        out.push_str("# HELP arle_");
        out.push_str(name);
        out.push(' ');
        out.push_str(help);
        out.push_str("\n# TYPE arle_");
        out.push_str(name);
        out.push(' ');
        out.push_str(kind);
        out.push_str("\narle_");
        out.push_str(name);
        out.push_str(&labels);
        out.push(' ');
        out.push_str(&value.to_string());
        out.push('\n');
    };

    push(
        "active_requests",
        "gauge",
        "Requests currently holding a scheduler slot.",
        counters.active_requests as u64,
    );
    push(
        "queue_depth",
        "gauge",
        "Requests waiting for admission.",
        counters.queue_depth as u64,
    );
    push(
        "kv_free_pages",
        "gauge",
        "Free physical KV pages in the host pool.",
        counters.kv_free_pages as u64,
    );
    push(
        "prefix_cache_cached_pages",
        "gauge",
        "KV pages currently retained by the radix prefix cache.",
        prefix.cached_pages as u64,
    );
    push(
        "prefix_cache_lookups_total",
        "counter",
        "Requests admitted while prefix caching was enabled.",
        prefix.lookups,
    );
    push(
        "prefix_cache_hits_total",
        "counter",
        "Lookups that restored at least one reusable prefix token.",
        prefix.hits,
    );
    push(
        "prefix_cache_hit_tokens_total",
        "counter",
        "Prompt tokens skipped via attached prefix pages.",
        prefix.hit_tokens,
    );
    push(
        "prefix_cache_hit_pages_total",
        "counter",
        "Page coverage of prompt tokens skipped by prefix restore.",
        prefix.hit_pages,
    );
    push(
        "prefix_cache_published_pages_total",
        "counter",
        "Prompt pages newly retained by the radix cache.",
        prefix.published_pages,
    );
    push(
        "engine_steps_total",
        "counter",
        "Executor steps whose output was applied.",
        counters.throughput.steps,
    );
    push(
        "prefill_tokens_total",
        "counter",
        "Prompt tokens advanced through chunked prefill.",
        counters.throughput.prefill_tokens,
    );
    push(
        "generated_tokens_total",
        "counter",
        "Tokens committed to requests (final prefill chunk + decode).",
        counters.throughput.generated_tokens,
    );
    push(
        "requests_completed_total",
        "counter",
        "Requests finished after holding a scheduler slot.",
        counters.throughput.requests_completed,
    );
    push(
        "requests_succeeded_total",
        "counter",
        "Requests that completed successfully (non-abort).",
        counters.throughput.requests_succeeded,
    );
    push(
        "requests_failed_total",
        "counter",
        "Requests that were aborted.",
        counters.throughput.requests_failed,
    );
    push(
        "ttft_micros_total",
        "counter",
        "Sum of time-to-first-token across all requests.",
        counters.throughput.ttft_micros_total,
    );
    push(
        "ttft_count",
        "counter",
        "Number of requests that produced a first token.",
        counters.throughput.ttft_count,
    );
    push(
        "tpot_micros_total",
        "counter",
        "Sum of per-output-token latency across all requests.",
        counters.throughput.tpot_micros_total,
    );
    push(
        "tpot_count",
        "counter",
        "Number of inter-token intervals measured.",
        counters.throughput.tpot_count,
    );
    push(
        "e2e_latency_micros_total",
        "counter",
        "Sum of end-to-end request latency.",
        counters.throughput.e2e_micros_total,
    );
    push(
        "e2e_latency_count",
        "counter",
        "Number of completed requests for e2e latency.",
        counters.throughput.e2e_count,
    );
    push(
        "forward_busy_micros_total",
        "counter",
        "Cumulative GPU forward-busy microseconds.",
        counters.throughput.forward_busy_micros,
    );
    let spec = &counters.spec_decode;
    let accept_rate = (spec.accepted * 100).checked_div(spec.drafted).unwrap_or(0);
    push(
        "spec_accept_rate",
        "gauge",
        "Speculative decode token accept rate (%).",
        accept_rate,
    );
    push(
        "spec_chains_total",
        "counter",
        "Speculative decode draft chains attempted.",
        spec.chains,
    );
    push(
        "spec_drafted_total",
        "counter",
        "Tokens drafted by speculative decode.",
        spec.drafted,
    );
    push(
        "spec_accepted_total",
        "counter",
        "Tokens accepted by speculative decode verification.",
        spec.accepted,
    );
    push(
        "spec_rejected_total",
        "counter",
        "Tokens rejected by speculative decode verification.",
        spec.rejected,
    );
    push(
        "kv_tier_resident_blocks",
        "gauge",
        "Prefix blocks currently resident in the host KV tier.",
        counters.kv_tier.resident_blocks as u64,
    );
    push(
        "kv_tier_demoted_pages_total",
        "counter",
        "Prefix pages demoted to the host KV tier instead of dropped.",
        counters.kv_tier.demoted_pages,
    );
    push(
        "kv_tier_promoted_pages_total",
        "counter",
        "Demoted pages promoted back to device pages on a prefix hit.",
        counters.kv_tier.promoted_pages,
    );
    push(
        "kv_tier_promote_failures_total",
        "counter",
        "Tier promotions that failed (tail re-prefilled instead).",
        counters.kv_tier.promote_failures,
    );
    push(
        "kv_tier_demoted_slots_total",
        "counter",
        "Whole-slot images demoted on preemption (page-less model route).",
        counters.kv_tier.demoted_slots,
    );
    push(
        "kv_tier_promoted_slots_total",
        "counter",
        "Whole-slot images promoted back on re-admission (decode resumed).",
        counters.kv_tier.promoted_slots,
    );
    push(
        "kv_tier_slot_promote_failures_total",
        "counter",
        "Whole-slot promotions that failed (request recomputed).",
        counters.kv_tier.slot_promote_failures,
    );
    push(
        "kv_system_resident_pages",
        "gauge",
        "Prefix pages currently resident in the fast working pool.",
        kv_system.resident_pages as u64,
    );
    push(
        "kv_system_resident_evictable_pages",
        "gauge",
        "Resident prefix pages currently evictable.",
        kv_system.resident_evictable_pages as u64,
    );
    push(
        "kv_system_host_demoted_pages",
        "gauge",
        "KV pages currently demoted to host RAM.",
        kv_system.host_demoted_pages as u64,
    );
    push(
        "kv_system_host_demoted_pending_inflight",
        "gauge",
        "KV pages with an in-flight host-demoted transfer.",
        kv_system.host_demoted_pending_inflight as u64,
    );
    push(
        "kv_system_disk_pages",
        "gauge",
        "KV pages currently stored on disk.",
        kv_system.disk_pages as u64,
    );
    push(
        "kv_system_reuse_hit_resident_total",
        "counter",
        "Prefix blocks reused from resident pages.",
        kv_system.reuse_hit_resident,
    );
    push(
        "kv_system_reuse_hit_host_demoted_total",
        "counter",
        "Prefix blocks reused from host-demoted pages.",
        kv_system.reuse_hit_host_demoted,
    );
    push(
        "kv_system_reuse_hit_disk_total",
        "counter",
        "Prefix blocks reused from disk.",
        kv_system.reuse_hit_disk,
    );
    push(
        "kv_system_reuse_miss_total",
        "counter",
        "Prefix attach lookups that restored no token.",
        kv_system.reuse_miss,
    );
    push(
        "kv_system_demote_mset_count_total",
        "counter",
        "Synchronous page-tier mset batches.",
        kv_system.demote_mset_count,
    );
    push(
        "kv_system_demote_mset_copy_bytes_total",
        "counter",
        "Bytes copied by synchronous page-tier mset.",
        kv_system.demote_mset_copy_bytes,
    );
    push(
        "kv_system_demote_mset_copy_ms_total",
        "counter",
        "Milliseconds spent in synchronous page-tier mset.",
        kv_system.demote_mset_copy_ms,
    );
    push(
        "kv_system_promote_mget_count_total",
        "counter",
        "Synchronous page-tier mget batches.",
        kv_system.promote_mget_count,
    );
    push(
        "kv_system_promote_mget_copy_bytes_total",
        "counter",
        "Bytes copied by synchronous page-tier mget.",
        kv_system.promote_mget_copy_bytes,
    );
    push(
        "kv_system_promote_mget_copy_ms_total",
        "counter",
        "Milliseconds spent in synchronous page-tier mget.",
        kv_system.promote_mget_copy_ms,
    );
    push(
        "kv_system_fetch_wait_ms_total",
        "counter",
        "Milliseconds spent waiting for synchronous KV fetch.",
        kv_system.fetch_wait_ms,
    );
    push(
        "kv_system_fallback_recompute_total",
        "counter",
        "KV restore failures that fell back to recompute.",
        kv_system.fallback_recompute,
    );
    push(
        "kv_system_prefix_match_full_blocks_total",
        "counter",
        "Prefix blocks matched before restore-boundary clamp.",
        kv_system.prefix_match_full_blocks,
    );
    push(
        "kv_system_prefix_match_clamped_blocks_total",
        "counter",
        "Prefix blocks left after restore-boundary clamp.",
        kv_system.prefix_match_clamped_blocks,
    );
    push(
        "kv_tier_io_useful_read_bytes_total",
        "counter",
        "Payload bytes read from the KV disk tier.",
        kv_system.tier_io_useful_read_bytes,
    );
    push(
        "kv_tier_io_useful_write_bytes_total",
        "counter",
        "Payload bytes written to the KV disk tier.",
        kv_system.tier_io_useful_write_bytes,
    );
    push(
        "kv_tier_io_submitted_read_bytes_total",
        "counter",
        "Aligned bytes submitted for KV disk reads.",
        kv_system.tier_io_submitted_read_bytes,
    );
    push(
        "kv_tier_io_submitted_write_bytes_total",
        "counter",
        "Aligned bytes submitted for KV disk writes.",
        kv_system.tier_io_submitted_write_bytes,
    );
    push(
        "kv_tier_io_metadata_write_bytes_total",
        "counter",
        "Metadata bytes written by the KV disk tier.",
        kv_system.tier_io_metadata_write_bytes,
    );
    push(
        "kv_tier_io_failures_total",
        "counter",
        "Failed KV disk I/O operations.",
        kv_system.tier_io_failures,
    );
    push(
        "kv_tier_io_completion_wait_ns_total",
        "counter",
        "Nanoseconds waiting for KV disk I/O completions.",
        kv_system.tier_io_completion_wait_ns,
    );
    let mode = match kv_system.tier_io_mode {
        infer_seam::KvTierIoMode::Disabled => "disabled",
        infer_seam::KvTierIoMode::Mmap => "mmap",
        infer_seam::KvTierIoMode::Direct => "direct",
    };
    out.push_str("# HELP arle_kv_tier_io_mode_info Active KV disk I/O mode.\n");
    out.push_str("# TYPE arle_kv_tier_io_mode_info gauge\n");
    out.push_str("arle_kv_tier_io_mode_info{model_name=\"");
    out.push_str(&escape_label_value(model));
    out.push_str("\",io_mode=\"");
    out.push_str(mode);
    out.push_str("\"} 1\n");
    if let Some(ref gpu) = counters.gpu {
        for i in 0..gpu.device_count as usize {
            let d = &gpu.devices[i];
            for (suffix, value) in [
                ("util_pct", d.util_pct as u64),
                ("memory_used_mb", d.memory_used_mb as u64),
                ("memory_total_mb", d.memory_total_mb as u64),
                ("temp_c", d.temp_c as u64),
                ("power_w", d.power_w as u64),
            ] {
                out.push_str(&format!(
                    "arle_gpu_{suffix}{{model_name=\"{}\",gpu_index=\"{}\"}} {value}\n",
                    escape_label_value(model),
                    d.gpu_index
                ));
            }
        }
    }
    out
}

/// Escape a Prometheus label value: backslash, double quote, and newline.
fn escape_label_value(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            _ => escaped.push(c),
        }
    }
    escaped
}
