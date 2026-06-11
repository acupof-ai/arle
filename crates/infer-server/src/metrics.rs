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
    let mut out = String::with_capacity(1536);
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
        "Lookups that attached at least one reusable prefix page.",
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
        "Prefix pages attached to admitted requests.",
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

#[cfg(test)]
mod tests {
    use infer_core::{KvTierStats, PrefixCacheStats, ThroughputStats};

    use super::*;

    const METRIC_COUNT: usize = 17;

    #[test]
    fn renders_help_type_and_labelled_samples() {
        let counters = CounterSnapshot {
            active_requests: 2,
            queue_depth: 1,
            kv_free_pages: 240,
            prefix_cache: PrefixCacheStats {
                lookups: 10,
                hits: 7,
                hit_tokens: 448,
                hit_pages: 28,
                published_pages: 31,
                cached_pages: 19,
            },
            throughput: ThroughputStats {
                steps: 90,
                prefill_tokens: 1200,
                generated_tokens: 256,
                requests_completed: 5,
            },
            kv_tier: KvTierStats {
                demoted_pages: 6,
                promoted_pages: 4,
                promote_failures: 1,
                resident_blocks: 2,
            },
        };
        let body = render_prometheus(&counters, "qwen3-dense");

        assert!(body.contains("# HELP arle_active_requests "));
        assert!(body.contains("# TYPE arle_active_requests gauge\n"));
        assert!(body.contains("arle_active_requests{model_name=\"qwen3-dense\"} 2\n"));
        assert!(body.contains("# TYPE arle_prefix_cache_lookups_total counter\n"));
        assert!(body.contains("arle_prefix_cache_lookups_total{model_name=\"qwen3-dense\"} 10\n"));
        assert!(body.contains("arle_prefix_cache_hits_total{model_name=\"qwen3-dense\"} 7\n"));
        assert!(
            body.contains("arle_prefix_cache_hit_tokens_total{model_name=\"qwen3-dense\"} 448\n")
        );
        assert!(body.contains("arle_prefix_cache_cached_pages{model_name=\"qwen3-dense\"} 19\n"));
        assert!(body.contains("arle_kv_free_pages{model_name=\"qwen3-dense\"} 240\n"));
        assert!(body.contains("# TYPE arle_engine_steps_total counter\n"));
        assert!(body.contains("arle_engine_steps_total{model_name=\"qwen3-dense\"} 90\n"));
        assert!(body.contains("arle_prefill_tokens_total{model_name=\"qwen3-dense\"} 1200\n"));
        assert!(body.contains("arle_generated_tokens_total{model_name=\"qwen3-dense\"} 256\n"));
        assert!(body.contains("arle_requests_completed_total{model_name=\"qwen3-dense\"} 5\n"));
        assert!(body.contains("arle_kv_tier_resident_blocks{model_name=\"qwen3-dense\"} 2\n"));
        assert!(body.contains("arle_kv_tier_demoted_pages_total{model_name=\"qwen3-dense\"} 6\n"));
        assert!(body.contains("arle_kv_tier_promoted_pages_total{model_name=\"qwen3-dense\"} 4\n"));
        assert!(
            body.contains("arle_kv_tier_promote_failures_total{model_name=\"qwen3-dense\"} 1\n")
        );

        // Every sample line carries the HELP/TYPE pair exactly once.
        assert_eq!(body.matches("# HELP ").count(), METRIC_COUNT);
        assert_eq!(body.matches("# TYPE ").count(), METRIC_COUNT);
    }

    #[test]
    fn zero_snapshot_renders_all_metrics() {
        let body = render_prometheus(&CounterSnapshot::default(), "m");
        assert_eq!(body.matches("# TYPE ").count(), METRIC_COUNT);
        assert!(body.contains("arle_queue_depth{model_name=\"m\"} 0\n"));
        assert!(body.contains("arle_generated_tokens_total{model_name=\"m\"} 0\n"));
    }

    #[test]
    fn label_value_is_escaped() {
        let body = render_prometheus(&CounterSnapshot::default(), "a\"b\\c\nd");
        assert!(body.contains("model_name=\"a\\\"b\\\\c\\nd\""));
    }
}
