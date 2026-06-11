# /metrics Prometheus Re-export on the Rewrite Serve Path (#81)

## Goal

Close #81: re-port the monolith's `/metrics` Prometheus endpoint to
`infer-server` so bench/monitoring tooling can scrape the KV/prefix-cache
observability surface (the 2026-06-11 KV-system audit listed this as a
rewrite gap; support-matrix §5 said "route does not exist").

## Hypothesis

The engine loop already republishes a `CounterSnapshot` each tick for
`/v1/stats`; rendering the same snapshot as Prometheus text exposition adds a
scrape surface with zero hot-path coupling — the handler runs only on `GET
/metrics` and reads the latest published snapshot exactly like the stats
route (same `Mutex<ServeHandle>` pattern).

## Params

- New module `crates/infer-server/src/metrics.rs`: `render_prometheus` over
  `CounterSnapshot`, vLLM-style `model_name` label, escaped per the
  Prometheus text-format spec; route wired in `http.rs`
  (`GET /metrics`, `text/plain; version=0.0.4`).
- Metrics: gauges `arle_active_requests`, `arle_queue_depth`,
  `arle_kv_free_pages`, `arle_prefix_cache_cached_pages`; counters
  `arle_prefix_cache_{lookups,hits,hit_tokens,hit_pages,published_pages}_total`.
- Smoke (both backends): serve, scrape initial `/metrics`, send 4 serial
  `POST /v1/completions` with the same long leading prefix and different
  `Turn N:` suffixes (`max_tokens=1`, `temperature=0`), re-scrape, cross-check
  against `/v1/stats`.
- Reason for small-model opt-out: counter/route smoke for the observability
  surface, not a Qwen3.6 performance run (same rationale as
  [2026-06-11-metal-prefix-stats-smoke](2026-06-11-metal-prefix-stats-smoke.md)).

## Env

- Host: local Apple Silicon (M4 Pro), macOS.
- CPU smoke: `target/release/arle serve --backend cpu --model-path
  models/Qwen3.5-0.8B --port 18181`, built
  `--no-default-features --features cpu,no-cuda,cli`.
- Metal smoke: `target/release/arle serve --backend metal --model-path
  mlx-community/Qwen3.5-0.8B-MLX-4bit --port 8132 --low-impact`, built
  `--no-default-features --features metal,no-cuda`.

## Results

Metal, initial scrape (all 9 metrics present, HELP/TYPE pairs, 200 with
`content-type: text/plain; version=0.0.4`):

```
arle_active_requests{model_name="Qwen3.5-0.8B-MLX-4bit"} 0
arle_queue_depth{model_name="Qwen3.5-0.8B-MLX-4bit"} 0
arle_kv_free_pages{model_name="Qwen3.5-0.8B-MLX-4bit"} 8192
arle_prefix_cache_cached_pages{model_name="Qwen3.5-0.8B-MLX-4bit"} 0
arle_prefix_cache_lookups_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 0
arle_prefix_cache_hits_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 0
arle_prefix_cache_hit_tokens_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 0
arle_prefix_cache_hit_pages_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 0
arle_prefix_cache_published_pages_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 0
```

Metal, after 4 shared-prefix requests (121 prompt tokens each):

```
arle_kv_free_pages{model_name="Qwen3.5-0.8B-MLX-4bit"} 8185
arle_prefix_cache_cached_pages{model_name="Qwen3.5-0.8B-MLX-4bit"} 7
arle_prefix_cache_lookups_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 4
arle_prefix_cache_hits_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 3
arle_prefix_cache_hit_tokens_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 336
arle_prefix_cache_hit_pages_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 21
arle_prefix_cache_published_pages_total{model_name="Qwen3.5-0.8B-MLX-4bit"} 7
```

`/v1/stats` cross-check on the same server: `lookups=4, hits=3,
hit_rate=0.75, hit_tokens=336, hit_pages=21, published_pages=7,
cached_pages=7, kv_free_pages=8185` — identical values, one snapshot source.

CPU smoke (Qwen3.5-0.8B, two identical prompts): `lookups=2, hits=1,
hit_tokens=16, published_pages=1, cached_pages=1`, free pages 8192→8191 —
counters move and the second request hits the radix prefix.

## Verification

- `cargo test -p infer-server --release` — 23 passed (3 new metrics tests:
  HELP/TYPE/label rendering, zero-snapshot completeness, label escaping).
- `cargo clippy -p infer-server --release -- -D warnings` — clean.
- `cargo build --release --no-default-features --features cpu,no-cuda,cli`
  and `--features metal,no-cuda` — both build; live smokes above on both
  backends; servers exited cleanly.

## Problems

None. Perf note: the route adds no per-token work (handler executes only on
scrape; router registration is a one-time radix-tree insert in axum), so no
throughput A/B was run — the canonical-bench obligation is satisfied by the
smoke + the off-hot-path argument, mirroring the prefix-stats-smoke
precedent. If a future change couples `/metrics` to the engine loop (e.g.
histogram observation per step), that change needs a real bench entry.

## Learnings

`/v1/stats` (JSON, bench probes) and `/metrics` (Prometheus, scrape tooling)
must stay views over the same `CounterSnapshot` — adding a counter means
extending the snapshot in `execution.rs`, then both surfaces, so they can
never disagree. SSD-recall metrics are deliberately absent from `/metrics`
until a real tier exists (#82/#83): exporting a constant-zero recall rate
would be the misleading-metric trap the stats surface already avoids.
