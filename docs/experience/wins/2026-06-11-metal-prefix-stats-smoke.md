# Metal Prefix Stats Smoke

## Goal

Answer the multi-turn question with runtime evidence: what is the prompt KV
prefix-cache hit rate, and what is the SSD KV recall rate, on the rewrite Metal
serve path.

## Hypothesis

The rewrite Metal path should show cross-request prompt-prefix reuse through the
shared scheduler/radix cache. SSD KV recall should be unavailable rather than a
zero-rate metric, because the rewrite serve path has no active SSD recall tier.

## Params

- Command: `target/release/arle serve --backend metal --model-path mlx-community/Qwen3.5-0.8B-MLX-4bit --port 8131 --low-impact`
- Reason for small-model opt-out: counter smoke for service/scheduler/KV
  observability; not a Qwen3.6 performance run.
- Requests: 4 serial `POST /v1/completions` calls with the same long leading
  prefix, different `Turn N` suffixes, `max_tokens=1`, `temperature=0`,
  `ignore_eos=true`.
- Prompt shape: 103 prompt tokens, 1 generated token per request.

## Env

- Host: local Apple Silicon / Metal.
- Binary: local `target/release/arle`, built with
  `cargo build --release --no-default-features --features metal,no-cuda`.
- Scheduler budget: `--low-impact` (`num_slots=1`, `total_pages=1024`,
  `chunked_prefill_size=32`).

## Results

Initial `/v1/stats`:

```json
{
  "scheduler": {"active_requests": 0, "queue_depth": 0, "kv_free_pages": 1024},
  "prefix_cache": {
    "lookups": 0,
    "hits": 0,
    "hit_rate": null,
    "hit_tokens": 0,
    "hit_pages": 0,
    "published_pages": 0,
    "cached_pages": 0
  },
  "ssd_recall": {"available": false, "lookups": 0, "hits": 0, "recall_rate": null}
}
```

Final `/v1/stats` after 4 requests:

```json
{
  "scheduler": {"active_requests": 0, "queue_depth": 0, "kv_free_pages": 1015},
  "prefix_cache": {
    "lookups": 4,
    "hits": 3,
    "hit_rate": 0.75,
    "hit_tokens": 192,
    "hit_pages": 12,
    "published_pages": 9,
    "cached_pages": 9
  },
  "ssd_recall": {
    "available": false,
    "lookups": 0,
    "hits": 0,
    "recall_rate": null,
    "not_available_reason": "ssd kv recall is not implemented in the rewrite serve path"
  }
}
```

Request-level prefix KV hit rate was 75%: one cold miss, then three prefix hits.
Each hit attached 4 pages / 64 tokens. SSD recall rate is not applicable on this
path; reporting it as 0% would be misleading because there is no SSD recall
lookup path in the rewrite server.

## Verification

- `cargo test -p infer-core --release`
- `cargo test -p infer-server --release`
- `cargo check -p cli --release --no-default-features --features metal,no-cuda`
- `cargo test -p cli --release --no-default-features --features cpu,no-cuda serve::tests -- --nocapture`
- `cargo build --release --no-default-features --features metal,no-cuda`
- Local Metal smoke above; server exited cleanly after `SIGINT`.

## Problems

Global `cargo fmt --check` is blocked by pre-existing formatting drift in
unrelated CUDA/Vulkan files. The touched diff passes `git diff --check`.

## Learnings

Keep SSD recall explicit as unavailable until a real SSD tier is wired into the
rewrite path. Prefix hit-rate counters belong in the shared scheduler/core layer,
not the Metal backend, so CUDA and Metal expose the same service semantics.
