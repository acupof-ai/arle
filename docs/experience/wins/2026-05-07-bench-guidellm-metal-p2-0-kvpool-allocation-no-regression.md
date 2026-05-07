# Bench — M_e.1 P2.0 Qwen3.5 MetalKVPool allocation: no regression

## Goal

Regression-only check that wiring `--kv-pool` through `Qwen35StepDriver`
and constructing a `MetalKVPool` per request does not impact decode
performance. The pool is allocated but never read or written this
commit; this is the bench gate before P2.1 lands the dual-write.

## Hypothesis

Pool allocation is one-time per request (~26 MB BF16 zeros for the
4096-token c=4 cell) — should not measurably affect ITL because no
hot-path operation touches it. TTFT may absorb the alloc cost (~ms);
output tok/s should hold within noise.

## Params

- 4096-in / 256-out / `--concurrencies 4` / max-seconds=30
- Model: `models/Qwen3.5-0.8B-MLX-4bit`
- ARLE: `target/release/metal_serve --max-running-requests 4 --kv-pool`
- Same machine (M4 Pro 20-core), same guidellm 0.6.0 version, same
  prompt distribution as the matched-c c=4 cell from
  `2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md`

## Env

- ARLE commit pre-bench: `24b6ecd` rebased onto origin
- Build: `cargo build --release --no-default-features --features metal -p infer --bin metal_serve` (39.26 s)

## Results

| Metric | --kv-pool OFF (prior baseline) | --kv-pool ON (this run) | Δ |
|---|---:|---:|---:|
| TTFT p50 | 1.197 s | 1.165 s | **-3%** (noise) |
| TTFT p99 | 4.071 s | 3.994 s | -2% |
| ITL p50 | 19.34 ms | 18.90 ms | **-2%** (noise) |
| ITL p95 | 19.96 ms | 19.91 ms | parity |
| Output tok/s | 147.5 | 154.4 | **+5%** (noise) |
| Conc p50 | 4 | 4 | parity |

Per `feedback_matched_ab_for_small_bench_effects.md`, single-sweep
deltas ≤ 10 % at the same binary are thermal noise. The pool path is
**structurally bit-identical** to the no-pool path (pool is constructed
but no hot-path operation reads or writes it), so the apparent +5 %
output tok/s sits inside per-run variance and not a real win.

Smoke evidence the pool actually allocates — server log line:

```
INFO infer::backend::metal::kv_pool: kv_pool.rs:391
MetalKVPool: 9 max tokens, 6 layers (2 kv_heads x 256 head_dim, kv_dim=512)
```

(9 tokens because the warmup prompt is short; real requests size to
`total_tokens_needed`.)

## Problems

None. Pool construction adds < 1 ms of per-request setup that's
absorbed in TTFT alongside the existing tokenization + scheduler-admit
cost. No allocator-pressure fall-out observed.

## Learnings

- The `Qwen35StepDriver` lifecycle accommodates a per-request
  `MetalKVPool` mirror of `Qwen3StepDriver` cleanly.
- DFlash is correctly excluded (`effective_kv_pool = use_kv_pool &&
  dflash_runtime.is_none()`); confirmed by the unchanged DFlash
  warmup behavior.
- The bench-gate obligation for P2.0 (`feedback_bench_every_change.md`)
  is satisfied by this entry. P2.1 dual-write needs its own
  before/after gate when the pool actually carries hot-path data.

## Reproduce

```bash
cargo build --release --no-default-features --features metal -p infer --bin metal_serve

RUST_LOG=warn target/release/metal_serve \
  --model-path models/Qwen3.5-0.8B-MLX-4bit \
  --port 8000 --bind 127.0.0.1 \
  --max-running-requests 4 --kv-pool

PATH=$HOME/.local/bin:$PATH ./scripts/bench_guidellm.sh metal-c4-kvpool \
  --concurrencies 4 --max-seconds 30 \
  --data 'prompt_tokens=4096,output_tokens=256' \
  --target http://localhost:8000 \
  --model Qwen3.5-0.8B-MLX-4bit \
  --processor /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit
```

Raw artefact: `bench-output/2026-05-07-metal-c4-kvpool/`.

## Cross-references

- Plan: [`docs/plans/M_e1-metal-paged-kv-hot-path.md`](../../plans/M_e1-metal-paged-kv-hot-path.md) §3 P2.0
- Master analysis: [`docs/projects/2026-05-07-metal-optimization-master-analysis.md`](../../projects/2026-05-07-metal-optimization-master-analysis.md) §5 P2.0
- Baseline matched-c c=4 (no flag): [`2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md`](2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md)
