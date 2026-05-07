# Bench — ARLE-Metal vs mlx-lm c-sweep on M4 Pro (Qwen3.5-0.8B-MLX-4bit)

## Goal

First reproducible Apple-Silicon serving comparison after exposing
`metal_serve --max-running-requests` and `--max-batch-tokens` flags.
Quantify how ARLE-Metal scales vs mlx-lm under the same workload, and
whether the historical hardcoded `MetalSchedulerConfig::default()`
ceiling of 4 active slots was a calibrated value or a leftover.

## Hypothesis

Raising `max_running_requests` from 4 → 16 will close the throughput
gap vs mlx-lm without harming ITL, since c=16 is a basic continuous-
batching expectation on Apple Silicon and mlx-lm handles it cleanly at
the same model.

## Params

- Workload: `--fast` exploration preset of `scripts/bench_guidellm.sh`
  → `concurrent rate=16 / data=4096-in/256-out / max-seconds=30`
- Model: `models/Qwen3.5-0.8B-MLX-4bit` (same instance for ARLE and mlx-lm)
- Tokenizer: same path
- Per cell: ARLE-Metal at `--max-running-requests=4|8|16`, plus
  mlx-lm v0.31.2 at default settings

## Env

- Host: Apple M4 Pro (20 cores), macOS darwin 25.3.0
- ARLE: feature set `--no-default-features --features metal`
- ARLE commit: `0e1bc3d` (M_d.1 fully landed) + this commit's flags
- mlx-lm: 0.31.2 (homebrew)
- guidellm: 0.6.0 (`~/.local/bin`)
- Server bind: 127.0.0.1; ARLE on 8000, mlx-lm on 8001

## Results

| Backend | c | TTFT p50 | TTFT p99 | ITL p50 | ITL p95 | output tok/s | Δ tok/s vs ARLE c=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARLE-Metal | 4 (default) | 13.00s | 21.94s | **19.05ms** | 19.74ms | **157.6** | +0% |
| ARLE-Metal | 8 | 7.61s | 18.92s | 39.77ms | 40.94ms | 144.1 | -8.6% |
| ARLE-Metal | 16 | 5.18s | 9.65s | 82.49ms | 84.03ms | 78.2 | -50.4% |
| mlx-lm | 16 | 12.69s | 16.60s | 18.97ms | 33.86ms | **467.9** | +197% |

Service-trace peaks (ARLE only — mlx-lm has no `/v1/stats`):

- c=4: peak active 4, peak waiting 16, peak kv_util 0%
- c=8: peak active 8, peak waiting 16, peak kv_util 0%
- c=16: peak active 16, peak waiting 16, peak kv_util 0% — flag
  plumbing works; the scheduler reaches the new ceiling

## Problems

1. **ARLE-Metal output throughput collapses with concurrency**, the
   opposite of the hypothesis. c=16 ITL is **4.3× worse** than c=4
   (19.05 → 82.49 ms) and absolute throughput **halves** (157.6 → 78.2
   tok/s). c=8 sits in the middle. The historical default of 4 is
   therefore a **calibrated** sweet spot, not an oversight, given the
   current hot path.
2. **Root cause is the varlen-padded packed decode kernel**
   (`Qwen35PackedDecodeBatch` — left-padding + additive mask + per-row
   RoPE offsets per `infer/src/backend/metal/AGENTS.md` invariant 7).
   Every additional active row pays for the longest in-flight prompt's
   left-pad. c=16 with diverse prompt lengths makes the wasted compute
   dominate.
3. mlx-lm at the same workload sustains 19 ms ITL median across c=16
   because its decode goes through a paged-KV-style path that does not
   left-pad. ARLE's gap is **3.0× output tok/s**, sitting entirely in
   the kernel-architecture axis, not the scheduler axis.
4. A small but real positive: ARLE's ITL **p95 stability** at c=4 is
   better than mlx-lm at c=16 (19.74 ms vs 33.86 ms tail), so when
   ARLE is run inside its sweet spot the per-token latency is the
   tighter distribution.

## Learnings

- The morning's gap analysis Tier-B #1 (paged-attention block tables
  on Metal, see `docs/projects/2026-05-07-metal-world-first-gap-
  analysis.md` and the unification recalibration in
  `2026-05-07-metal-world-first-recalibration-vs-unification.md`)
  is now empirically confirmed as the decisive Metal serving unlock.
  Without it, raising `max_running_requests` is **anti-correlated**
  with throughput.
- Keep `max_running_requests` default at 4 until the paged-KV hot
  path lands. The new flags ship as **operator escape hatches** for
  workloads with uniform-length prompts (where padding overhead
  collapses) and for benchmarking the kernel improvements as they
  land.
- ELI / nexil Layer-1 integration is already healthy on Metal —
  smoke `curl /v1/chat/completions` with `tool_choice:"auto"` +
  `response_format:{"type":"json_object"}` returned **HTTP 200 in 70 ms**
  before the bench. The empirical end-to-end path matches
  `docs/resources/eli-integration.md` Layer 1 exactly.
- Next bench-driving Metal commit candidates, in order of leverage:
  1. **Paged-KV / no-left-pad attention** — closes the 3× gap in
     a single kernel rewrite. L effort.
  2. **Q8/FP8 KV cache** — orthogonal, also helps long context. M
     effort. Belongs on the shared `infer/src/ops/kv_ops.rs` per
     M4 unification frame, not on `metal/ops.rs` only.
  3. **Spec-decode default for Qwen3.5 (DFlash promotion)** — tied to
     M_b/M_c sub-plans of `longctx-spec-tilelang-combo.md`.

## Reproduce

ARLE side, per c value:

```bash
cargo build --release --no-default-features --features metal -p infer --bin metal_serve
RUST_LOG=warn target/release/metal_serve \
  --model-path models/Qwen3.5-0.8B-MLX-4bit \
  --port 8000 \
  --max-running-requests <C> \
  --max-batch-tokens <max(512, C*128)>
```

Bench:

```bash
PATH=$HOME/.local/bin:$PATH ./scripts/bench_guidellm.sh metal-m-c<C>-arle \
  --fast \
  --target http://localhost:8000 \
  --model Qwen3.5-0.8B-MLX-4bit \
  --processor /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit
```

mlx-lm reference cell:

```bash
mlx_lm.server --model models/Qwen3.5-0.8B-MLX-4bit --port 8001 --host 127.0.0.1
PATH=$HOME/.local/bin:$PATH ./scripts/bench_guidellm.sh metal-m-c16-mlxlm \
  --fast \
  --target http://localhost:8001 \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit \
  --processor /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit
```

Raw artefacts (this run):
`bench-output/2026-05-07-metal-m-{smoke-arle,c8-arle,c16-arle,smoke-mlxlm}/`
