# Metal Qwen3.5 session handoff checkpoint — guidellm quick sweep, 2026-04-28

## Goal

Diagnosis/checkpoint: preserve the Metal Qwen3.5 compiled-session handoff fix
and the Metal-local logical plan scaffold, then verify that the local serving
path no longer trips `qwen35_session_begin requires an inactive session` under
a short concurrent GuideLLM load.

## Hypothesis

The C++ Qwen3.5 compiled model owns one process-local active session. Before
prefill or scalar decode begins for another request, draining any other active
Qwen3.5 C++ session should remove the session-begin failure seen in the
previous long-prompt sweep. The logical plan scaffold should be behavior-neutral:
legacy `decode` / `prefill` DTOs are still derived from the same scheduler
selection.

## Command

Server:

```bash
./scripts/start_metal_serve.sh models/Qwen3.5-0.8B-GGUF 8013 -- --warmup 0
```

Aborted canonical sweep that reproduced the failure before the decode-side
handoff:

```bash
scripts/bench_guidellm.sh metal-qwen35-session-handoff \
  --target http://127.0.0.1:8013 \
  --model models/Qwen3.5-0.8B-GGUF \
  --processor models/Qwen3.5-0.8B \
  --trace-interval-ms 1000
```

Short verification sweep after the decode-side handoff:

```bash
scripts/bench_guidellm.sh metal-qwen35-session-handoff-quick \
  --target http://127.0.0.1:8013 \
  --model models/Qwen3.5-0.8B-GGUF \
  --processor models/Qwen3.5-0.8B \
  --quick \
  --trace-interval-ms 1000
```

The quick wrapper resolved to:

```text
profile=concurrent
data=prompt_tokens=512,output_tokens=128
rate=1,2,4,8
max_seconds=60
random_seed=20260416
warmup=5
backend=openai_http request_format=/v1/completions
```

## Environment

- **Backend:** Metal
- **Model:** `models/Qwen3.5-0.8B-GGUF`
- **Processor:** `models/Qwen3.5-0.8B`
- **Hardware:** Apple Silicon local Mac, Metal/MLX backend
- **Commit:** `3757be1` plus the local Metal handoff/logical-plan diff; this
  was a checkpoint run from a dirty workspace, not a clean release baseline.
- **Feature set:** `cargo check -p infer --release --no-default-features --features metal,no-cuda`
- **Non-default flags:** `--warmup 0` on the server, `--quick` on the
  verification bench.

## Results — quick sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| conc1 | 129.9 | 136.5 | 5.05 | 5.27 | 166.11 | 1.291 |
| conc2 | 232.1 | 362.7 | 7.73 | 18.53 | 197.34 | 1.564 |
| conc4 | 339.6 | 503.4 | 15.59 | 22.91 | 217.50 | 1.727 |
| conc8 | 1788.9 | 2489.3 | 11.51 | 16.43 | 295.35 | 2.382 |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| samples | 383 ok / 0 failed |
| peak waiting | 7 |
| peak active | 4 |
| peak running_batch | 4 |
| peak prefill_queue | 1 |
| peak kv_util | 0.0% |
| prefix_hit_rate | 0.0% |
| prefix_skip_rate | 0.0% |
| active_mem after | 2149.3 MB |
| peak_mem after | 2309.1 MB |
| cache_mem after | 8716.6 MB |
| `metal_decode` after | `batch:13897/43237,scalar:10104,fallback:0,qwen35_packed:13897/43237` |

After snapshot:

```text
requests=416 active=4 waiting=4 scheduled=4 decode_rows=3 prefill_rows=1 running_batch=3 prefill_queue=1 batch_width=4 decode_tokens=3 prefill_tokens=509 tokens_out=53128 step_last=86.5ms step_p50=10.0ms tier_fetch_wait=0.0ms tier_store_wait=0.0ms kv_util=0.0% prefix_hit_rate=0.0% active_mem=2149.3MB peak_mem=2309.1MB cache_mem=8716.6MB queue_p50=75.0ms active_ttft_p50=200.0ms ttft_p50=300.0ms ttft_p99=5000.0ms service_p50=2000.0ms tpot_p50=15.0ms metal_decode=batch:13897/43237,scalar:10104,fallback:0,qwen35_packed:13897/43237 prefix_skip_rate=0.0%
```

## Results — request accounting

| rate | completed input | incomplete input | errored input | completed output | incomplete output | errored output |
|---|---:|---:|---:|---:|---:|---:|
| conc1 | 36936 | 0 | 0 | 9216 | 0 | 0 |
| conc2 | 44631 | 512 | 0 | 11136 | 126 | 0 |
| conc4 | 48735 | 2048 | 0 | 12160 | 131 | 0 |
| conc8 | 67716 | 3584 | 0 | 16896 | 375 | 0 |

## Problems

- The first canonical 4096-in / 256-out sweep was stopped after reproducing
  deeper decode-side session failures. Its trace showed peak waiting 256,
  peak active 2, peak running_batch 1, and no final after-stats snapshot.
- The quick sweep is not the canonical serving benchmark. It is a
  short-shape checkpoint to prove the session handoff stopped the immediate
  session-begin failure under concurrent load.
- Three server-side `stream consumer dropped` messages were observed near the
  end of the conc4 stage, but GuideLLM reported zero errored requests and the
  service trace had zero failed stats polls.
- The workspace was dirty because unrelated local files outside this Metal
  tranche were already modified. Treat these numbers as checkpoint evidence,
  not a clean baseline for SOTA comparison.

## Learnings

- The compiled Qwen3.5 session must be treated as runtime-owned shared state,
  even though Rust stores per-request `Qwen35CppState`. Any prefill or scalar
  decode path that can run while another request owns the C++ session needs an
  explicit drain/handoff point.
- `peak running_batch` and backend counters must be read together. The quick
  run reached `qwen35_packed` rows, but single-request and serving SOTA remain
  separate questions.
- The logical plan should stay Metal-local until it actually feeds
  runtime-owned batched state. Cross-backend scheduler abstractions would add
  surface area before the Metal lifecycle is stable.

## Delta vs baseline

- **Baseline:** [`2026-04-28-metal-qwen35-guidellm-sweep-no-batch.md`](../errors/2026-04-28-metal-qwen35-guidellm-sweep-no-batch.md)
- **Delta:** no matched numeric delta. The baseline was an aborted canonical
  4096-in sweep with `qwen35_packed:0/0`; this checkpoint is a completed
  512-in quick sweep with `qwen35_packed:13897/43237`.

| signal | prior aborted sweep | checkpoint quick sweep |
|---|---:|---:|
| GuideLLM completed | no | yes |
| stats samples failed | 0 | 0 |
| peak waiting | 256 | 7 |
| peak active | 1-2 | 4 |
| peak running_batch | 1 | 4 |
| `qwen35_packed` rows | 0 | 43237 |
| `qwen35_session_begin requires an inactive session` | yes | not observed |

## Artefacts

Raw bench-output directories were intentionally removed after the table and
trace summary were transcribed, per cleanup request:

- `bench-output/2026-04-28-metal-qwen35-session-handoff/`
- `bench-output/2026-04-28-metal-qwen35-session-handoff-quick/`

## Code checkpoint

This entry backs a small Metal-only checkpoint, not a completed serving
architecture:

- `infer/src/backend/metal/request_state.rs` exposes a Qwen3.5 C++ session
  drain hook.
- `infer/src/backend/metal/runtime.rs` drains other Qwen3.5 C++ sessions
  before prefill and scalar decode.
- `infer/src/backend/metal/plan.rs` and `scheduler.rs` preserve a Metal-local
  logical plan alongside the legacy `decode` / `prefill` step fields for later
  runtime-owned batched state work.
