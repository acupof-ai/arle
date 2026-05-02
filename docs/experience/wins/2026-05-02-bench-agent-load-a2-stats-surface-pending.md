# A2 Stats Surface - agent-load benchmark, W3 stats, arle-cuda-l4, 2026-05-02

> Status: local L4 run complete after reading
> `memory/project_remote_cuda_box.md` and confirming the NVIDIA L4 is present
> locally.

## Goal

- Regression: verify the A2 `/metrics` and `/v1/stats?format=json` fields are
  observable under real ARLE CUDA traffic and usable by W3/W4 clients.

## Hypothesis

- The stats-only change should expose prefix hit/skip, session-affinity,
  matched-prefix, and resume-prefill counters without requiring A1/A3 stubs.

## Command

Server:

```bash
python3 -m pip install -e '.[tilelang]'
python3 -c "import tilelang; print(tilelang.__version__)"  # 0.1.9
CUDA_HOME=/usr/local/cuda \
CARGO_HOME=/tmp/cargo-home-local \
PEGAINFER_CUDA_SM=89 \
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
INFER_TILELANG_PYTHON=/usr/bin/python3 \
cargo run --release -p infer --no-default-features --features cuda -- \
  --model-path models/Qwen3-4B --port 8000 \
  --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 \
  --max-num-batched-tokens 16384 --max-prefill-tokens 16384 \
  --schedule-policy fcfs
```

Client:

```bash
PATH="$HOME/.local/bin:$PATH" GUIDELLM__MP_CONTEXT_TYPE=forkserver \
  scripts/bench_guidellm.sh a2-stats-surface-regression \
    --target http://localhost:8000 \
    --model Qwen/Qwen3-4B \
    --processor infer/models/Qwen3-4B

PATH="$HOME/.local/bin:$PATH" GUIDELLM__MP_CONTEXT_TYPE=forkserver \
  scripts/bench_guidellm.sh a2-stats-surface-regression \
    --target http://localhost:8000 \
    --model Qwen/Qwen3-4B \
    --processor infer/models/Qwen3-4B \
    --concurrencies 16 --max-seconds 60

python3 scripts/bench_agent_trace.py \
  --workload agent-w3-short-multiturn \
  --server http://localhost:8000 \
  --label a1-session-affinity-w3-tilelang-rerun \
  --out bench-output/2026-05-02-agent-load-a1-session-affinity-w3-tilelang-rerun/results.json \
  --trace-out bench-output/2026-05-02-agent-load-a1-session-affinity-w3-tilelang-rerun/trace.jsonl
```

Local verification:

```bash
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo test --release -p infer --no-default-features --features no-cuda
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo check -p infer --no-default-features --features cuda,no-cuda
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
  cargo test --release -p infer --no-default-features --features no-cuda
```

## Environment

- **Workload:** GuideLLM 4096-in/256-out c=16 sanity + W3 stats probe
- **Backend / engine:** arle-cuda
- **Model:** Qwen3-4B Instruct (`models/Qwen3-4B`, `eos_token_id=151645`)
- **Tokenizer / processor:** `infer/models/Qwen3-4B`
- **Hardware:** NVIDIA L4 24GB, driver 580.82.07, CUDA 13.0, SM 89
- **Commit:** A2 `e5dd3296`; local run included A1 commit `b1716819`
- **Feature set:** release CUDA binary, `--no-default-features --features cuda`
- **KV dtype / cache mode:** auto FP8E4M3 paged KV, prefix cache on
- **Session / prefix flags:** HTTP `session_id`, RadixCache metadata enabled
- **Python tools:** `guidellm 0.6.0`, `tilelang 0.1.9` installed via
  `pip install -e '.[tilelang]'`

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | GuideLLM c=16; W3 c=16 |
| sessions | W3: 64 warm + 64 cold |
| scored turns | W3: 256 warm + 64 cold |
| prompt shape | GuideLLM: 4096 in; W3: 1024 base + 64-token warm tails |
| max output tokens | GuideLLM 256; W3 64 |
| warm/cold mix | W3 80% warm / 20% cold |
| tool output tokens | n/a |
| run cap | GuideLLM 60s; W3 full trace completion |

## Results - Headline

| metric | value |
|---|---:|
| W3 successful scored turns | 320 / 320 |
| W3 incomplete scored turns | 0 |
| W3 scored tokens | 18837 |
| W3 summed scored wall (s) | 1450.71 |
| W3 TTFT p50 (ms) | 233.0 |
| W3 TTFT p99 (ms) | 5049.6 |
| W3 ITL p50 (ms) | 50.7 |
| W3 ITL p99 (ms) | 53.7 |
| GuideLLM c16 completed / incomplete | 1 / 15 |
| GuideLLM c16 completed-output tok/s | 4.26 |
| GuideLLM canonical run3 | stalled locally at high pressure; no JSON/CSV/HTML |

## Results - W3 Warm/Cold

| metric | warm | cold |
|---|---:|---:|
| scored turns | 256 | 64 |
| TTFT p50 (ms) | 226.1 | 637.2 |
| TTFT p99 (ms) | 676.3 | 7181.0 |

## Results - W4 Resume

| metric | value |
|---|---:|
| resume TTFT p50 (ms) | n/a |
| resume TTFT p99 (ms) | n/a |
| resume E2E p50 (ms) | n/a |
| resume E2E p99 (ms) | n/a |
| cold 8k TTFT p99 (ms) | n/a |
| matched prefix tokens | exposed; W3 final request value `1408` |
| avoided-prefill ratio | exposed; W3 final request `1 - 124 / 1532 = 91.9%` |

## Results - Service-Side Cache / Scheduler

| metric | value |
|---|---:|
| peak active | 16 |
| peak waiting | GuideLLM canonical run3 trace peak 497; W3 final 0 |
| peak prefill_queue | GuideLLM canonical run3/c16 trace peak 14 |
| peak kv_util | GuideLLM canonical run3 47.9%; c16 50.7%; W3 final 69.0% |
| `prefix_hit_rate` | W3 final `95.8%` |
| `prefix_skip_rate` | W3 final `57.9%` |
| `session_affinity_hit` | W3 final `368` |
| `session_affinity_miss` | W3 final `16` |
| `matched_prefix_tokens` | W3 final `1408` |
| `resume_prefill_tokens` | W3 final `124` |
| `tool_resume_count` | n/a |
| `tool_resume_prefill_tokens` | n/a |
| `kv_fetch_q` | `0/16` |
| `kv_fetch_waiters` | `0` |
| `kv_store_q` | `0/16` |
| `kv_store` | `sub:0,done:0,fail:0,rej:0` |
| `kv_bp` | `fetch:0,store:0` |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Four-Engine Comparison

| engine | commit/tag | output tok/s | TTFT p99 (ms) | E2E p99 (ms) | cache report | raw artefact |
|---|---|---:|---:|---:|---|---|
| ARLE | `b1716819` | see artefact | 5049.6 | see artefact | A2 fields populated | `bench-output/2026-05-02-agent-load-a1-session-affinity-w3-tilelang-rerun/results.json` |
| SGLang | pending | pending | pending | pending | pending | pending |
| vLLM | pending | pending | pending | pending | pending | pending |
| TensorRT-LLM | pending | pending | pending | pending | pending | pending |
| Mooncake | pending | pending | pending | pending | pending | pending |

Mission margin:

```text
Not claimed from A2. This entry validates the stats surface, not competitor margin.
```

## Problems

- The first canonical GuideLLM sweep was local on the L4 box, but stalled
  after the high-pressure sweep phase with `active=16`, `waiting=510`,
  `decode_rows=0`, and GPU util 0%. Raw trace kept at
  `bench-output/2026-05-02-a2-stats-surface-regression/`.
- The official TileLang-extra rerun of the same canonical GuideLLM command
  reproduced the local high-pressure stall with `active=16`, `waiting=496`,
  `running_batch=16`, `decode_rows=0`, and GPU util 0%. Raw trace kept at
  `bench-output/2026-05-02-a2-stats-surface-regression-run3/`.
- The bounded c=16 GuideLLM run completed the harness and wrote JSON/CSV/HTML,
  but only 1/16 streams finished inside the 60s cap. Use it as a regression
  smoke artefact, not a performance claim.
- `GET /health` returned 404 on this router; the benchmark wrapper validates
  readiness through `GET /v1/models`, which returned the served model id.
- W3 harness hard-codes model `"default"`; the server was restarted with
  `models/default -> Qwen3-4B` so model validation stayed strict without an A4
  code change.

## Learnings

- The W3/W4 client can fill cache validity rows from `/v1/stats?format=json`
  under real CUDA traffic.
- `resume_prefill_tokens` and `matched_prefix_tokens` are enough for W4's
  avoided-prefill numerator/denominator once the W4 trace lands.

## Delta Vs Baseline

- **Baseline:** first local A2 stats-surface run on this L4 after the GPU-box
  correction.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| W3 TTFT p99 | 4514.0 ms | 5049.6 ms | +11.9% |
| W3 warm TTFT p99 | 887.2 ms | 676.3 ms | -23.8% |
| GuideLLM canonical stall | active=16 waiting=510 | active=16 waiting=496 | reproduced |

## Artefacts

- GuideLLM stalled sweep: `bench-output/2026-05-02-a2-stats-surface-regression/`
- GuideLLM stalled sweep rerun with official TileLang extra:
  `bench-output/2026-05-02-a2-stats-surface-regression-run3/`
- GuideLLM bounded c16: `bench-output/2026-05-02-a2-stats-surface-regression-run2/`
- W3 raw turns / summary: `bench-output/2026-05-02-agent-load-a1-session-affinity-w3-tilelang-rerun/results.json`
- W3 generated trace: `bench-output/2026-05-02-agent-load-a1-session-affinity-w3-tilelang-rerun/trace.jsonl`
- Server logs:
  `bench-output/server-logs/2026-05-02T07-37-33-port8000-a2-a1-tilelang-cargo-run.log`,
  `bench-output/server-logs/2026-05-02T07-22-14-port8000-a1-a2-c16.log`,
  `bench-output/server-logs/2026-05-02T07-48-32-port8000-a1-w3-tilelang-cargo-run-default.log`

## Notes

- What changed in the code since baseline: added request/session cache stats
  rendering and scheduler prefix accounting for W3/W4 service-side rows.
- Suspected cause of any regression: stats recording only; no scheduler
  decision input was added by A2.
- Follow-ups: fix or bound the GuideLLM sweep/client backpressure mode before
  using canonical sweep output as a perf claim.
