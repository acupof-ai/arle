# A3 Tool Resume Gate Miss - agent-load benchmark, agent-w4-tool-resume, arle-cuda, 2026-05-02

## Goal

- Diagnosis: verify the A3 session-resume prefix lookup path against the W4-H1
  entrance gate and identify the root cause if `avoided-prefill` does not reach
  the required `>= 90%`.

## Hypothesis

- A3 should let a same-session W4 resume attach the warmup KV prefix and prefill
  only the tool-output suffix, moving `matched_prefix_tokens` from 32 to roughly
  8k and `avoided-prefill` from 0.35% to at least 90%.

## Command

Build and verification:

```bash
CUDA_HOME=/usr/local/cuda \
CARGO_HOME=/tmp/cargo-home-local \
PEGAINFER_CUDA_SM=89 \
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
INFER_TILELANG_PYTHON=/usr/bin/python3 \
cargo build --release -p infer --features cuda

cargo fmt -p infer
git diff --check

CUDA_HOME=/usr/local/cuda \
CARGO_HOME=/tmp/cargo-home-local \
PEGAINFER_CUDA_SM=89 \
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
INFER_TILELANG_PYTHON=/usr/bin/python3 \
cargo test -p infer --release --no-default-features --features no-cuda --lib

cargo check -p infer --no-default-features --features no-cuda
cargo check -p infer --no-default-features --features cuda,no-cuda
```

Canonical W4 server:

```bash
CUDA_HOME=/usr/local/cuda \
CARGO_HOME=/tmp/cargo-home-local \
PEGAINFER_CUDA_SM=89 \
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig \
INFER_TILELANG_PYTHON=/usr/bin/python3 \
./target/release/infer \
  --model-path models/default \
  --port 8000 \
  --num-slots 8 \
  --max-seq-len 12288 \
  --kv-cache-dtype fp8 \
  --mem-fraction-static 0.85
```

Canonical W4 client:

```bash
python3 scripts/bench_agent_trace.py \
  --workload agent-w4-tool-resume \
  --server http://127.0.0.1:8000 \
  --label a3-tool-resume \
  --num-concurrent 8 \
  --max-tokens 256 \
  --probe-stats \
  --out bench-output/2026-05-02-agent-load-a3-tool-resume/client_summary.json \
  --trace-out bench-output/2026-05-02-agent-load-a3-tool-resume/trace.jsonl
```

One-session diagnostic control:

```bash
sed -n '1p' \
  bench-output/2026-05-02-agent-load-a3-tool-resume/trace.jsonl \
  > bench-output/2026-05-02-agent-load-a3-tool-resume/trace-one-session.jsonl

python3 scripts/bench_agent_trace.py \
  --workload trace \
  --trace bench-output/2026-05-02-agent-load-a3-tool-resume/trace-one-session.jsonl \
  --server http://127.0.0.1:8000 \
  --label a3-tool-resume-one-session \
  --num-concurrent 1 \
  --max-tokens 256 \
  --probe-stats \
  --out bench-output/2026-05-02-agent-load-a3-tool-resume/one_session_summary.json
```

## Environment

- **Workload:** `agent-w4-tool-resume`
- **Backend / engine:** `arle-cuda`
- **Model:** Qwen3-4B Instruct, `models/default -> models/Qwen3-4B`
- **Tokenizer / processor:** `models/default -> models/Qwen3-4B`
- **Hardware:** NVIDIA L4, 23,034 MiB, CUDA 12.8 runtime path
- **Commit:** `20b7f6e2` tree, with A3 code in `8aa5d7ab` and tests in
  `e8954b67`
- **Feature set:** `cargo build --release -p infer --features cuda`
- **KV dtype / cache mode:** FP8E4M3 paged KV, RadixCache enabled
- **Session / prefix flags:** OpenAI-compatible `session_id` on every request
- **Non-default flags / env vars:** `--num-slots 8`, `--max-seq-len 12288`,
  `--kv-cache-dtype fp8`, `--mem-fraction-static 0.85`,
  `PEGAINFER_CUDA_SM=89`, `INFER_TILELANG_PYTHON=/usr/bin/python3`

Server startup resolved:

```text
Scheduler ready: model=default, slots=8, max_seq_len=12288,
chunked_prefill_size=2048, max_num_batched_tokens=16384,
prefix_cache=on, short_prompt_bypass_tokens=256, host_pool=155.7MB
TokenKVPool: 137328 max tokens (8583 pages @ page_size=16), 11.0 GB,
format=FP8E4M3
```

## Workload Params

| field | value |
|---|---|
| seed | `20260502` |
| global concurrency | `8` |
| sessions | `128` |
| scored turns | `128` resume turns |
| prompt shape | base `8192 +/- 64` tokens, then tool output `256 +/- 16` tokens |
| max output tokens | warmup `64`, resume `256` |
| warm/cold mix | n/a |
| tool output tokens | `256 +/- 16` |
| run cap | full trace completion |

## Results - Headline

The harness does not record external elapsed time. `successful output tok/s`
uses the client timestamp (`2026-05-02T09:20:59.258174Z`) to
`client_summary.json` mtime (`2026-05-02T09:31:33.664735Z`), or 634.407s.

| metric | value |
|---|---:|
| successful scored turns | 76 / 128 |
| incomplete scored turns | 52 / 128 |
| successful output tok/s | 28.75 |
| TTFT p50 (ms) | 2430.6 |
| TTFT p99 (ms) | 13542.5 |
| ITL p50 (ms) | 79.4 |
| ITL p99 (ms) | 80.1 |
| E2E p50 (ms) | 38956.5 |
| E2E p99 (ms) | 48290.4 |

## Results - W4 Resume

| metric | canonical W4 | one-session control |
|---|---:|---:|
| resume TTFT p50 (ms) | 2430.6 | 151.4 |
| resume TTFT p99 (ms) | 13542.5 | 151.4 |
| resume E2E p50 (ms) | 38956.5 | 10300.5 |
| resume E2E p99 (ms) | 48290.4 | 10300.5 |
| matched prefix tokens | 32 | 8256 |
| resume prefill tokens | 8456 | 323 |
| avoided-prefill ratio | 0.38% | 96.2% |

## Results - Service-Side Cache / Scheduler

| metric | canonical W4 final |
|---|---:|
| peak active | 8 observed |
| peak waiting | 1 observed via sampled `/v1/stats` |
| peak prefill_queue | 2 observed via sampled `/v1/stats` |
| peak kv_util | 99.9% observed, 81.7% final |
| `prefix_hit_rate` | 95.3% final |
| `prefix_skip_rate` | 0.4% final |
| `prefix_request_hit_rate` | 100.0% final |
| `prefix_request_skip_rate` | 0.4% final |
| `session_affinity_hit` | 244 |
| `session_affinity_miss` | 12 |
| `tool_resume_count` | 128 scored resume requests, 76 successful |
| `tool_resume_prefill_tokens` | `resume_prefill_tokens=8456` latest request |
| `kv_fetch_q` | 0/16 final |
| `kv_fetch_waiters` | 0 final |
| `kv_store_q` | 0/16 final |
| `kv_store` | `sub:0,done:0,fail:0,rej:0` final |
| `kv_bp` | `fetch:0,store:0` final |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Problems

- **Gate miss:** canonical W4 stayed at 32 matched tokens and 0.38%
  avoided-prefill, far below the `>= 90%` direct submetric gate.
- **Root cause:** the A3 lookup works only if the warm prefix is still resident.
  The canonical W4 driver starts one task per session and shares an
  `asyncio.Semaphore(8)`. By the time a session's warmup finishes and its
  resume tries to acquire the semaphore, the remaining warmup tasks are already
  queued ahead of it. Sampled stats showed `requests=117` before resume phase,
  and `requests=142` after resumes began, with `matched_prefix_tokens=32`.
- **Capacity bound:** one warmup consumes about `8268 / 16 = 517` KV pages. The
  L4 run has 8583 total pages and a 90% retain hard cap, while 128 warm
  sessions would require roughly 66k pages. The run logs showed repeated prefix
  cache pressure fallback and emergency eviction. Host spill is only 155.7MB,
  so it cannot absorb the missing 8k prefixes for this shape.
- **Control result:** a clean one-session W4 run attached `8256/8579` tokens and
  prefilled only 323 tokens, proving the prompt shape and A3 session lookup are
  correct when the resident prefix survives.

## Learnings

- A server-side resume-prefix lookup is necessary but not sufficient for W4 at
  128 sessions: the benchmark must either deliver resume requests while the
  matching session is still resident, or the runtime needs an explicit
  retention/tiering policy sized for 128 long prefixes.
- `prefix_request_hit_rate=100%` is misleading for this gate when every request
  hits only the 32-token shared system prefix; `matched_prefix_tokens` and
  `resume_prefill_tokens` are the authoritative A3 counters.

## Delta Vs Baseline

- **Baseline:** [`2026-05-02-bench-agent-load-arle-w4-tool-resume.md`](../wins/2026-05-02-bench-agent-load-arle-w4-tool-resume.md)

| metric | baseline `e577d670` | now | delta |
|---|---:|---:|---:|
| successful output tok/s | 25.24 | 28.75 | +13.9% |
| successful scored turns | 78 / 128 | 76 / 128 | -2 turns |
| resume TTFT p99 (ms) | 26279.7 | 13542.5 | -48.5% |
| resume E2E p99 (ms) | 50476.7 | 48290.4 | -4.3% |
| matched prefix tokens | 32 | 32 | no change |
| resume prefill tokens | 8539 | 8456 | -83 tokens |
| avoided-prefill ratio | 0.35% aggregate / 0.37% latest | 0.38% latest | +0.01 to +0.03 pp |

## Artefacts

- Canonical client summary:
  `bench-output/2026-05-02-agent-load-a3-tool-resume/client_summary.json`
- Canonical generated trace:
  `bench-output/2026-05-02-agent-load-a3-tool-resume/trace.jsonl`
- One-session trace:
  `bench-output/2026-05-02-agent-load-a3-tool-resume/trace-one-session.jsonl`
- One-session client summary:
  `bench-output/2026-05-02-agent-load-a3-tool-resume/one_session_summary.json`

## Notes

- Code changed since baseline: `8aa5d7ab` added session-resume prefix lookup and
  admission preference; `e8954b67` added cache/runtime tests.
- `cargo check -p infer --no-default-features --features cuda,no-cuda` still
  reports the two pre-existing speculative CUDA unused-import warnings from
  `infer/src/speculative/cuda.rs`.
- A direct `cargo test -p infer --release --no-default-features --features
  cuda,no-cuda ...` was not used as the gate because that feature combo fails
  at link time on no-cuda stubs for test binaries; the no-GPU lib test suite
  passed instead.
