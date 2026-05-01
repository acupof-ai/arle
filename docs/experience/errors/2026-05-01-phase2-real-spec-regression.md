# Phase 2 real spec decode regression

## Context

P2.3/P2.4 exposed that the previous speculative decode path was only a
single-token canary. This run landed a real external-draft foundation:
persistent per-request draft state, K-token draft proposals, target paged-KV
rollback, greedy verifier accounting, bonus-token commit, and live spec
counters.

`Qwen/Qwen3-0.5B` is not available as a public Hugging Face model repo, so this
run used the nearest official Qwen3 small text model, `Qwen/Qwen3-0.6B`,
downloaded to `infer/models/Qwen3-0.6B`.

## Root Cause

The correctness-first verifier is still sequential on the target model. The
current target verifier runs the Qwen3-4B paged decode path once per verifier
position to preserve greedy bit identity with normal decode. That makes each
spec step pay:

- K draft forwards on Qwen3-0.6B
- K+1 target verifier decode forwards on Qwen3-4B
- target paged-KV truncate/rollback
- draft-state commit or truncation

This fixes the fake-acceptance bug but cannot produce the expected Phase 2
speedup. A true acceleration path still needs a packed K+1 target verifier that
is bit-identical to decode, or a MagicDec-style sparse-KV self-spec path.

Three measured contributors explain the regression:

- Qwen3-0.6B is not cheap enough as a draft model in this single-GPU envelope.
- Loading the draft model first shrank target KV capacity from the Phase 1 close
  pool (`136976` tokens) to `121472` tokens.
- Persistent draft state is correct, but draft/target state switching and
  target KV rollback add scheduler overhead before any verifier speedup exists.

## Command

Server:

```bash
ZIG=/tmp/zig14/zig CUDA_HOME=/usr/local/cuda target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --num-slots 16 \
  --max-seq-len 131072 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 16384 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs \
  --spec-enabled \
  --spec-draft-k 5 \
  --spec-acceptance-threshold 0.3 \
  --spec-draft-model external:/content/workspace/agent-infer/infer/models/Qwen3-0.6B
```

Benchmark:

```bash
WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 LONGCTX_MAX_SECONDS=300 \
  scripts/bench_guidellm.sh phase2-real-spec-external-qwen06b-c4 \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Environment

- **Backend:** CUDA
- **Hardware:** NVIDIA L4, 24 GB class VRAM
- **Target model:** Qwen3-4B, `infer/models/Qwen3-4B`
- **Draft model:** Qwen3-0.6B, `infer/models/Qwen3-0.6B`
- **KV cache:** FP8E4M3
- **Feature set:** `cargo build --release -p infer --features cuda`
- **Build env:** `ZIG=/tmp/zig14/zig`, `CUDA_HOME=/usr/local/cuda`
- **Code state:** dirty working tree for
  `feat(scheduler): real multi-token speculative decode with external draft model`
- **KV pool after draft load:** 121472 max tokens, 7592 pages

## Results

| metric | value |
|---|---:|
| completed requests | 6 |
| incomplete requests | 1 |
| completed output tokens | 1536 |
| incomplete output tokens | 7 |
| successful-only effective out tok/s | 5.12 |
| GuideLLM headline out tok/s | 9.73 |
| TTFT p50 | 37475.6 ms |
| ITL p50 | 105.27 ms |
| ITL p95 | 350.04 ms |
| peak active | 3 |
| peak waiting | 1 |
| peak kv_util | 94.8% |
| plan labels | `idle=1255,decode=514,prefill=50,split=0,mixed=11` |

Spec counters:

| metric | value |
|---|---:|
| draft tokens | 25 |
| verified tokens | 25 |
| accepted tokens | 3 |
| acceptance rate | 12.0% |
| spec step count | 5 |

Delta vs Phase 1 close baseline:

| metric | Phase 1 baseline | now | delta |
|---|---:|---:|---:|
| effective c=4 out tok/s | 26.169 | 5.12 | -80.4% |
| GuideLLM headline out tok/s | 26.169 equivalent baseline | 9.73 | -62.8% |
| acceptance threshold | 30.0% | 12.0% | failed |

## Problems

- Acceptance was real and low (`3/25 = 12%`), not the previous 100% canary
  artifact.
- The adaptive disable threshold did not prevent the first expensive spec
  attempts from dominating this longctx run.
- Loading Qwen3-0.6B before KV-pool sizing reduced available target KV capacity
  from the Phase 1 close pool (`136976` tokens) to `121472` tokens.
- The server still had 3 active and 1 waiting request after the 300s benchmark
  window, so successful-only throughput is the trustworthy comparison metric.

## Fix

Land the foundation because it closes the correctness and lifecycle gaps:

- external draft request-state lifecycle
- target paged-KV truncate rollback
- true K-token draft proposals
- greedy verifier accounting with bonus-token commit
- spec metrics from real draft/verify/accept events

Do not claim a Phase 2 throughput lift from this implementation. The next
performance slice must replace sequential target verification with a packed
K+1 verifier that remains greedy bit-identical, or move to the MagicDec sparse
KV path.

Next path: pause Phase 2 throughput claims after this plumbing commit. Resume
only as Phase 2.B with MagicDec sparse-KV/self-spec, or shift mission execution
to the H20/multi-GPU grid where Phase 1 already has a stronger scaling lever.

## Rule

External draft support is only a speed path if target verification is cheaper
than normal decode. A correct sequential verifier is useful foundation, but its
bench artifact must be treated as a regression/diagnosis result.

## Artefacts

- Raw: `bench-output/2026-05-01-phase2-real-spec-external-qwen06b-c4/benchmarks.json`
- CSV: `bench-output/2026-05-01-phase2-real-spec-external-qwen06b-c4/benchmarks.csv`
- HTML: `bench-output/2026-05-01-phase2-real-spec-external-qwen06b-c4/benchmarks.html`
- Service trace before:
  `bench-output/2026-05-01-phase2-real-spec-external-qwen06b-c4/service_stats_before.txt`
- Service trace during:
  `bench-output/2026-05-01-phase2-real-spec-external-qwen06b-c4/service_stats_trace.jsonl`
- Service trace after:
  `bench-output/2026-05-01-phase2-real-spec-external-qwen06b-c4/service_stats_after.txt`
- Service trace summary:
  `bench-output/2026-05-01-phase2-real-spec-external-qwen06b-c4/service_stats_trace_summary.md`
