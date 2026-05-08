# B3 Step 2 — PrefixAwareAdmission CUDA Runtime Gate

## Goal

Wire the existing `PrefixAwareAdmission` policy into the CUDA runtime admission path after radix prefix lookup, without adding `RadixCache` to `SchedulerHandle` or changing the default queue-bound behavior.

## Hypothesis

Warm/shared-prefix turns should avoid being buried behind cold requests once the prefix-aware policy is enabled. The target from the pickup directive was multi-tenant TTFT p50 `318ms -> 157ms`.

## Command

Server:

```bash
CUDA_HOME=/opt/cuda \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 8 \
  --max-seq-len 8192 \
  --admission-policy prefix-aware
```

GuideLLM attempt:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh b3step2-prefix-aware \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=6000,prompt_tokens_stdev=1,prompt_tokens_min=6000,prompt_tokens_max=6000,output_tokens=200,output_tokens_stdev=1,output_tokens_min=200,output_tokens_max=200,turns=3,session_count=4'
```

Session/shared-prefix burst validation:

```bash
/home/ckl/projects/arle/.venv/bin/python \
  scripts/bench_multitenant_burst.py \
  http://localhost:8000 Qwen/Qwen3-4B
```

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** RTX 4070 Ti SUPER 16GiB, CUDA 13.2
- **Commit:** this commit
- **Feature set:** `cargo check --release -p infer --features cuda`
- **Non-default flags:** `--admission-policy prefix-aware`, `--num-slots 8`, `--max-seq-len 8192`

## Results

GuideLLM produced raw artefacts but failed validation: `turns=3` expanded the actual input to p50 `12201` / p95 `18205` tokens, exceeding the 8192-token server context and producing many zero-output completions. The raw output is retained for diagnosis, but it is not used as the license result.

Raw artefacts:

- `bench-output/2026-05-09-b3step2-prefix-aware/benchmarks.json`
- `bench-output/2026-05-09-b3step2-prefix-aware/benchmarks.csv`
- `bench-output/2026-05-09-b3step2-prefix-aware/service_stats_trace_summary.md`

Warm-cache shared-prefix burst:

| run | TTFT p50 | min | max | burst wall |
|---:|---:|---:|---:|---:|
| 1 | 244 ms | 146 ms | 245 ms | 1563 ms |
| 2 | 241 ms | 147 ms | 242 ms | 1559 ms |
| 3 | 218 ms | 119 ms | 218 ms | 1533 ms |
| 4 | 239 ms | 133 ms | 239 ms | 1554 ms |
| 5 | 249 ms | 135 ms | 249 ms | 1564 ms |

Summary:

| metric | baseline | now | delta |
|---|---:|---:|---:|
| multi-tenant TTFT p50 | 318 ms | 241 ms median | -24.2% |
| TTFT p50 mean | n/a | 238 ms | n/a |
| TTFT p50 sigma / mean | n/a | 4.5% | under 5% |

## Problems

- The exact GuideLLM `turns=3,session_count=4` shape is not valid with `--max-seq-len 8192`; it generates 12k-18k-token requests and trips length completion / zero-output validation.
- The measured gain does not reach the 157ms target. With the default `max_waiting_requests=256`, the cold-headroom gate only activates under large queue pressure. This patch lands the CUDA runtime radix-lookup gate only; ingress remains queue-bound so true prefix hits are never rejected before `lookup_or_stage` can classify them.

## Learnings

- Keep `RadixCache` inside the CUDA runtime. The right integration point is the existing `lookup_or_stage` result in `runtime/admission.rs`, not a cross-backend field on `SchedulerHandle`.
- Do not guess prefix warmth at `SchedulerHandle::submit`. A backend-neutral ingress gate cannot distinguish a new cold session id from a real shared-prefix hit. The safe boundary is queue-bound ingress plus prefix-aware gating after CUDA radix lookup.
- `turn_depth` is still unavailable on `IncomingRequest`; `turn_depth=0` is safe for Step 2 because prefix hits and session holds already classify warm requests.
- Default `queue-bound` remains the production-safe path; `prefix-aware` is opt-in pending a wider multi-tenant bench matrix.

## Verification

- `cargo fmt --all --check` PASS
- `cargo check --release -p infer --features cuda` PASS
- `cargo check --release -p infer --no-default-features --features metal,no-cuda` PASS
- `cargo clippy --release -p infer --features cuda -- -D warnings` PASS
- `cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings` PASS
- `cargo test --release -p infer scheduler::types::tests` PASS
- `cargo test --release -p infer --features cuda scheduler::` PASS (`182` tests)
- `cargo test --release -p infer` ran 565 lib tests PASS, then failed in pre-existing `metal_eval_audit` allowlist classification for `infer/src/backend/metal/kv_pool.rs`.
- `cargo clippy --release -p infer --features cuda --all-targets -- -D warnings` is blocked by existing test-target lint debt outside this patch.

## Rule

For config-coupled scheduler changes, grep and bench both the default path and the new opt-in path. If the benchmark generator expands request shape (`turns`, chat templates, synthetic columns), validate actual token counts before accepting TTFT/ITL.
