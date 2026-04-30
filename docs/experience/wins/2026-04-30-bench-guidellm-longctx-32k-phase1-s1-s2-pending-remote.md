# Longctx 32k Phase 1 S1/S2 — pending remote guidellm

## Goal

- Optimization: validate Phase 1 S1/S2 from
  `docs/projects/2026-04-30-longctx-32k-128k-leadership.md` on Qwen3-4B
  FP8 KV, L4, prompt=32768 / output=256 / c=4.

## Hypothesis

- Split-KV varlen quantized attention plus mixed FP8/INT8 wiring should make
  Qwen3-4B use `StepPlan::Mixed` at 32k instead of falling back to `Split`,
  creating the foundation required before the world-#1 mission can move to
  Phase 2.

## Command

```bash
WORKLOAD=longctx-32k scripts/bench_guidellm.sh longctx-32k-phase1-s1-s2
scripts/bench_sglang_longctx.sh longctx-32k-phase1-s1-s2
```

Local validation run before remote bench:

```bash
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo fmt --all --check
cargo test --release
git diff --check
bash -n scripts/bench_guidellm.sh scripts/bench_sglang_longctx.sh
scripts/bench_guidellm.sh --help
WORKLOAD=longctx-32k scripts/bench_guidellm.sh --help
scripts/bench_sglang_longctx.sh --help
codex review --uncommitted
```

## Environment

- **Backend:** CUDA
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote NVIDIA L4, CUDA host with nvcc
- **Commit:** pending; fill after commit lands
- **Feature set:** `CUDA_HOME=/usr/local/cuda cargo build --release`
- **Non-default flags / env vars:** `WORKLOAD=longctx-32k`
- **Server launch:** pending remote launch with FP8 KV, `--num-slots 16`,
  `--max-seq-len 131072`, and `--mem-fraction-static 0.85`

## Canonical params

- `--profile concurrent`
- `--rate 1,4`
- `--data prompt_tokens=32768,output_tokens=256`
- `--max-seconds 300`
- Secondary c=1 publication run: `LONGCTX_SECONDARY_C1_ONLY=1`, 360s
- Wrapper: `WORKLOAD=longctx-32k scripts/bench_guidellm.sh <label>`

## Results

- Status: `pending-remote`

| metric | value |
|---|---:|
| ARLE tok/s c=4 | pending |
| SGLang tok/s c=4 | pending |
| ARLE/SGLang | pending |
| `StepPlan::Mixed` count | pending |
| `StepPlan::Split` count on Qwen3-4B | pending |
| TTFT p50/p99 | pending |
| ITL p50/p99 | pending |

## Problems

- This macOS workspace has no `nvcc`, so CUDA C compilation, kernel numerical
  tests, e2e CUDA tests, and guidellm must run on the remote L4 host.
- The implementation intentionally keeps FP8 scale pointers in the varlen ABI:
  current FP8 durable KV is scaled E4M3 after the 2026-04-30 numerical drift
  fix, so scale-null-as-format-discriminant would be wrong for this tree.
- `cargo clippy -p infer --no-default-features --features cuda,no-cuda --
  -D warnings` was attempted locally and is blocked by pre-existing lint debt
  outside this change, including `infer/src/model/common.rs`
  `struct_field_names`, `infer/src/model/generation_state.rs`
  `unnecessary_wraps`, and multiple Qwen3.5/scheduler lint failures.

## Learnings

- Pending remote run.

## Delta vs baseline

- **Baseline:** pending SGLang run from `scripts/bench_sglang_longctx.sh` on
  the same host and same shape.

| metric | baseline | now | delta |
|---|---:|---:|---:|
| out tok/s c=4 | pending | pending | pending |
| ITL p50 c=4 | pending | pending | pending |
| TTFT p50 c=4 | pending | pending | pending |

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- Code since baseline:
  - `decode_attention_varlen_fp8.cu` now uses two-phase split-KV for varlen
    FP8/INT8 mixed batches.
  - Qwen3 mixed batching now admits BF16, FP8 E4M3, and INT8 KV formats.
  - Mixed quantized rows are written back from the BF16 work buffer before
    varlen attention reads durable KV.
  - Mixed quantized scratch is allocated only for FP8/INT8 KV so BF16 mixed
    capacity does not regress.
