# Hybrid W4 Phase 1b Loader — substrate gate, CUDA, 2026-05-09

## Goal

- **Diagnosis / regression gate:** prove that `marlin_w4_hybrid` checkpoints load both the W4A16 Marlin tensors and the W4A8 side tensors without changing the active single-format runtime paths.

## Hypothesis

- A hybrid checkpoint can reuse `DeviceMatrix` storage by adding W4A8 sidecar tensors beside the existing W4A16 Marlin fields; no new `HybridLinear` type is needed.
- Throughput should be unchanged in this phase because runtime dispatch is intentionally not changed. The useful signal is loader correctness and no regression in existing single-format CUDA tests.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo check --release -p infer --features cuda

cargo test --release -p infer --features cuda \
  load_hybrid_w4_marlin_linear_populates_side_tensors -- --nocapture

cargo test --release -p infer --features cuda marlin_w4_hybrid -- --nocapture

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo clippy --release -p infer --features cuda --lib -- -D warnings

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda --test e2e -- --test-threads=1

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda --test greedy_consistency -- --test-threads=1

PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh hybrid-phase1b-loader-regression \
  --concurrencies 1 --max-seconds 30 --warmup 5 \
  --data 'prompt_tokens=512,prompt_tokens_stdev=1,prompt_tokens_min=512,prompt_tokens_max=512,output_tokens=64,output_tokens_stdev=1,output_tokens_min=64,output_tokens_max=64'
```

The GuideLLM run is an exploration-mode regression smoke because Phase 1b does not route serving traffic through the hybrid W4A8 side tensors yet. It still exercises the same BF16 Qwen3-4B serving path after the loader and warmup changes.

## Environment

- **Backend:** CUDA
- **Model:** loader test uses `infer/models/Qwen3-4B-W4-hybrid-zpfix`
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- **CUDA / driver:** CUDA 13.2.78, driver 595.71.05
- **Commit:** this commit, based on `919c0fb`
- **Feature set:** `--features cuda`
- **Non-default flags / env vars:** `NVCC_CCBIN=/usr/bin/g++-14`, `INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python`, `TORCH_CUDA_ARCH_LIST=8.9`
- **Server launch:** n/a, loader substrate gate only

## Results

| check | result | notes |
|---|---:|---|
| `cargo check --release -p infer --features cuda` | PASS | CUDA typecheck clean |
| `load_hybrid_w4_marlin_linear_populates_side_tensors` | PASS | `hybrid_w4a8_qweight`, `hybrid_w4a8_s_channel`, and `hybrid_w4a8_s_group` are populated for a Linear layer |
| `cargo test --release -p infer --features cuda marlin_w4_hybrid` | PASS | config parse + loader config arm |
| `cargo clippy --release -p infer --features cuda --lib -- -D warnings` | PASS | library target clean |
| `cargo test --release -p infer --features cuda --test e2e -- --test-threads=1` | PASS | existing e2e suite |
| `cargo test --release -p infer --features cuda --test greedy_consistency -- --test-threads=1` | PARTIAL | 2/3 pass; existing W4A8 accuracy gate still fails; see Problems |
| `scripts/bench_guidellm.sh hybrid-phase1b-loader-regression ...` | PASS | c=1, 512-in/64-out smoke, 0 request errors |
| `codex review --uncommitted` | FIXED | packed length validates at load time; qweight-less hybrid matrices now fail fast at linear dispatch instead of falling through to the W4A16 GEMV unwrap path |

## Results — loader behavior

| checkpoint type | expected path | result |
|---|---|---|
| `marlin_w4_hybrid` | W4A16 Marlin fields + W4A8 side tensors | PASS |
| `marlin_w4a8` / `w4a8_marlin` | existing W4A8-only loader arm | unchanged |
| GPTQ / W4A16 Marlin | existing quantized loader arms | unchanged |

## Results — GuideLLM smoke

| label | shape | TTFT p50 | ITL p50 | out tok/s | total tok/s | req/s | errors |
|---|---|---:|---:|---:|---:|---:|---:|
| `hybrid-phase1b-loader-regression` | 512-in / 64-out, c=1 | 68.4 ms | 14.02 ms | 67.4 | 607.61 | 1.04 | 0 |

Service trace: peak active 1, peak waiting 0, peak running_batch 1, peak `kv_util=28.9%`, plan labels `idle=61261`, `decode=2086`, `prefill=34`, `split=0`, `mixed=0`.

## Problems

- Full `cargo clippy --release -p infer --features cuda --all-targets -- -D warnings` is blocked by unrelated pre-existing test-target warnings (`float_cmp`, unreadable literals, single-character names, and similar lints under `infer/src/ops/tests.rs`, scheduler runtime tests, Metal tests, and several integration tests). The library clippy gate for this diff is clean.
- Full `greedy_consistency` still fails `test_w4a8_vs_bf16_token_diff` with the known W4A8 accuracy issue: BF16 starts with `Paris`, while W4A8 diverges later in the 32-token sample and reports an 84.4% token diff. The hybrid loader path is not active in that test; `marlin_w4_hybrid: false` is logged for the failing W4A8-only checkpoint.
- Hybrid checkpoints do not carry row-major `.qweight` / `.scales`; they carry only Marlin-packed W4A16 tensors plus W4A8 side tensors. Phase 1b deliberately does not add a serving dispatch path for those tensors because the existing Marlin helper allocates scratch per call. Phase 2 must add a non-allocating dispatch path before hybrid checkpoints are used for serving.
- A review caught that simply tagging hybrid matrices as W4A16 would let decode reach the row-major W4A16 GEMV path and panic on missing `qweight`. Phase 1b now rejects hybrid matrices at linear dispatch with an explicit "runtime dispatch is not enabled" error until Phase 2 wires a real path.

## Learnings

- Hybrid W4 can be represented as W4A16 runtime fields plus optional W4A8 sidecar tensors on `DeviceMatrix`; adding a parallel `HybridLinear` type would duplicate ownership without improving dispatch.
- "W4A16" is not enough to pick a serving kernel: row-major W4A16 uses GEMV, while Marlin-only hybrid W4A16 needs a dedicated non-allocating path before it can serve traffic safely.
- Prefill admission caps and decode warmup slot counts are different invariants. Warmup batch sizes must map to real scheduler slots; with `num_slots=4` and an admission cap of 8, warming 8 decode rows indexes slot-local paged-KV state out of bounds.

## Δ vs baseline

- **Baseline:** Phase 1a checkpoint merge substrate `b6502f7`

| metric | baseline | now | Δ |
|---|---:|---:|---:|
| hybrid loader side tensors visible | no runtime loader support | yes | unlocked |
| active dispatch path | W4A16 or W4A8 single-format | unchanged | 0 |
| 512-in/64-out c=1 smoke | n/a | TTFT p50 68.4 ms, ITL p50 14.02 ms | no errors |
| expected hybrid throughput change | n/a | n/a | Phase 2 dispatch required |

## Artefacts

- Raw bench artefacts: `bench-output/2026-05-09-hybrid-phase1b-loader-regression/`
- Test artefacts: terminal output from the commands above.

## Notes

- Code changed since baseline: `QuantMeta::MarlinW4Hybrid`, `QuantLoadConfig::marlin_w4_hybrid`, `DeviceMatrix::from_hybrid_w4_marlin`, the hybrid loader branch in `weight_loader.rs`, and decode warmup clamping to real slot count.
- Follow-up: Phase 2 should route prefill linear dispatch to `hybrid_w4a8_*` once the W4A8 accuracy gate is fixed.
