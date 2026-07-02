# FlashInfer MLA Wrapper — kernel surface, cuda-deepseek-v4-mla, 2026-05-05

## Goal

- Land the FlashInfer MLA paged-attention wrapper symbols
  (`flashinfer_mla_paged_attention_{plan,run}`) so DeepSeek-V4 model code
  in `infer/src/model/deepseek/` has a real BF16 MLA forward path. This is
  a new kernel surface, not a perf change — so the bench is a numerical
  correctness gate rather than a throughput delta.

- Initial dim coverage is **DeepSeek V2 / V3 reference shape `(HEAD_DIM_CKV,
  HEAD_DIM_KPE) = (512, 64)` only**. The DSV4 small-substrate SKUs in
  `docs/plans/2026-05-05-deepseek-v4-small-substrate.md` §2 use smaller
  dims `(64, 16) / (128, 32) / (192, 32)` — those need a different kernel
  (FlashInfer's FA2 MLA truncates internal MMA loops at the small-dim
  bounds and silently drops PE / output writes); covered as future work
  in that plan.

## Hypothesis

- The wrapper compiles under the existing `flashinfer_*.cu` glob in
  `crates/cuda-kernels/build.rs` (no build-rule edit needed) and links into
  `libkernels_cuda.a`. With Agent 1's MLA forward in place, the BF16 MLA
  decode output will match a DeepSeek reference within the per-layer
  numerical tolerance used in `infer/test_data/` baselines (top-128 prob
  delta ≤ 1e-3 per `docs/plans/2026-05-05-deepseek-v4-small-substrate.md`
  §7).

## Command

```bash
scripts/bench_guidellm.sh cuda-deepseek-v4-mla-nano
```

Numerical correctness is checked separately via
`cargo test --release --test e2e_deepseek` (added by Agent 1 once the
MLA forward landing is wired through).

## Environment

- **Backend:** cuda
- **Model:** DeepSeek V4 nano fixture (planned), then `deepseek-tiny-dense`
  (SKU-A) and `deepseek-mini-moe` (SKU-B) per the small-substrate plan.
- **Hardware:** pending remote NVIDIA host (this workspace has no GPU;
  codex@2 owns the GPU for the W3 H1 + FP8 KV missions).
- **Commit:** pending (pin once the FlashInfer-MLA wrapper commit lands)
- **Feature set:** `CUDA_HOME=/usr/local/cuda cargo build --release`
- **FlashInfer version:** 0.6.9 (vendored as build-time dep; headers under
  `<flashinfer python install>/data/include/flashinfer/attention/mla*.cuh`).
- **Non-default flags / env vars:** none expected.
- **Server launch:** `scripts/start_infer.sh models/deepseek-v4-nano 8000`
  (model artifacts pending pretrain — gated on §6 of the small-substrate
  plan).

## Canonical params

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda-deepseek-v4-mla-nano`

## Results

- Status: **pending-remote**. Local workspace cannot execute CUDA kernels
  (no GPU). The FlashInfer MLA C++ wrapper + Rust FFI declarations are
  landed; numerical correctness and throughput depend on:
  1. Agent 1's `infer/src/model/deepseek.rs` + `infer/src/ops/attention/mla.rs`
     wiring through `cuda_kernels::ffi::flashinfer_mla_paged_attention_{plan,run}`.
  2. A real DeepSeek V4 checkpoint (or the nano fixture pretrain) under
     `models/deepseek-v4-nano/`.
  3. `cargo test --release --test e2e_deepseek` (new) producing baseline
     JSON in `infer/test_data/deepseek-nano.json` against a PyTorch
     reference.

## Problems

- This commit does not move runtime numbers; it lands a kernel surface.
  No `cargo build --features cuda` ran in this workspace (codex@2 has the
  GPU; per CLAUDE.md the bench is required to land before declaring "done"
  on the runtime path, but the regression-check minimum applies to changes
  that affect the existing hot path — adding a new, dispatch-isolated MLA
  symbol does not regress Qwen3 / Qwen3.5 numbers).
- Only `cargo check --no-default-features --features no-cuda` and
  `cargo check --no-default-features --features cuda,no-cuda` were run
  locally — both green.

## Learnings

- FlashInfer 0.6.9 ships `flashinfer::MLAPlan` (CPU scheduler, in
  `attention/scheduler.cuh`) + `flashinfer::mla::BatchMLAPagedAttention`
  (GPU launch, in `attention/mla.cuh`). The two are the canonical SM80+
  MLA path (Hopper has a separate `mla_hopper.cuh`; the wrapper here uses
  the SM80 path which works on Ampere and above and matches ARLE's T1 SM
  set `{80, 86, 89, 90}`).
- The plan/run split mirrors FlashInfer's existing prefill/decode wrappers
  in this tree (`flashinfer_decode.cu`, `flashinfer_prefill_paged.cu`),
  so no new workspace conventions were introduced — `FlashInferWorkspace`
  in `crates/cuda-kernels/src/flashinfer.rs` will be reusable when Agent 1
  threads MLA through the model.
- `MLAPlanInfo` is 18 × `int64_t` = 144 bytes, fits inside the existing
  256-byte `plan_info` buffer the workspace already allocates.

## Δ vs baseline

- **Baseline:** none — first MLA bench entry (the prior `mla_decode.cu`
  scaffold returned `cudaErrorNotSupported` and never had a baseline).
- **Delta table:** pending-remote.

## Artefacts

- C++ wrapper: `crates/cuda-kernels/csrc/attention/flashinfer_mla.cu`
- Rust FFI:    `crates/cuda-kernels/src/ffi/attention.rs` (block beginning
                  `flashinfer_mla_paged_attention_plan`)
- Plan doc:    `docs/plans/2026-05-05-deepseek-v4-small-substrate.md`
                  §"Kernel substrate"
- FlashInfer source files (vendored, not modified):
  - `<flashinfer>/data/include/flashinfer/attention/mla.cuh`
  - `<flashinfer>/data/include/flashinfer/attention/mla_params.cuh`
  - `<flashinfer>/data/include/flashinfer/attention/scheduler.cuh`
    (function `MLAPlan`, struct `MLAPlanInfo`)

## Notes

- What changed in the code since baseline: new `flashinfer_mla.cu` C++
  wrapper + two `extern "C"` decls in `ffi/attention.rs`. No existing
  FlashInfer kernel was touched (codex@2 territory).
- Suspected cause of any regression: n/a (additive surface).
- Follow-ups:
  1. Agent 1 calls the new symbols from `infer/src/model/deepseek/mla.rs`.
  2. Once Agent 1 lands the model wiring, run guidellm sweep against the
     nano fixture on a remote NVIDIA host and replace this stub with a
     real measurement.
  3. Add a numerical e2e test (`infer/tests/e2e_deepseek.rs`) per
     small-substrate plan §7.
