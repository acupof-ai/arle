# CUDA TileLang prefill HD128 — guidellm sweep, cuda, 2026-04-26

> Phase 0 stub — Option A (build-time AOT). Created at the same time as
> the implementation tranche per `docs/plans/tilelang-integration.md` §7.
> Replace this file with a completed entry (and a separate `…-off` baseline
> entry) once the H100 sweep runs.

## Goal

- Optimization (per `docs/bench-and-trace-spec.md` §goal taxonomy): measure
  end-to-end TTFT and saturation throughput delta between the FlashInfer
  prefill HD128 path (default build) and the TileLang prefill HD128 path
  (`--features cuda,tilelang-attn`) on H100, for the Qwen3-4B sweep.

## Hypothesis

- TileLang's CuTeDSL / TMA + WGMMA path on Hopper produces a measurable
  TTFT improvement over the current FlashInfer batch prefill HD128 kernel.
  The Phase 0 decision threshold (`docs/plans/tilelang-integration.md` §5)
  is **≥10% on TTFT or saturation tok/s** to advance to Phase 1; ±5% flat
  → ship-and-hold; ≥5% worse → revert.

## Command

```bash
# Off (default build, FlashInfer)
scripts/bench_guidellm.sh tilelang-prefill-off

# On (TileLang path)
scripts/bench_guidellm.sh tilelang-prefill-on
```

Invoked via: pending remote H100 host (user-driven verification).

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote NVIDIA H100 host
- **Commit:** pending; will be filled in once the implementation tranche
  lands and the remote sweep runs.
- **Feature set (off):** `CUDA_HOME=/usr/local/cuda cargo build --release --features cuda` (workspace root)
- **Feature set (on):**  `CUDA_HOME=/usr/local/cuda cargo build --release --features cuda,tilelang-attn` (workspace root)
- **Non-default flags / env vars:** `INFER_TILELANG_PYTHON` if a non-default
  Python interpreter is needed for AOT, otherwise none.
- **Server launch (off):** `INFER_FEATURES="cuda" scripts/start_infer.sh models/Qwen3-4B 8000`
- **Server launch (on):**  `INFER_FEATURES="cuda,tilelang-attn" scripts/start_infer.sh models/Qwen3-4B 8000`
  Both invocations build and run the `infer` binary directly so the Tile-
  Lang feature actually reaches the prefill path.

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh tilelang-prefill-{on,off}`

## Results

- Status: `pending-remote`
- Local verification completed (macOS workspace):
  - `cargo fmt --check`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda,tilelang-attn`
  - `cargo check --features cuda,no-cuda,tilelang-attn` verifies the explicit
    workspace root feature chain root → cli → infer → cuda-kernels.
  - `cargo check --features tilelang-attn,no-cuda` verifies `tilelang-attn`
    itself implies the CLI CUDA backend.
  - (Optionally: `cargo check -p cuda-kernels --features tilelang-attn`
    once the spike lands; this verifies the build.rs Python probe path
    without nvcc.)

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| synchronous | pending | pending | pending | pending | pending | pending |
| saturation  | pending | pending | pending | pending | pending | pending |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | pending |
| peak waiting | pending |
| peak prefill_queue | pending |
| peak kv_util | pending |
| `prefix_hit_rate` | pending |

## Problems

- No CUDA runtime host is available in this macOS workspace, so the
  canonical `guidellm` sweep is pending.
- Risk gate per plan §5: TileLang AOT export for `sm_90` may fail on
  first run; if so, this entry is replaced with an `errors/` entry and
  the implementation is reverted.

## Learnings

- Filled in after the remote run, per the bench-and-trace spec.

## Δ vs baseline

- **Baseline:** the matched `tilelang-prefill-off` run on the same
  commit + same H100 host, taken as the immediately-preceding sweep
  (per `feedback_matched_ab_for_small_bench_effects.md`).
- **Delta table:** pending remote run.

| metric | baseline (off) | now (on) | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | pending | pending | pending |
| out tok/s @ saturation | pending | pending | pending |

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in code since baseline: see
  `docs/plans/tilelang-integration.md` §4 — TileLang AOT track added to
  `crates/cuda-kernels/build.rs`, new FFI for
  `tilelang_batch_prefill_paged_hd128_run`, compile-time dispatch in
  `infer/src/ops/attention.rs::prefill_attention_paged_batch`, and a
  TileLang Qwen3 prefill path that uploads shared metadata without
  constructing a FlashInfer `BatchPrefillPagedPlan`. No change to the prep
  kernel, HD256 path, decode path, or any default build.
- Suspected cause of any regression: AOT compile may pick suboptimal
  pipeline stages or warp count; tunables logged in the kernel module
  per plan §6.
- Follow-ups: replace this stub with completed off + on entries; if win
  ≥10%, open Phase 1 plan for decode migration.
