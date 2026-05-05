# 2026-05-05 · TileLang Phase 0 (dead Triton scaffolding) — pending-remote regression check

## Status: pending-remote

The Phase 0 commit (`38d4d773 refactor(cuda): TileLang Phase 0 — delete dead
Triton scaffolding`) is build-only — it removes Triton AOT generation specs for
5 kernels whose runtime symbols are already provided by native CUDA C
implementations at `crates/cuda-kernels/csrc/misc/elementwise_basic.cu` (silu_mul,
add, embedding_decode, embedding_batched) plus a zero-caller dead kernel
(flash_attention_prefill_hd256). The C ABI symbols (`silu_mul_triton_aot_cuda`,
`add_cuda`, `embedding_*_cuda`) are unchanged across the change — only the
duplicate Triton dispatch wrappers go away.

Per `CLAUDE.md` §Benchmarks, build-only diffs that don't change the runtime
path are eligible for the `bench: exempt` claim. Round 3 codex review flagged
this as a P2 anyway out of caution because the diff touches files under
`crates/cuda-kernels/csrc/` and `crates/cuda-kernels/build.rs`. To honor the
3+ review rounds + bench discipline rule, this entry stands as the formal
regression-check artifact.

## Pending bench

When codex@2's in-flight W4 H5 canonical bench (commit `bae845e0`) finishes
and the GPU is free, restart the server with the Phase 0 binary and run a
single guidellm regression check against the most recent Qwen3-4B FP8 L4
baseline (`docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-headline-summary.md`):

```bash
source /tmp/arle-env.sh
CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda --bin infer
./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 8192 \
  --kv-cache-dtype fp8

# Separate terminal:
scripts/bench_guidellm.sh tilelang-phase0-qwen3-fp8 \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --concurrencies 16 \
  --max-seconds 120
```

## Acceptance bar

Out tok/s within ±2% of `138.17` (the 2026-04-29 Qwen3-4B FP8 L4 baseline) →
`135.41 ≤ out_tok_s ≤ 140.93`.

If the regression-check passes, this entry is updated with the actual numbers
and the `pending-remote` tag is removed.

If the regression-check fails (which would be surprising given the runtime
path is unchanged), file a `docs/experience/errors/` entry pointing back at
this win-stub and revert `38d4d773`.

## Hypothesis

Net zero perf delta. Removing duplicate Triton AOT C wrappers does not change
which symbols the linker resolves (csrc native was already winning the link
for the 4 deduped symbols); it only stops the build from generating dead
intermediate `.c` files and per-SM cubins. Build time should drop slightly
(no longer paying for 5 Triton AOT runs); runtime perf is unchanged.

## Reviews summary

- Round 1 self-review (Claude): PASS
- Round 2 `cargo check --no-default-features --features no-cuda`: PASS (33s v1, 30s v2)
- Round 3 v1 `codex review --uncommitted` (with release CUDA cargo build):
  CAUGHT silu_mul_triton_aot_cuda multi-definition link error
- Round 3 v2 (post-fix): PASS on no-cuda compile, P2 finding asking for this bench entry
- Round 4 codex@0 peer review: PASS on code structure; REQUEST_CHANGES on 4
  stale docs references — addressed in follow-up commit

## Pointers

- Commit: `38d4d773 refactor(cuda): TileLang Phase 0 — delete dead Triton scaffolding`
- Plan: `docs/plans/2026-05-05-cuda-kernel-tilelang-unification.md` §3 Phase 0
- 2026-04-29 Qwen3-4B FP8 L4 baseline: `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-headline-summary.md`
- Round 3 v1 codex review log (where the silu collision was caught): `/tmp/phase0_codex_review_2026-05-05.log`
- Round 3 v2 codex review log: `/tmp/phase0_v2_codex_review_2026-05-05.log`
- Round 4 codex@0 peer review: `/tmp/phase0_review_2026-05-05.md`
