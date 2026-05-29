# DSv4 decode restored at HEAD — oplib::linear missing BF16-graphsafe decode branch

## Context

Chasing native-deepep, I synced the pod to TRUE HEAD (it had been a stale scp
snapshot, pre dispatch-governance merge) and found the DEFAULT allreduce DSv4
decode panicking at HEAD:
`internal error: entered unreachable code: linear decode plan Bf16GraphsafeGemm
has no matching weight storage` (ops/linear.rs). Not my RoPE work — a regression
from the dispatch-governance / `oplib::linear` refactor (a33ed6fd relocated
`plan()` into `oplib::linear::plan`, f9aa62f7 added the Observe-gate counter).

## Root cause (verified locally vs HEAD source)

`oplib::linear::plan()` returns `Bf16GraphsafeGemm` for `(N=1, DenseBf16)` decode
weights (oplib/linear.rs:383, :477) — e.g. DSv4's attention + shared-expert
projections. But the decode executor `gemv_with_marlin_scratch`
(ops/linear.rs:660) only wired `Bf16Gemv` for dense-bf16; storage branches cover
marlin / TurboQuant / Dsv4Fp8/Fp4 / quantized(qweight), then fall through to
`unreachable!("no matching weight storage")`. So a `Bf16GraphsafeGemm`
selection on a plain-bf16 decode weight panicked. Affected EVERY DSv4 decode at
HEAD (and any dense-bf16 model the policy routes to graphsafe). Masked for weeks
only because the pod build was a stale pre-refactor snapshot.

## Fix

`infer/src/ops/linear.rs` (commit `dbf7fde7`): add the bf16-dense decode branch
before the final unreachable —
```rust
if matches!(plan, LinearKernel::Bf16GraphsafeGemm | LinearKernel::Bf16CublasGemm) {
    // weight.data dense bf16, N=1 → gemm_graphsafe_cuda (mirrors run_bf16_linear:2134)
    ffi::gemm_graphsafe_cuda(w_ptr, x_ptr, y_ptr, weight.rows, 1, weight.cols, stream)?;
    return Ok(());
}
```

## Validation (8×H20 TP=8, CLEAN true-HEAD build — `rm -rf infer && tar -x` then build)

Default allreduce path, needle-in-haystack (greedy):

| prompt_tok | 40 | 472 | 1147 | 2272 |
|---|----|-----|------|------|
| result | HIT | HIT | HIT | **HIT** |

**4/4 HIT** — decode no longer panics (oplib fix) AND long-context retrieval works
(the RoPE fix, re-confirmed on a clean true-HEAD build). Before the oplib fix,
every decode HTTP-500'd at HEAD.

## Rule

- A new dispatch-`plan()` variant (here `Bf16GraphsafeGemm`) MUST have a matching
  executor branch for every weight storage it can be selected for. When relocating
  plan selection, grep every `unreachable!("no matching ...")` and confirm the new
  variants are wired in ALL executors (decode `gemv_with_marlin_scratch` was
  missed). A CPU-testable `plan()` doesn't catch a missing GPU executor arm.
- The pod `/data01/build/arle` is a scp snapshot, NOT a git checkout — it silently
  ran weeks-stale `oplib`/`mlp` while accepting freshly-pushed individual files,
  producing a frankenstein tree whose "build OK" meant *stale-but-consistent*
  compiled, not HEAD. See [[project_h20_pod_access]]: sync the WHOLE tree
  (`rm -rf infer && tar -xf` IN TMUX — kubectl-exec SIGTERM-kills long extracts
  mid-way, leaving partial trees) and verify by BUILD, never by flaky greps.
