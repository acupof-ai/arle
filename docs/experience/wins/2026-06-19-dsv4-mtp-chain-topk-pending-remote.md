# DSv4 MTP Chain Topk Pending Remote — guidellm, CUDA, 2026-06-19

## SLO-shape probed? N — pending remote

No H20/pod run in this local tranche. This entry exists to keep the runtime
change traceable until the DSv4 TP4/TP8 remote bench is rerun.

## Roofline check

Deferred — no GPU profile in this local tranche.

## Goal

- Keep DSv4 MTP verify on the bounded top-1 chain (`depth + 1` rows) while
  allowing each draft row to record `topk` candidates for target-top1 matching.

## Hypothesis

- D2/K2 should still verify 3 rows, not 7. `topk` can turn a non-chain
  candidate into the bonus token, but it cannot extend the committed prefix
  beyond that candidate unless the later rows were actually drafted/verified on
  that branch.

## Command

```bash
pending remote: scripts/bench_guidellm.sh <dsv4-label> ...
```

## Environment

- **Backend:** cuda
- **Model:** DSv4/GLM family
- **Hardware:** pending remote H20/pod
- **Commit:** pending
- **Feature set:** local static gate used `--release --no-default-features --features cuda,no-cuda`
- **Non-default flags / env vars:** `CUDARC_CUDA_VERSION=12090`
- **Profiling state:** OFF for local checks
- **Server launch:** pending remote

## Results

Pending remote. Local gates passed:

- `rustfmt --check crates/infer-cuda/src/executor/spec_decode.rs crates/infer-cuda/src/dsv4.rs`
- `CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release --no-default-features --features cuda,no-cuda spec_decode --lib`
- `CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib -- -D warnings`

## Problems

- GPU throughput and acceptance-rate impact remain unmeasured in this tranche.

## Learnings

- Current ARLE CLI has `depth` and `topk`, but no SGLang-style
  `draft_token_num` selector. Therefore topk must not silently expand into a
  full `K^D` verify tree; that overstates verifier cost and regresses latency.
- A full SGLang-style topk tree is a separate feature: it needs a draft-token
  budget/selector plus tree-mask metadata. Until that exists, this lane remains
  chain-shaped and performance-comparable to D2K1 on verifier row count.

## Delta vs baseline

- **Baseline:** `docs/experience/wins/2026-06-18-dsv4-chain-verify-batched-sparse-pending.md`
- **Delta:** pending remote.
