# DSv4 MTP Tree Verify Pending Remote — guidellm, CUDA, 2026-06-19

## SLO-shape probed? N — pending remote

No H20/pod run in this local tranche. This entry exists to keep the runtime
change traceable until the DSv4 TP4/TP8 remote bench is rerun.

## Roofline check

Deferred — no GPU profile in this local tranche.

## Goal

- Replace chain-only MTP acceptance with a bounded top-k tree schedule verified
  by one target pass using explicit ancestor metadata.

## Hypothesis

- D2/K2 should expose 7 verifier rows in a single sparse verify forward, so an
  off-chain top-k child can be accepted only when its full ancestor path is
  present in the verified tree.

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

- `rustfmt --check crates/infer-cuda/src/executor/spec_decode.rs crates/infer-cuda/src/dsv4.rs crates/infer-cuda/src/loader.rs crates/infer-cuda/src/moe.rs`
- `CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release --no-default-features --features cuda,no-cuda spec_decode --lib`
- `CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`
- `CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib -- -D warnings`

## Problems

- GPU throughput and acceptance-rate impact remain unmeasured in this tranche.

## Learnings

- Top-k MTP must be represented as tree nodes plus an ancestor mask. Treating
  top-k as diagnostics on a top-1 chain loses valid branch hits; verifying rows
  by replaying paths would be the wrong cost model.

## Delta vs baseline

- **Baseline:** `docs/experience/wins/2026-06-18-dsv4-chain-verify-batched-sparse-pending.md`
- **Delta:** pending remote.
