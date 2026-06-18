# DSv4 Chain Verify Sparse Restore — pending remote bench, cuda, 2026-06-18

## SLO-shape probed? N — pending remote DSv4 pod

No throughput verdict in this entry. The local machine verified Rust/no-cuda
shape and call-site correctness only; DSv4 wall-clock bench is pending because
the previously used node 61/62 services are explicitly out of scope.

## Roofline check

Deferred — no CUDA trace/bench was run in this checkout.

| Op | Achieved | Peak (this HW) | % | Verdict |
|---|---:|---:|---:|---|
| chain verify FlashMLA sparse attention | pending | pending | pending | deferred: needs non-61/62 DSv4 pod |

## Goal

Restore the DSv4 MTP chain verifier's batched sparse-attention lane without
reintroducing complete top-k tree verify rows.

## Hypothesis

D2/T2 should stay at `depth + 1 = 3` target rows, while attention should run one
FlashMLA sparse verify call per slot/layer instead of one MLA call per row/layer.

## Command

Not run. Required remote command shape:

```bash
scripts/bench_guidellm.sh dsv4-chain-verify-sparse \
  --target http://localhost:8000 \
  --model <DSv4 model> \
  --processor <DSv4 processor>
```

## Environment

- **Backend:** cuda
- **Model:** DSv4-Flash
- **Hardware:** pending non-61/62 DSv4 pod
- **Commit:** pending
- **Feature set:** pending remote CUDA build
- **Non-default flags / env vars:** pending
- **Profiling state:** pending; must be OFF for throughput baseline
- **Server launch:** pending

## Results — local verification

| gate | result |
|---|---|
| `CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release --no-default-features --features cuda,no-cuda spec_decode --lib` | pass, 6/6 |
| `CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | pass |
| `CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings` | pass |
| `which nvcc` | `nvcc not found`; CUDA csrc compile pending remote |

## Problems

- Remote DSv4 speed bench not run in this turn because node 61/62 must not be used.

## Learnings

- Chain-only MTP still needs prefix-ancestor metadata so FlashMLA can verify the
  whole `[pending, d0, ...]` chunk once. Removing that metadata silently falls
  back to per-row attention and turns D2 into three MLA attention launches per
  layer.

## Delta vs baseline

- **Baseline:** pending remote rerun with the latest known DSv4 bench envelope.
- **Delta table:** deferred until remote bench.

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in the code since baseline: restored chain prefix-ancestor
  metadata and the batched FlashMLA sparse verify lane; did not restore complete
  top-k tree verify rows.
- Follow-ups: run DSv4 ShareGPT/guidellm bench on a clean non-61/62 pod.
