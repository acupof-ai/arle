# DSv4 SM90 MegaMoE raw ABI — pending remote

> Status: pending-remote — the raw ABI is not wired into serving yet.

## SLO-shape probed? N

No H20 run in this local tranche; no performance verdict or default flip.

## Roofline check

Deferred until the H20 component A/B. Required counters: fused-kernel TFLOPS,
HBM GB/s, launch count, and wall-clock MoE ms/request.

## Goal

Optimization substrate: expose the vendored SM90 MegaMoE workspace and launch
without Torch, while leaving the serving path unchanged.

## Hypothesis

Replacing per-rank MoE staging with the upstream fused dispatch/L1/L2/combine
kernel will improve TP=4 aggregate throughput; no percentage is licensed yet.

## Command

```bash
# Upstream PR #323 component probe; ARLE raw ABI remains pending.
python3 test_mega_moe_hopper.py --fused-only-sweep --num-processes 4 \
  --num-max-tokens-per-rank 128 --batches 1 4 8 16 \
  --hidden 4096 --intermediate-hidden 2048 --num-experts 256 --num-topk 6

# Pending after serving wiring.
scripts/bench_guidellm.sh dsv4-sm90-mega-moe \
  --concurrencies 1,4,16,64 --max-seconds 120
```

## Environment

- Backend: CUDA, SM90 only
- Model: DeepSeek-V4-Flash
- Hardware: 4x H20, driver 535.161.08, CUDA 12.9
- Upstream: DeepGEMM PR #323 head `9e3afe91cb145ddfa0b18ae874a11dbb449e16a9`
- Source commit: `b94e2fc44` plus the raw-ABI tranche
- Non-default path: SM90 MegaMoE raw ABI, not yet reachable from serving

## Results

The upstream component probe passed on 4x H20; this validates the pinned kernel,
not the ARLE raw ABI or serving path.

| Tokens/rank | MegaMoE us | TFLOPS | HBM GB/s |
|---:|---:|---:|---:|
| 1 | 103.0 | 3.4 | 1467 |
| 4 | 262.5 | 4.8 | 2015 |
| 8 | 322.1 | 6.7 | 2190 |
| 16 | 455.4 | 9.8 | 2821 |

The exact Flash shape matched the independent Torch reference at
`calc_diff=0.0006 < 0.07`. At 16 tokens, the optional DeepEP + grouped-FP8
pipeline took 1.54-1.58 ms versus 0.46-0.52 ms fused (3.06-3.36x), but its
per-128 L2 activation scale differs from MegaMoE's per-64 contract, so it is a
timing control, not the numerical reference.

Local raw-ABI gates passed:

- `CUDARC_CUDA_VERSION=12080 cargo check -p cuda-kernels --release --no-default-features --features cuda,no-cuda --lib`
- `git diff --check`

## Problems

- H20 NVCC/JIT compile, tensor-map construction, symmetric-buffer dispatch, and
  numerical output are unverified.
- Serving has no call site, so guidellm cannot attribute a delta yet.

## Learnings

An unreachable raw kernel ABI is implementation progress, not a throughput win;
license only after same-binary serving A/B and decoded-output correctness.

## Delta vs baseline

Deferred. Use the latest DSv4 TP=4 fixed-concurrency baseline when wiring lands.

## Artefacts

Pod logs: `/host/deepgemm-pr323-probe/t0-exact-shape-max128.log`,
`t0-n16-baseline.log`, and `t0-accuracy-flash-exact.log`.
