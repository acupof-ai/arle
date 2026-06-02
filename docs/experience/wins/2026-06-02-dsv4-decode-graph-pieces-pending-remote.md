# DSv4 Decode Graph Pieces Pending Remote Gate

Date: 2026-06-02

## Context

This entry covers the DSv4 CUDA decode changes in
`infer/src/model/deepseek/`: piecewise graph capture for stable input/head
decode regions and reused local-routed MoE scratch.

Local verification is deferred because this Mac cannot run the CUDA DSv4
target. This is a `pending-remote` gate, not a performance win.

## Goal

Validate that DSv4 piecewise decode graph replay and local-routed scratch reuse
are reachable and do not regress the SGLang-comparison workload before any
throughput claim or default decision.

## Hypothesis

Capturing the stable decode input/head pieces and reusing local route buffers
should reduce repeated launch/allocation overhead in the DSv4 SGLang-profile
decode path. The claim is only licensed by same-binary remote A/B evidence.

## Commands

Pending remote on the DSv4 host:

```bash
cargo build --release --features cuda
ARLE_BIN=/data01/build/arle/target/release-fast/infer \
SGLANG_DIR=/workspace/sglang@0d51db3 \
./scripts/dsv4_beat_sglang_bench.sh run-all
```

## Environment

- Status: `pending-remote`.
- Backend: CUDA DSv4.
- Expected shape: DeepSeek-V4 Flash, 8xH20 TP=8, ISL=1024, OSL=512.
- Script profile: `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`,
  native-DeepEP MoE, DeepGEMM experts, FP8 KV, CUDA graph max batch 16.

## Results

Pending remote. No TTFT, TPOT, throughput, or default-flip claim is made by
this entry.

## Problems

- Local Apple Silicon cannot run the CUDA DSv4 benchmark.
- The CUDA graph change must be checked on the actual remote binary; source
  reachability alone is not evidence.

## Learnings

The code path is intentionally gated by a remote benchmark requirement. Treat
the local commit as a fix/staging step until remote numbers exist.

## Rule

For DSv4 decode graph or scratch-pool changes, do not claim a win from source
survey or compilation alone. Run same-binary remote A/B and report TTFT, TPOT,
and throughput together.
