# DSv4 TP4 topk2 branch path regressed MTP shape and speed

## Context

TP4 DSv4-Flash verification on node 62 found two separate issues:

- TP4 no longer failed the O-LoRA projection shape after `67f4252a`
  (`wo_a.cols=4096`, local attention width `8192`, two output groups per rank).
- The follow-up `topk=2` MTP path was still the wrong shape: the special
  `topk > 1` branch verifier bypassed the depth-2 chain and drafted only one
  root level.

The bad runtime log shape was:

```text
[dsv4-mtp-branch] depth=2 topk=2 draft_rows=1 verify_rows=3 ...
```

That is not D2/T2. It is D1/T2 with a `depth=2` label. It can emit at most two
tokens per spec step (`accepted=1` plus bonus), while a DSv4 MTP step on TP4 costs
about 2.4x a no-spec decode step.

## Root Cause

Commit `35ebbdf9` introduced a `DraftBranch` special case:

- `spec_step()` returned early to `spec_step_branch()` when `topk > 1`.
- `spec_step_batched()` returned early to `spec_step_batched_branch()` when
  `topk > 1`.
- `draft_branch()` called only one `mtp_forward_level`.
- `SpecVerifySchedule::branch_root()` verified `[pending, root_topk...]`.

That made off-chain first-token hits commit-safe, but only by deleting the
second draft level. The code kept `depth=2` for ring restore and reject counts,
so the log looked like D2/T2 while the actual draft matrix had one row.

Correct invariant for the current chain verifier:

- Draft produces a depth-row candidate matrix along one top-1 chain.
- Target verifies only `[pending, d0, d1, ...]`.
- Verify rows stay `depth + 1`; D2/T2 is 3 rows.
- Off-chain top-k hits are diagnostics/free bonus candidates only. They are not
  commit-safe accepted rows unless their prefix was target-verified.

## Fix

Deleted the branch special case and restored the single generic chain path:

- Removed `DraftBranch`, `BranchAccept`, `draft_branch`,
  `spec_step_branch`, and `spec_step_batched_branch`.
- Removed `SpecVerifySchedule::branch_root`.
- B=1 and B>1 now both use `draft_chain(depth, topk)` and verify the chain.
- Removed branch-shape tests; retained tests that off-chain top-k hits are not
  committable and D2/T2 verify rows stay chain-shaped.

Code commit: `70912d7d fix(cuda): restore dsv4 topk chain verify`.

## Verification

Local static gates:

```text
rustfmt --edition 2024 --check crates/infer-cuda/src/executor/spec_decode.rs crates/infer-cuda/src/dsv4.rs
PASS

git diff --check -- crates/infer-cuda/src/executor/spec_decode.rs crates/infer-cuda/src/dsv4.rs
PASS

rg "branch_root|DraftBranch|BranchAccept|dsv4-mtp-branch|branch verifier|branch verify|root_topk" \
  crates/infer-cuda/src/executor/spec_decode.rs crates/infer-cuda/src/dsv4.rs
PASS: no matches
```

Local `cargo test -p infer-cuda ... spec_decode` was blocked by an unrelated
dirty `crates/deepseek-spec/src/v4.rs` change adding
`DeepSeekV4AttentionMode::SparseIndexed`; compile errors were exhaustive-match
failures in `crates/infer-cuda/src/attention.rs`, not in this MTP diff.

Remote clean bundle:

```text
bundle: /tmp/agent-infer-70912d7d-increment.bundle
sha256: 2162764029f6c1848027694e9e85026672dcaeed87f18f98bb43b6c3a507b1d9
pod tree: /data01/arle-gpu-verify-35ebbdf9-tp4
HEAD: 70912d7d
features: cuda,nccl
profile: release-fast
GPUs for serve: INFER_CUDA_DEVICES=0,1,2,3
TP: INFER_TP_SIZE=4
```

Remote build and unit gate:

```text
scripts/dsv4_fast_build.sh
PASS: release-fast binary rebuilt in 10.58s using prebuilt CUDA kernels

cargo test -p infer-cuda --profile release-fast --features cuda,nccl spec_decode --lib
PASS: 6 passed

strings target/release-fast/arle | grep dsv4-mtp-branch
PASS: no branch string; only [dsv4-mtp] remains
```

TP4 same-prompt trace, 64 output tokens:

| arm | engine steps | step sum | steady p50 | MTP shape |
|---|---:|---:|---:|---|
| no-spec baseline | 66 | 1952.1 ms | 26.3 ms | n/a |
| bad topk2 branch (`67f4252a`) | 39 | 2564.4 ms | 63.8 ms | `draft_rows=1 verify_rows=3` |
| fixed topk2 chain (`70912d7d`) | 39 | 2617.7 ms | 64.9 ms | `draft_rows=2 verify_rows=3` |

ShareGPT small sample, 8 prompts, 128 output tokens, TP4 first four GPUs,
`--num-slots 16`, `ARLE_DSV4_MOE_BACKEND=allreduce`,
`ARLE_DSV4_EXPERT_BACKEND=deepgemm`, profiling off:

| arm | c | success | output tok/s | TTFT p50 | ITL p50 |
|---|---:|---:|---:|---:|---:|
| no-spec baseline (`67f4252a`) | 1 | 8/8 | 34.5 | 282.7 ms | 26.3 ms |
| bad topk2 branch (`67f4252a`) | 1 | 8/8 | 27.3 | 269.1 ms | 61.6 ms |
| fixed topk2 chain (`70912d7d`) | 1 | 8/8 | 27.7 | 289.3 ms | 62.8 ms |
| no-spec baseline (`67f4252a`) | 4 | 8/8 | 44.2 | 1210.4 ms | 80.7 ms |
| bad topk2 branch (`67f4252a`) | 4 | 8/8 | 23.9 | 1069.0 ms | 203.9 ms |
| fixed topk2 chain (`70912d7d`) | 4 | 8/8 | 33.4 | 1082.7 ms | 68.4 ms |

Fixed-chain MTP log summary across the trace plus ShareGPT runs:

```text
lines=592
draft_rows: {2: 592}
verify_rows: {3: 592}
branch_lines=0
accepted_dist={0: 186, 1: 267, 2: 139}
candidate_hits_dist={0: 114, 1: 253, 2: 225}
```

Artifacts on node 62:

```text
/data01/arle-gpu-verify-35ebbdf9-tp4/bench-output/sharegpt8_tp4_mtp_d2_topk2_chain_70912d7d.sharegpt.log
/data01/arle-gpu-verify-35ebbdf9-tp4/bench-output/sharegpt8_c4_tp4_mtp_d2_topk2_chain_70912d7d.sharegpt.log
/data01/arle-gpu-verify-35ebbdf9-tp4/bench-output/sharegpt_tp4_mtp_d2_topk2_chain_70912d7d.mtp.log
/tmp/arle-tp4-topk2-chain-trace64-70912d7d.log
```

## Rule

`topk` must widen the draft candidate matrix, not switch the verifier schedule.
For the current chain verifier, off-chain candidate hits cannot be folded into KV
as accepted rows. Any future commit-safe branch acceptance must first state the
target-verified prefix shape and prove the emitted tokens per step can beat the
measured step-cost ratio.

MTP remains explicit. On this TP4 ShareGPT sample, fixed D2/T2 is correct-shape
but still slower than no-spec because the MTP step is about 2.4x a no-spec step
and the realized accepted-token count is below break-even.
