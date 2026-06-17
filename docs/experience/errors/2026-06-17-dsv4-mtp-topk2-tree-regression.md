# DSv4 MTP top-k was implemented as the wrong verifier shape

## Context

`--mtp-draft-topk K` was first implemented as a complete verifier tree:
`K=2, depth=2` drafted and verified `root + 2 + 4 = 7` rows. That passed
correctness, but it was not the intended shape.

Correct shape:
- Draft MTP produces a `[depth, vocab]` logits matrix along one top-1 chain.
- Each draft level extracts top-k candidates from that matrix.
- The target model still verifies only the top-1 chain tokens:
  `[pending, d0, d1, ...]`.
- Verify rows therefore stay `depth + 1`; D2/T2 must be 3 rows, not 7.
- Top-k hits off the verified chain can become the divergence bonus/pending
  token, but they cannot be folded into KV as accepted deep branch rows because
  those prefixes were not target-verified.

## Root Cause

I conflated "one target forward" with "constant row cost". The first
implementation did run one target forward, but it fed the complete flattened
tree into that forward. On DSv4 every extra verify row still runs attention,
MLP/MoE, HC, and lm_head work, so D2/T2 grew verifier work from 3 rows to 7
rows.

The concrete wrong code was:
- `DraftTree` / `SpecShape` in `executor/spec_decode.rs`.
- `complete_tree_nodes(depth, topk) = 1 + K + K^2 + ...`.
- `SpecVerifySchedule { positions, restores, saves }` plus per-node ring
  scratch.
- `forward_tokens_verify_scheduled(..., &tree.tokens, ...)`, which made the
  logits matrix size equal to the flattened tree node count.

## Fix

Delete the complete-tree verifier path. Keep `--mtp-draft-topk`, but make it
only widen draft candidate extraction:
- `DraftChain { tokens, candidates }` stores the top-1 chain plus per-level
  top-k candidate rows.
- `forward_tokens_verify_scheduled` receives only `chain.tokens`.
- `SpecVerifySchedule` is chain positions only.
- `spec_nodes` and spec-node save/restore helpers are deleted.
- Commit-fold scratch is sized back to `MAX_SPEC_DRAFT_DEPTH + 1`.
- D2/T2 unit coverage asserts `verify_rows == 3`.

## Verification

Local:
- `CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` passed.
- `CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release --no-default-features --features cuda,no-cuda spec_decode --lib` passed, 6 tests.

Remote:
- Node 61 (`sglang-bench-61`) cannot run pods: kubelet reports
  `DiskPressure=True`; even a privileged no-GPU cleanup pod is rejected with
  `Pod was rejected: The node had condition: [DiskPressure]`.
- Fallback node 62 used a clean `/data01/arle-build` cloned from the local git
  bundle for this fix commit.
- Remote clean build passed:
  `CARGO_NET_OFFLINE=true CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090 ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 ARLE_DSV4_EXPERT_BACKEND=deepgemm ARLE_DSV4_MOE_BACKEND=allreduce FEATURES=cuda,nccl PROFILE=release-fast BIN=arle bash scripts/dsv4_fast_build.sh`.
- Remote unit gate passed:
  `CARGO_NET_OFFLINE=true CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090 ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 ARLE_CUDA_KERNEL_SET=dsv4_flash ARLE_CUDA_KERNELS_PREBUILT_DIR=/data01/arle-build/target/dsv4-cuda-kernels-prebuilt cargo test -p infer-cuda --profile release-fast --features cuda,nccl spec_decode --lib`
  (`6 passed`).
- Binary string gate: new `dsv4-mtp` log contains `verify_rows` and
  `tree_hits`; old `dsv4-mtp-tree` string is absent.
- Full DSv4 D2/T1 smoke passed on node 62:
  `--spec-type mtp --mtp-draft-tokens 2 --mtp-draft-topk 1`; one completion
  request returned 24/24 completion tokens, and serve log lines showed
  `depth=2 topk=1 draft_rows=2 verify_rows=3`.
- Full DSv4 D2/T2 smoke passed on node 62:
  `--spec-type mtp --mtp-draft-tokens 2 --mtp-draft-topk 2`; one completion
  request returned 24/24 completion tokens, and serve log lines showed
  `depth=2 topk=2 draft_rows=2 verify_rows=3`.
- Short real-prompt bench, 8 prompts, `max_tokens=96`, no zero-token or errored
  requests:

| arm | c1 seq success | c1 agg tok/s | c4 success | c4 agg tok/s | bad verify rows |
|---|---:|---:|---:|---:|---:|
| D2/T1 | 8/8 | 24.25 | 8/8 | 27.01 | 0/783 |
| D2/T2 | 8/8 | 24.25 | 8/8 | 26.64 | 0/788 |

Artifacts on node 62:
- `/data01/arle-test-30290e07/d2_t1_bench_bench.json`
- `/data01/arle-test-30290e07/d2_t1_bench_serve_20260617_122545.log`
- `/data01/arle-test-30290e07/d2_t2_bench_bench.json`
- `/data01/arle-test-30290e07/d2_t2_bench_serve_20260617_122732.log`

D2/T2 had 137 `tree_hits > accepted` lines: top-k matrix hits occurred off the
verified top-1 chain, but the verifier rows still stayed 3 and those off-chain
hits were not committed as accepted KV rows.

## Rule

For MTP top-k, state the tensor shape before coding. "One verify forward" is
not the invariant; "verifier rows stay `depth + 1`" is the invariant for this
matrix-topk design.
