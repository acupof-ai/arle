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

### Follow-up speed rerun after the "why slower" review

The first speed read above was a short helper run and was not comparable to the
historic 53 tok/s B=1 entry. I restored `sglang-eic-test` on node 62 and reran
same-binary/same-env checks on current `origin/main` (`ee0ddb8d`, which includes
the fix commit as an ancestor).

Startup confounder found and removed first:
- Initial no-MTP startup sat for 600s, and an MTP `num-slots=64` retry sat for
  >210s, both with workers alive and CPU busy.
- `/data01/deepgemm-warm/locks` contained 16 stale lock files from 2026-06-16
  while the cache contained 16 cubins. Removing only the stale lock files (not
  the cubin cache) made no-MTP and MTP `num-slots=16` both ready in 25s.
- This means the failed startups were pod/cache hygiene contamination, not a
  top-k verifier-shape issue.

Fixed synthetic prompt, `max_tokens=128`, `num-slots=16`, no errored or
zero-token requests:

| arm | c1 agg tok/s | c2 agg tok/s | c4 agg tok/s | c8 agg tok/s | bad verify rows | avg accepted | avg tree_hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-MTP | 34.74 | 34.75 | 34.82 | 54.34 | n/a | n/a | n/a |
| D2/T1 | 21.96 | 21.97 | 21.97 | 33.71 | 0/1224 | 0.764 | 0.764 |
| D2/T2 | 21.96 | 21.95 | 21.94 | 33.65 | 0/1224 | 0.764 | 0.913 |

ShareGPT small sample, 8 first-turn prompts from
`/data01/mashisong/ShareGPT_V3_unfiltered_cleaned_split.json`, c=1,
`max_tokens=128`, no errored or zero-token requests:

| arm | success | median tok/s | mean tok/s | bad verify rows | off-chain hits | avg accepted | avg tree_hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-MTP | 8/8 | 34.46 | 31.13 | n/a | n/a | n/a | n/a |
| D2/T1 | 8/8 | 24.66 | 24.85 | 0/516 | 0 | 0.981 | 0.981 |
| D2/T2 | 8/8 | 24.64 | 24.63 | 0/516 | 99 | 0.981 | 1.219 |

Artifacts:
- `/data01/arle-speed-ee0ddb8d/topk-ab-slots16-lockclean-20260617_142651/`
- `/data01/arle-speed-ee0ddb8d/no-mtp-lockclean-20260617_143115/`
- `/data01/arle-speed-ee0ddb8d/sharegpt-topk-20260617_143342/`

Verdict: `topk=2` now has the intended constant verifier row cost
(`verify_rows=3` for D2), and its speed is effectively identical to `topk=1`.
The speed regression versus no-MTP in these reruns is not caused by extra
top-k compute; it is caused by acceptance being below break-even. `topk=2`
does increase off-chain `tree_hits`, but the current chain-shaped verifier
cannot commit those off-chain paths as accepted KV rows because their prefixes
were not target-verified. So `--mtp-draft-topk 2` is correctness-safe and
generic, but it is not a speed lever until the branch-hit path is paired with a
commit-safe verification/rollback design.

## Rule

For MTP top-k, state the tensor shape before coding. "One verify forward" is
not the invariant; "verifier rows stay `depth + 1`" is the invariant for this
matrix-topk design.
