# DSv4 FFN all-reduce overlap is implemented, pending remote A/B

## Context

The DSv4 all-reduce MoE path pays `ffn_all_reduce` before the shared expert.
The intended first cut is an env-gated overlap path:
`ARLE_DSV4_COMBINE_OVERLAP=1` enqueues the post-routed FFN all-reduce on the
communicator stream, then lets `ffn_shared` run on the compute stream before
waiting on the routed fence at the final add.

## What Worked

- Code landed in commit `99442f28c3e158b1afba763dfc192bab261802ca`.
- The default path remains unchanged when `ARLE_DSV4_COMBINE_OVERLAP=0`.
- Local checks passed:
  - `cargo fmt --check`
  - `git diff --check`
  - `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
  - `cargo check -p infer --no-default-features --features no-cuda`
- Remote `/data01/build/arle` could not fetch from GitHub during the session, so
  the exact two-file patch was applied directly after verifying both remote
  source blobs matched the local parent commit.

## Pending Remote

The required A/B was not run yet. During the remote build attempt, a separate
SGLang server appeared on the pod and occupied all 8 H20 GPUs, so ARLE service
validation would have contended with someone else's workload. The in-progress
Cargo/CUDA compile was stopped, including orphan `nvcc`/`cicc` children.

Required gate before keeping the perf claim:

| Gate | Required result |
|---|---|
| EOS output | Normal EOS request returns sane decoded text |
| 32-token decode | `max_tokens=32` decode returns sane generated text |
| request trace | Trace contains `ffn_all_reduce` for off and `ffn_all_reduce_overlap_enqueue` for on |
| TPOT | `ARLE_DSV4_COMBINE_OVERLAP=1` must recover real ms/token vs `=0` under matched warm workload |

## Rule

Do not claim an overlap win from source structure alone. The overlap path is
licensed only by same-binary, same-workload `ARLE_DSV4_COMBINE_OVERLAP=0/1`
remote A/B with correctness and request-trace evidence.
