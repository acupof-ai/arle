# DSv4 TP=8/EP=8 token-1 greedy parity PASS (rewrite == legacy oracle)

## Context

R6 clean-CUDA DSv4-Flash forward (MLA + CSA/HCA + hyperconnections + hash routing
+ FP8 MoE + native experts + TP all-reduce), 8×H20 sm_90a, TP=8/EP=8. First
multi-GPU correctness gate for the rewrite's DSv4 port. Run on the pod
(`/data01/build/rewrite-dsv4-9def-target`, `dsv4_parity` example via
`scripts/dsv4_multigpu_parity.sh`); `pending-remote` (no local CUDA).

## What Worked

With the corrected DeepSeek prompt `671,6102,294,8760,344` ("The capital of France
is"), all 8 ranks loaded (~19.6 GB/rank) and rank-0 produced:

```
[dsv4-parity rank=0] prefill argmax (token #1) = 11111
clean_tokens=[11111]
```

`11111 = ' Paris'` = the legacy oracle's first token, exactly. The full-prefix
prefill argmax exercises the **entire** forward stack over all 61 layers and every
layer type (SW / CSA / HCA all recompute attention over `[0, len)`), through the
FP8 MoE and the TP all-reduce — so a token-1 match is a strong whole-stack
correctness signal, not a shallow one.

Enabling fixes (all mirrored to the repo): NCCL file-rendezvous
(`INFER_NCCL_ID_FILE`, `e91cf0da`), MTP-tolerant config load (`7a7bd70d`), launcher
`INFER_CUDA_DEVICE=0` under per-rank GPU mask (`3889ed5d`), and the corrected
prompt ids (`a882823b`). The "native bypass numerically broken" conclusion that
preceded this was a prompt-id confounder — see
`errors/2026-06-04-dsv4-parity-prompt-id-confounder.md`.

## Rule

Prefill-argmax token-1 parity across all layer types is a strong forward-stack
correctness signal but **not** full verification. It does not exercise the
incremental-decode (`start_pos > 0`) KV-retention path — the harness bails after
token 1 because DSv4 reallocates its SW ring caches per `forward_tokens` call — so
the full 16-token oracle and multi-prompt/multi-shape parity remain the gates that
close DSv4 correctness. The DeepGEMM production expert backend
(`cuLibraryGetKernelCount` multi-rank) is decoupled: a perf/goal item, not a
correctness blocker, since the native path is already token-1-correct.
