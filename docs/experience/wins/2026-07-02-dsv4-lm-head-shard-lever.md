# DSv4 lm_head vocab shard (`ARLE_DSV4_LM_HEAD_SHARD`) — STUB, pending-remote

**Status: pending-remote** — the lever is CUDA/TP>1-only; no local Mac run is
possible. Commissioned by
[`docs/plans/2026-07-02-dsv4-6ms-token-plan.md`](../../plans/2026-07-02-dsv4-6ms-token-plan.md)
lever **H3** (#99). Off by default; TP=1 and knob-off are byte-identical to
baseline (loader + sampling both dispatch on `lm_head_shard: Option<_>`).

## Context

Replicated lm_head runs the full `[vocab=129280, hidden]` FP8 GEMV on every
rank every token (`dsv4.rs::lm_head_project`). `ARLE_DSV4_LM_HEAD_SHARD=1`
row-shards the head: 128-aligned contiguous vocab slices padded to a uniform
16256 rows/rank at TP=8 (~8× less head-weight HBM read per rank), then:

- greedy: local argmax over the real rows + ONE 8-byte host all-gather of
  `(max_value, global_index)` per rank; the merge
  (`infer_plan::merge_vocab_shard_argmax`) is exact vs the replicated device
  argmax (lowest-index tie rule, matching `sampling.cu`).
- non-greedy (or `ARLE_PROBE_TOKEN_ENTROPY=1`): bf16 logits all-gather
  (vocab/N per rank) into full vocab, then the unchanged host sampler.

MTP spec decode and the token-entropy probe are refused loudly at load (the
MTP verify/draft heads consume full-vocab batched lm_head logits);
`lm_head_project_batch` also hard-errors under the shard.

## Pod A/B required before any license (matrix rows)

Same binary, same session, one env flip (`ARLE_DSV4_LM_HEAD_SHARD` 0↔1), MTP
OFF in both arms (the knob refuses MTP), TP=8, 8×H20, DSv4-Flash-FP8:

| Row | Config | Metric |
|---|---|---|
| 1 | knob=0, greedy, B=1, 2k prompt, ≥512 decode | ms/token (`/v1/stats` delta) |
| 2 | knob=1, greedy, B=1, same shape | ms/token + Δ% vs row 1 |
| 3 | knob=0 vs 1, non-greedy (temp>0), same shape | ms/token Δ% (gather path) |
| 4 | needle gate x3 same-config repeats, knob=1 | exact-retrieval = baseline envelope |

The greedy win bound is ~0.1–0.25 ms/token of head GEMV read minus one extra
tiny collective (+2 host round-trips); the A/B decides sign — no invented
number here. Correctness gate = needle ladder (bench-spec §7 / lever-gate),
NOT byte-identity.

## What Worked / Learnings

(pending-remote — fill from the pod run)

## Rule

A replicated per-rank GEMM whose output is consumed by a rank-symmetric
reduction (argmax/sample) can be sharded with an O(ranks) merge collective —
but the merge must reproduce the device kernel's tie semantics exactly
(`sampling.cu` resolves ties to the LOWEST index; `infer_plan::argmax_logit`
resolves to the highest — the device kernel is the parity reference).
