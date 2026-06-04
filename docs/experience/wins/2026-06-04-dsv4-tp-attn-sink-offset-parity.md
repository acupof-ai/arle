# DSv4 multi-GPU bf16 greedy parity CLOSED — TP attention-sink offset fix

**Status:** PASS — 3/3 prompts exact 16/16 vs legacy bf16 oracle on H20 TP=8/EP=8.
**Track:** R6 clean-CUDA DSv4-Flash (`crates/infer-cuda`), branch `arch/ideal-inference-engine`.
**SKU:** H20 8×sm_90a, CUDA 12.9, DeepSeek-V4-Flash, bf16 (`--kv-cache-dtype auto`), MAX_NEW=16.

## Context

After the prompt-id confounder was resolved
([`errors/2026-06-04-dsv4-parity-prompt-id-confounder.md`](../errors/2026-06-04-dsv4-parity-prompt-id-confounder.md)),
a residual gap remained: of three test prompts, two diverged from the legacy bf16
oracle a few tokens in (e.g. the hash prompt flipped at token 3), one matched.
This was initially filed as plausible small-magnitude bf16 decode noise. It was not.

A layer-bisect on H20 (per-layer hidden-state dump, legacy-bf16 vs rewrite-bf16)
localized the FIRST divergence to the **attention output on non-zero TP ranks** —
not MoE, not hyperconnections, and absent on rank 0.

## Root Cause

The per-head attention sink (`attn_sink`, one logit per attention head) is loaded
**whole on every rank** (`loader.rs` `load_dsv4_vec` — no TP slice). But the SW /
hybrid attention kernels were launched with `sink_offset = 0` hardcoded
(`attention.rs`, the SAFETY comment even flagged it as "EP/TP head sharding is a
multi-rank follow-up"). Legacy passes `tp.rank * local_heads`.

So under TP=8, rank `r` owns global heads `[r*local_heads, (r+1)*local_heads)` but
indexed `attn_sink[0..local_heads]` — every non-zero rank applied **rank 0's** sink
logits to its own heads. The error is per-head and small, so:

- **single-GPU (rank 0 only) is unaffected** → it hid through all single-GPU work;
- **multi-GPU** surfaces it as a small head-dependent perturbation that flips tight
  argmax margins at *some* token positions on *some* prompts — exactly the
  "prompt-dependent, 2/3 diverge, small margin" signature that looked like noise.

## What Worked

Thread the TP rank into the attention launch (FFI already exposed `sink_offset`):

- `mla_attention(...)` gains a `tp_rank: usize` param; computes
  `sink_offset = tp_rank * local_heads`;
- passes `sink_offset as i32` to both `dsv4_swa_attention_cuda` and
  `dsv4_hybrid_attention_cuda` (was `0`);
- the `attn_sink.len` guard now requires the whole vector covers this rank's slice
  (`>= sink_offset + local_heads`);
- the model layer-loop passes `self.tp.config().rank` (the TP rank — cleaner than the
  GPU hotfix's `ep_rank`; identical value under the TP=8/EP=8 mirror layout).

Kernel + loader unchanged; this is a launch-scalar fix only.

**Verification (H20, TP=8/EP=8, clean rebuild, no debug dump):**

| Prompt | ids | Match |
|---|---|---|
| In computer science, a hash table is | 1124,6341,6262,14,260,19657,4184,344 | **16/16** |
| The largest planet in our solar system is | 671,9152,13540,295,1132,11250,1487,344 | **16/16** |
| A concise recipe for pancakes begins with | 35,47468,17144,362,90246,12600,418 | **16/16** |

First divergence: none, all three. The DSv4 multi-GPU bf16 greedy correctness gate
is closed; the earlier "diverge = bf16 noise" hypothesis was wrong — the fix yields
deterministic exact parity.

## Rule

- **A whole-loaded per-head/per-expert vector MUST be indexed by `rank * local_count`
  under TP/EP.** A hardcoded `offset = 0` silently corrupts only non-zero ranks —
  invisible on single-GPU, surfacing multi-GPU as prompt-dependent token flips.
- **"Prompt-dependent + small + only some prompts" diverging on multi-GPU is a
  per-rank indexing/offset bug suspect, NOT bf16 noise.** Don't accept "small bf16
  decode noise" for a multi-GPU gap until a rank-0-vs-rank-N comparison rules out a
  sharding offset. Layer-bisect localizes it cheaply.
- A SAFETY comment that admits a deferred gap ("multi-rank follow-up") is a live bug
  marker — grep for them before declaring a path verified.
