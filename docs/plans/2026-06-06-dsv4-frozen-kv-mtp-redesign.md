# DSv4 MTP — FROZEN-KV redesign (SGLang's approach; un-kills the s_q=K "fundamental" conclusion)

**Date:** 2026-06-06. **Status:** redesign spec. **The s_q=K kill was wrong** — it
came from (a) a WRONG implementation (I re-ran the compressor during verify) and
(b) a WRONG workload (synthetic 64-tok decode, not GSM8K/ShareGPT). SGLang does
native MTP on exactly this sparse attention and it works.

## What SGLang actually does (`frozen_kv_mtp_*`, `dsa_mtp_fixture.py`)

- **"The assistant reads target KV only"** (`frozen_kv_mtp_worker.py:16`). The draft
  + verify read the **frozen** target KV — they do **NOT** re-run the compressor /
  re-compute the sparse selection during draft/verify. `frozen_kv_target_view`
  swaps the attention to the frozen target pool.
- **DSA = DeepSeek Sparse Attention = DSv4's CSA/HCA.** `dsa_mtp_fixture.py`:
  `model=deepseek-ai/DeepSeek-V3.2`, `speculative_algorithm=EAGLE`,
  `num_steps=3, eagle_topk=1, num_draft_tokens=4`, **`accept_length_thres=2.7`** —
  ~2.7 tokens/step on GSM8K/ShareGPT-class workloads.
- **`skip_topk`** (`deepseek_v2.py:1556`, `dsa_indexer.py:1359`): the sparse top-k
  selection is **reused across layers** (and is skippable when `kv_len ≤ index_topk`)
  — the prepare-chain is computed ~once, not per-query-per-layer.

## Why frozen-KV fixes BOTH of my kill reasons

My s_q=K verify ran `dsv4_compressor_update` on the draft tokens → re-compressed
mid-batch → (1) diverged from autoregressive at compression boundaries, (2) paid the
prepare-chain K×. **Freezing the compressor kills both:**
- **Correctness**: with the compressor frozen at the committed prefix, a K-draft span
  that does not cross a compression boundary is EXACTLY autoregressive (the compressed
  blocks are identical; only the SW ring — which IS appendable/causal — grows). No
  divergence. (Boundary-crossing within the K span is a controlled approximation —
  the draft just misses the new block → slightly lower acceptance there, not wrong
  output, because the verify accepts only what the target confirms.)
- **Perf**: no `dsv4_compressor_update` + a shared/reused `csa_select` → the K-token
  verify pays the prepare-chain ~once, so it AMORTIZES.

## Implementation (DSv4, single-process per-rank; §0.1 granularity)

1. **FREEZE during verify** (`forward_tokens_verify` / the per-layer attention):
   add a `frozen` mode that, for the K draft/verify tokens, **SKIPS**
   `dsv4_compressor_update_cuda` (`attention.rs:3421`) and **reuses** the
   `csa_select` (`attention.rs:3461/3545`) result computed for the committed prefix
   (mirror `skip_topk`: compute the top-k once against the frozen compressed blocks,
   reuse for all K draft queries — they are consecutive positions selecting from the
   SAME frozen set). The `dsv4_hybrid_attention` reads frozen compressed blocks + the
   SW ring (with the draft K appended via `update_bf16_sw_window` — that stays).
2. **Chain draft** (`num_draft_tokens=4`, like SGLang): `mtp_forward` generates a
   chain of K tokens, each reading the FROZEN KV (it already reads `h_prev`; ensure
   it does not mutate the compressor). topk=1 (a linear chain, not a tree) to start.
3. **COMMIT on accept**: after the verify accepts the longest matching prefix, run
   `dsv4_compressor_update` for the ACCEPTED tokens (catch the frozen compressor up
   to the new committed length). Rejected drafts never touched the compressor.
4. **ROLLBACK collapses**: A1's compressor/indexer snapshot is **removed** (the
   compressor is frozen, never mutated mid-verify → nothing to revert). Only the SW
   ring slots the rejected drafts wrote need handling — single-slot revert (A1's
   `capture_sw`/`restore_sw`, kept) or rely on overwrite self-heal. The
   `fp8_kv_pool` SW slot likewise. This is far simpler + faster than A1.

## Workload + gate (the OTHER half of the kill — test the SLO shape)

- **Workload: GSM8K / ShareGPT-class**, NOT a synthetic 64-tok decode. Long real
  decodes are where MTP amortizes (acceptance high, decode long). Target
  accept_length ≈ 2.7 (SGLang's DSA number).
- **Correct-inference gate** (needle + same-config-twice floor + coherence on the
  real prompts) — NOT byte-identity. The frozen-span is exactly autoregressive
  off-boundary, a controlled approximation on-boundary.
- **Perf A/B**: spec-ON (frozen-KV, num_draft=4) vs spec-OFF on the GSM8K/ShareGPT
  workload. Report decode tok/s + accept_length. Expected: spec-ON FASTER (the whole
  point), ~(accept_length)× toward SGLang's 2.7×.

## Default-on (the requirement)

ckl: "投机解码功能必须默认有并且默认好用." Once frozen-KV MTP is correct +
amortizing on the real workload, flip it **default-ON** (opt-OUT
`ARLE_DSV4_SPEC_DECODE=0`), matching SGLang where spec decode is a first-class
default. This supersedes A1's default-off per-token verify and the killed s_q=K.

## Sequence

Un-kill the s_q=K errors entry (wrong impl + wrong workload). Implement frozen-KV
(freeze compressor + reuse selection + chain draft + commit-update). Gate on
GSM8K/ShareGPT. Flip default-on. Then depth>1 tree (topk>1) for higher acceptance.
