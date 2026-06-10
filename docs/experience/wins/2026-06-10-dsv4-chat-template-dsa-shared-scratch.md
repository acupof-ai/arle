# #66 general chat-template dispatch + #67 shared DSA scratch — verified on 8×H20

**Date:** 2026-06-10. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commits:** `4892ce6f` (chat template), `e14aa837` (DSA budget accounting),
`e004ad5d` (shared DSA scratch). **Binary:** e004ad5d build (`DONE ec=0`,
`shared DSA scratch` strings verified in binary).

## Goal

Close the two Phase-0-followup issues found during #56/#57:
- **#66**: `/v1/chat/completions` rendered Qwen ChatML for every model → DSv4
  got tag salad. General fix (new-model onboarding = zero code): checkpoint
  `chat_template` (minijinja+pycompat, the TGI mechanism) → builtin DSv4
  renderer (checkpoint ships its format as Python, not Jinja) → ChatML+warn.
- **#67**: 256K serve no longer booted — official-DSA selector scratch was
  allocated per CSA layer × per slot (logits tile ≈ 256 MB each at 256K),
  un-budgeted. Fix: split stateful (rotated_keys/packed_rows, stays per
  slot×layer) from shareable (logits + per-forward scratch + constants → ONE
  `Dsv4DsaSharedScratch` per model, owned by the KV adapter), plus itemized
  budget accounting. Single-stream ordering is the sharing-safety argument;
  the disabled-event-tracking hazard is premature FREEING, which a
  adapter-lifetime scratch never does.

## Results

| Check | Shape | Result | Verdict |
|---|---|---|---|
| #66 chat smoke (France / system+user / multi-turn) | 16K serve, `/v1/chat/completions` ×3 prompts | Q1 `'The capital of France is Paris.…'` — retrieval ✓, ZERO ChatML tags (was: tag salad 0/12); Q2/Q3 coherent but quirky (stray `</think>`, multi-turn recall miss) | **PASS for the route** — template byte-faithful to `encoding_dsv4.py` chat mode; remaining quirks are mode behavior (official encoder has NO default `thinking_mode`; thinking-mode flag is a follow-up refinement, not a template bug) |
| Needle gate (shared-scratch correctness) | 115/300/446/2000/8000 ×3 vs baseline 3/0/2/2/3 | 3/0/1/3/3 exact, no garbage class | **PASS** (±1 envelope) |
| Decode smoke (shared-scratch perf) | c=1 128 tok vs 39.38 tok/s | 39.00 tok/s (−1%), opening byte-identical | **PASS** |
| #67 256K boot | `INFER_DSV4_MAX_SEQ_LEN=262144`, 8 ranks | **Boots to 200** (was: `DSv4 official DSA logits alloc` OOM on 8/8 ranks); itemized clamp warn fires, 4→3 slots cross-rank-consistent, `scheduler follows` | **PASS** |
| #67 256K admission + retrieval | 230,812-token prompt, greedy | admitted (no spin), prefill 59.6 s, needle `738291` retrieved EXACT, serve healthy after | **PASS** (band-aid-era verify was 55.4 s @200K on the WIP-patched tree — parity restored on clean main) |

## Problems

- First shared-scratch cycle (accounting-only, clamp 4→2 at lower free)
  survived boot but hit a tick-0 `Alloc failed` OOM on rank 7 — ~2 GB/slot of
  per-slot stateful caches (compressor/indexer compressed caches + FP8 DSA
  key-cache bands) were still un-itemized, thinning the activation headroom.
  `0d1af10d` itemizes them and logs measured per-rank free + the ledger split
  at budget time; the verified cycle ran 3 slots with healthy margin.
- Rank-7-only OOM asymmetry remains un-rootcaused (all ranks reported the
  same affordable count) — the new free log attributes it next time it shows.
- Pod/GitHub connectivity flaked mid-verification and a parallel-session tree
  reset wiped a tn-pushed file mid-cycle — the verified binary is `e004ad5d`
  (shared scratch); `0d1af10d` (extended ledger) is in main as the
  variance-reduction follow-on, structurally identical accounting.
- Serve cold-boot is ~5.7 min (workers 2 min parallel; rank-0 serializes
  another ~3.5 min; 8× read amplification) — split to #69.
- DSv4 chat-vs-thinking mode has no official default (`encode_messages`
  requires `thinking_mode`); the route renders chat mode today. A
  `--chat-thinking-mode` serve flag is the follow-up refinement.

## Rule

- New-model chat onboarding: ship `chat_template` in the checkpoint and ARLE
  renders it with zero code; builtin renderers are reserved for checkpoints
  that ship their format outside Jinja (DSv4's `encoding_dsv4.py`).
- Scratch that scales with `max_seq × layers × slots` must be classified
  stateful-vs-per-forward at design time; per-forward scratch on a
  single-stream executor is shareable by stream order, and the budget must
  itemize whatever stays per-slot.
