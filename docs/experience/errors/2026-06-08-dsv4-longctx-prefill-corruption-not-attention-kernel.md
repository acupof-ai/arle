# DSv4 long-context garbage is shared-prefill corruption, not the attention kernel — and the parity "garbage" was a harness artifact

**Date:** 2026-06-08. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** root cause narrowed (shared prefill forward, length-dependent
regression) but not yet pinned to a commit/kernel. Bug 2 (prefix cache) fixed
separately — see `wins/2026-06-08-dsv4-qwen3moe-prefix-cache-recurrent-kv-fix.md`.

## Context

The 900K needle was reported blocked by "pre-existing DSv4 DSA-active decode
garbage, verified by the parity example exhibiting it too." Re-investigating on a
clean serve with controlled experiments.

## Root cause (what the evidence actually shows)

**Wrong framing #1 — "parity proves decode is broken."** The `dsv4_parity`
example's incremental decode is a **harness artifact**: it reallocates the SW ring
caches per `forward_tokens` call (its own doc, `dsv4_parity.rs:14-21`), so its
decode steps cannot see prior-step KV. The serve path keeps persistent per-slot
state (`Dsv4SlotState`, reset only at `start_pos==0`, `executor.rs:1215`), so the
parity garbage is **not** evidence about the serve. Independent confirmation
needed — and obtained — on the serve.

**The real bug — length-dependent prefill corruption.** Clean controlled needle
runs on the serve (raw `/v1/completions`, no chat template, greedy `temp=0`):

| prompt tokens | result |
|---|---|
| 5 ("The capital of France is") | `" Paris…"` correct, **deterministic** (×2 identical) |
| ≤ 75 (needle at depth 0) | `hit=True` `" 738291…"` retrieves |
| 86 | partial — emits `"738"` then collapses |
| ≥ 115 | full garbage, **non-deterministic** (run1 ≠ run2) |
| 200, needle at depth **0.9** (next to query, inside SW) | also garbage |

The depth-0.9 failure is the key: the corruption is **position-independent** —
once total length passes ~80–122 tokens the whole prefill output is garbage
regardless of where the needle sits, so it is **not** an attention-reach / SW
eviction problem (`sliding_window=128`, and 115 < 128 already fails). The first
generated token (prefill argmax) is already wrong ⇒ the bug is in **prefill**,
not incremental decode.

**Components exonerated by serve A/B (each a same-binary env flip, single var):**
- **DSA indexer** — `ARLE_DSV4_DSA_INDEXER=0` (legacy bf16) also garbage.
- **FlashMLA** — `ARLE_DSV4_FLASHMLA_PREFILL/DECODE=0` is *worse* (n=4 fails too;
  default FlashMLA-on retrieves n=4). The 2047-needle inverse-rope fix
  (`arle_dsv4_output_inverse_rope_cuda`) lives only in the FlashMLA path
  (`attention.rs:3325,3671`), which is why scalar is worse — but FlashMLA-on still
  fails ≥86, so rope is not the regression.
- **DeepGEMM proj/linear** — `ARLE_DSV4_{PREFILL_PROJ,FP8_LINEAR,DECODE_PROJ}_DEEPGEMM=0`
  is also *worse* (n=4 partial `"738"`). The documented "DeepGEMM skew" suspect is
  not it; the **default (all-on) config retrieves the most**.

⇒ The corruption is in a path **shared** by both attention kernels and both
linear backends — the compressor or the core HC/MoE forward — and it is a
**regression**: `project_dsv4_compressed_attention_longctx_bug` validated needle
11/12 to 2047.

**Wrong framing #2 — "DIFF@122 = precision margin."** Commit `8bcd8ce3`
("FlashMLA closeout … DIFF@122 = precision margin") observed a divergence at
exactly this boundary and dismissed it as precision. It is **catastrophic**
(full garbage / failed retrieval), not a precision margin — a §0 framing trap
(narrow-window dismissal of a real correctness break).

## Fix

Not yet landed. Narrowed to the shared prefill forward; regression window is the
FlashMLA / official-prefill-kernel / DeepGEMM closeout
(`8bcd8ce3 → 38160622 "default official prefill kernels" → 149d7377 → DeepGEMM
levers`). Next: git-bisect that window on the raw-needle repro (n=4 retrieves,
n=20 garbage), or a per-layer `ARLE_DSV4_TAIL_DUMP` diff (good vs bad length) to
find the first diverging layer / its `compress_ratio`.

## Rule

- "Parity exhibits it too" is **not** corroboration when the parity harness has a
  known structural defect (per-call cache realloc). Confirm on the production
  path with a clean controlled repro before attributing a bug to "pre-existing."
- A divergence at a fixed prompt length is **config-suspect first**, but once
  multiple same-binary env A/Bs all leave it (and the all-default path is the
  *best*), it is a real shared-forward bug — stop toggling, bisect or dump.
- Catch greedy degenerate output with the actual decoded text + a determinism
  control (same prompt ×2) and a position control (needle near vs far) before
  calling it "decode broken" — it localized this to prefill, position-independent.
- `arle_serve.sh` inherited stale `infer`-crate env (`ARLE_DSV4_INCREMENTAL_KV`,
  `EXPERT_BACKEND`, `LOCAL_GROUPED_EXPERTS`): **dead in the rewrite** (no readers).
  Verify env knobs have a reader before trusting them in a bench config.
