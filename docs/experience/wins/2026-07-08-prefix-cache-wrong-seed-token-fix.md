# Full Prefix-Cache-Match Wrong-Seed-Token Fix

> Status: Shipped
> Date: 2026-07-08
> Env: 4×H20 (GPUs 2/3/4/5), TP4, DeepSeek-V4-Flash-FP8, FlashMLA on, prefix cache ON (default)

## Context

Closes the root cause found in
[`docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md`](../errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md),
"Layer-0-15 residual bisection" section: repeating the exact same prompt
against a warm server was 100% deterministically corrupted from the 2nd call
onward (`738291` → `738292`), independent of the already-fixed #8 CUDA-graph
page-table bug and independent of concurrency (n=1 solo repeats reproduced it
too).

## Root Cause

A full-prompt RadixCache match (`matched_len == prompt_len`) jumped straight
to `RequestPhase::Decoding` with `generated_tokens` empty — no forward pass
ever ran to sample a genuine first token. `planner.rs`'s decode-row builder
then silently fell through:

```rust
let Some(last_token) = request.generated_tokens.last().copied()
    .or_else(|| request.prompt_tokens.last().copied())
else { continue };
```

feeding the prompt's own final token (`</think>`, DeepSeek's forced
non-reasoning marker) as the decode seed — duplicating it into KV and
shifting every subsequent RoPE position + generated token by one slot for
the rest of the generation. This explained every prior observation:
100% reproducibility, why the first 3-4 digits usually survive (a 1-token
shift the model absorbs for a few steps), and why later near-tied digits
flip.

## Fix

`crates/infer-core/src/prefix.rs`: on a full-prompt match, trim the last
matched block so the tail always re-prefills through standard
chunked-prefill (which samples the real first token from logits) instead of
skipping straight to `Decoding` — applied to both the radix-page attach path
(`attach_prefix_to_request`) and the DSv4 position-0 image-restore path
(`attach_cached_prefix`).

`crates/infer-core/src/lib.rs`: added `RequestState.used_prefix_restore`,
set whenever any of a request's KV came from a restore. Gates
`capture_cached_prefix`/prefix-cache publish off for such requests — a
partially-restored slot's snapshot isn't proven equivalent to a genuine
from-scratch capture, and empirically corrupted the *next* restore if
published (a derivative-of-derivative image).

## Verification (H20 pod, 2026-07-08)

Build: `cargo build --release --features cuda,nccl --bin arle` on 4×H20 —
`BUILD_EXIT=0`. Boot: `INFER_CUDA_DEVICES=2,3,4,5 INFER_TP_SIZE=4`,
`ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_EXPERT_BACKEND=deepgemm`, `--max-total-tokens 2048`.

- **Established repro, `trace_probe.py` solo (n=1), 12 reps of the
  byte-identical TRACKED prompt:** 12/12 exact `738291`. Before the fix this
  was call-1-correct, calls-2+ deterministically `738292` (near-100%
  reproducible per the errors doc). Fixed.
- **Fresh unique-content, no cache hit (`matched_len == 0`):** 2/5 exact,
  3/5 truncation/hedging misses. Not a regression — this code path never
  reaches the changed branch (`prefix_match.is_empty()` short-circuits
  before it), and the errors doc's own "Part B" round already measured a
  ~6% (up to 20-33% in earlier, contamination-affected sweeps) pre-existing
  solo miss rate on this exact hard near-tied-digit needle task, unrelated
  to this fix and still open.
- **Genuine partial match (`matched_len < prompt_len`, 2-turn: shared
  prefix + new tail content):** exact `738291`. The `matched_len ==
  prompt_len` gate does not fire on this path — unaffected, still correct.
- **#8 composition check** (same prompt sent 3× via
  `/v1/chat/completions`, per
  [`2026-07-07-prefix-cache-graph-page-table-fix.md`](2026-07-07-prefix-cache-graph-page-table-fix.md)'s
  own recipe): 3/3 byte-identical `reasoning_content`, no corruption —
  composes cleanly with the already-landed #8 fix.

## Rule

- A full-prompt cache/restore match must still run one genuine forward+sample
  step before entering `Decoding` — an empty-`generated_tokens` `Decoding`
  state is itself the bug signal, never a valid steady state to paper over
  with a `.or_else` fallback.
- A restore-derived slot's snapshot is not automatically equivalent to a
  from-scratch capture; publishing it unconditionally lets one restore's
  imperfection propagate into the next.
- Scope: this fixes the RadixCache/DSv4-image full-match repeat-corruption
  class only. The errors doc's original concurrent (n≥3, unique-content)
  digit-corruption bug and the pre-existing ~6% solo near-tied-digit miss
  rate remain open, separate, unattributed issues.
