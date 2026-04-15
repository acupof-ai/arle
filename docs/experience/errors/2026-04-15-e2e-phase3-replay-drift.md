# 2026-04-15 · e2e Phase 3 replay drift — single-token `forward_prefill` is not numerically consistent with cold batch prefill

## Context

`cargo test -p infer --release --test e2e` on Qwen3-4B had been passing
`test_e2e_generation` Phase 1 (6 cold-prefill baselines) since the HF
`Qwen/Qwen3-4B` Instruct weights stabilised against
`infer/test_data/Qwen3-4B.json`. Phase 3 (stream/non-stream consistency)
had never actually been exercised until Phase 1 stopped failing on baseline
drift, because the test panics at the first failing phase.

On the 2026-04-15 remote CUDA box (L4 24GB, CUDA 13.0, driver 580.82.07,
commit `bfbfd6f` at the time of first reproduction, then `7ca766e` after
the Metal-side pull), Phase 3 iter 0 panicked:

```
assertion `left == right` failed: stream/non-stream text mismatch for: Tell me a story
  left:  " about a young girl who is a talented artist, but she is not allowed to paint because of her gender. …"
  right: " about a young girl named Lila who is a talented painter, and her journey to become a successful artist…"
```

`left` is `non_stream.text` (byte-identical to the baseline) and `right`
is `streamed` (divergent around the 4th generated token onwards).

Phase 3 iter 0 logs, right before the panic:

```
server_engine.rs:559 KV prefix cache MISS: resetting
    ← non-stream call: cached_prompt = last Phase-2 prompt, miss → reset → cold prefill → CORRECT
server_engine.rs:582 KV prefix cache FULL HIT: exact match, replaying final token with 3/4 tokens reused
    ← stream call: cached_prompt = "Tell me a story", exact match → ReplayFinalToken → WRONG
kv_cache.rs:379 KV cache truncated to 3 tokens
```

So only the second call in each Phase 3 pair — the one taking the
`PrefixReuseAction::ReplayFinalToken` branch — produced wrong greedy
tokens. Every other call (Phase 1 non-stream cold, Phase 2 stream cold,
Phase 3 non-stream cold) produced the correct output.

## Root Cause

`ModelInferenceEngine::prepare_with_prefix_cache` (the single-request
engine used by the agent CLI REPL and the e2e tests) dispatched exact
cached-prompt matches to `PrefixReuseAction::ReplayFinalToken`, which
then ran:

```rust
self.state.truncate_to(replay_from)?;         // seq_len = prompt.len() - 1
self.cached_prompt.truncate(replay_from);
Ok((prompt_tokens[replay_from..].to_vec(), replay_from))
```

followed downstream by `model.forward_prefill([last_token], state)` and
`select_token` reading the resulting `prefill_logits`.

For Qwen3 `forward_prefill`, `tokens.len() == 1` routes into the same
`process_all_layers_batch` + `ops::prefill_attention_batch` +
`ffi::flashinfer_single_prefill` path that the cold 4-token prefill
takes, just with `seq_len=1`, `start_pos=3`, `kv_len=4`. The KV at
positions `[0..3)` is whatever the prior cold prefill (which wrote all
four positions in a single batch launch) left in cache; the KV at
position 3 is freshly computed by the batch-1 launch.

Even though the logits at position 3 *should* be a pure function of the
cached KV at `[0..3)` plus the freshly-projected Q/K/V at 3, FlashInfer's
tile layout and block-level accumulation order are batch-size sensitive.
Cold prefill runs the attention kernel once with `num_tokens=4,
kv_len=4`; the replay runs it with `num_tokens=1, kv_len=4`. The two
paths differ by a few ULPs on the dot-product reductions, and greedy
argmax flips on the very next token — ~4 generations in, the text
diverges entirely.

This is not a bug in any one kernel. It is a **numerical consistency
assumption** that the replay path has no legal way to satisfy while
(a) the K/V that positions `[0..3)` hold came from batch-4 and (b) the
K/V at position 3 is recomputed at batch-1. The scheduler sidesteps the
same divergence by using `forward_prefill_with_pool`'s explicit
`if tokens.len() == 1 { self.forward_decode(...) }` dispatch combined
with its paged-KV + prefix-cache architecture, but the single-request
engine never had that short-circuit.

## Fix

First attempt — **mirror the scheduler's single-token dispatch inside
plain `forward_prefill`**: if `tokens.len() == 1`, call
`forward_decode(tokens[0], state)` instead of running the batch prefill
kernel with batch=1. This moved the replay path onto the decode kernel,
matching what `forward_prefill_with_pool` does.

Result: **still wrong**, just a different wrong variant ("about a young
girl named Lila who is a talented painter, but she is not allowed to
paint because of her family's rules…"). The decode kernel produces its
own slightly-off logits at position 3 vs. what the cold batch-4 kernel
originally computed, so the assertion still fails. Reverted the forward
dispatch change.

Second attempt (landed) — **retire the `ReplayFinalToken` optimisation
in the single-request engine entirely and fall through to a full
recompute on exact match**. Diff in `infer/src/server_engine.rs`'s
`prepare_with_prefix_cache` now reads:

```rust
PrefixReuseAction::ReplayFinalToken { replay_from: _ } => {
    info!("KV prefix cache FULL HIT: exact match, falling back to full recompute for numerical consistency");
    self.state.reset()?;
    self.cached_prompt.clear();
    Ok((prompt_tokens.to_vec(), 0))
}
```

The `ReplayFinalToken` enum variant itself is preserved (still produced
by `choose_prefix_reuse_action` and still exercised by its own unit
tests) so the test matrix for the planner stays intact; only the
handler behaves like `FullRecompute` now.

Cost: an exact-same-prompt retry in the single-request engine re-runs
the full prefill instead of replaying the final token. On "Tell me a
story" (4 tokens) that is roughly 1 extra forward-prefill in exchange
for greedy bit-stability. The scheduler path — which carries all of
M2b's real cross-session reuse wins — is untouched.

Verification:
- `cargo test -p infer --release --test e2e` — green, Phases 1 through
  4 all PASS on Qwen3-4B.
- `cargo test --workspace --exclude mlx-sys --release --lib` — 348
  tests pass.
- `cargo test -p infer --release prefix_cache` — 33 pass (the planner
  unit tests still cover the `ReplayFinalToken` enum variant).
- `cargo test -p infer --release kv_tier` — 20 pass.
- `cargo fmt --all -- --check` — clean.

Pre-existing unrelated failures (tracked separately, not in scope):
`greedy_consistency` (B=3 batched decode —
`docs/experience/errors/2026-04-13-batched-decode-high-concurrency.md`),
`e2e_qwen35` (baseline drift vs current HF Qwen3.5-4B weights),
clippy `unused import: Path` in `crates/infer-tools/src/lib.rs`.

## Rule

**Any "replay a single token on top of partial KV" shortcut in a pure-
attention model only holds if the same kernel family produces the KV
at the replayed position and everything to its left.** The single-
request engine has no way to guarantee that today, because its cold
prefill goes through the batched FlashInfer prefill path while a
single-token replay would come out of either (a) the same path at
batch=1 (different tile shape → different accumulation) or (b) the
decode kernel family (different math entirely). Either way, greedy
argmax will flip after a handful of decodes.

**Corollary**: when you add a "replay N tokens of cached prefix" fast
path, you need a byte-exact-output consistency test the first time
you add it, not years later after its downstream test silently
stopped passing for an unrelated reason. Phase 3 of `test_e2e_generation`
is that test; keep it green going forward and the whole class of
numerical-consistency regressions in the replay path is contained.

**Corollary 2**: if a failure symptom is greedy divergence on a
specific phrase and the only difference between the failing and
passing paths is batch size, the answer is almost never "find the
bug in the kernel" and almost always "the two paths are
numerically-different-but-both-valid and you cannot safely assert
byte-equality across them." Fall back to a single canonical path
and accept the performance hit.
