# Metal Qwen3.5 — post-double-buffer c=1 audits, /loop re-closed

> **STATUS: /loop ARCHIVED AGAIN (2026-04-20).** The double-buffer win
> (commit `f6be5f6`, +12.7% step-driver c=1) reopened the 2026-04-19
> terminal-state doc, but two follow-up audits + an mlx_lm source
> comparison confirm no further /loop-reachable "大招" levers exist for
> Qwen3.5-4B-MLX-4bit on M4 Max. **Do not re-enter the /loop with the
> stale "continue Metal Qwen3.5 optimization" prompt.** Next real upside
> needs Xcode Metal capture of the GDR kernel
> ([`docs/plans/metal-gdr-kernel-xcode-capture.md`](../../plans/metal-gdr-kernel-xcode-capture.md))
> or the M5.3a device-resident-tensor architecture work
> ([`docs/plans/m5.3-device-resident-tensor.md`](../../plans/m5.3-device-resident-tensor.md)).

**Date**: 2026-04-20
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA), macOS 26.3.1
**Model**: `mlx-community/Qwen3.5-4B-MLX-4bit`
**Parent win**: `2026-04-20-metal-qwen35-decode-double-buffer.md` (commit `f6be5f6`)

## Audits performed in this session

### Audit 1 — DFlash batched speculative path sync taxes

Question: the scalar-path double-buffer saved ~1.5 ms/step. Does the
DFlash batched speculative path (`try_decode_qwen35_dflash_speculative_batch`
at `request_state.rs:1692-1999` → `qwen35_dflash_speculative_block_batched`
at `dflash.rs:2092-2476`) have a bigger deferrable sync tax?

Verdict: **(b) medium upside, 2-5% at c=2**, not a 大招.

- Dominant sync tax is the per-row `.item_i32()` prefix-match scan at
  `dflash.rs:2400` (via `sample_rows()` → `dflash.rs:1505`). This
  materialization is **unavoidable** because prefix matching is a
  token-by-token CPU equality scan.
- Unlike the scalar `sampled` token (independent of verification,
  deferrable cleanly), DFlash `accepted_inputs` is **input-shaped** —
  it gates (i) the per-row hidden slice at `dflash.rs:2432-2437` and
  (ii) the rollback branch condition at `dflash.rs:2419`.
- Only deferrable item is the terminal `eval([packed_kv, packed_gdr, ...])`
  at `dflash.rs:2456`. Saves ~0.5-1.0 ms/block ≈ ~2-5% of c=2 TPOT
  (23 ms baseline). Not pursued.
- Rollback branching requires CPU accept_len read — GPU-side accept_len
  via cumsum+argmin is technically feasible but forces a GPU→CPU fence
  anyway to decide whether to run tape replay. Net win <0.3 ms.

### Audit 2 — HTTP-layer per-token overhead at c=1

Question: step-driver c=1 is now 85.4 tok/s (beats mlx_lm's 84.4 FFI
baseline). But HTTP c=1 is 66.2 tok/s aggregate = 3.4 ms/token of
HTTP-specific tax. Where does it go?

Verdict: **(b) several medium chunks, ~2-3 ms total fixable, no single
lever**.

| source | file:line | Δms/tok | fixability |
|---|---|---|---|
| Tokio cross-thread wake (native scheduler thread ↔ Axum task) | `runtime.rs:603` | 0.5-1.5 | Medium (architectural) |
| `process_token` hop (detokenize + `send_text_delta` mpsc) | `runtime.rs:107-114,1720` | 0.5-1.0 | Medium |
| JSON serialization per chunk | `http_server.rs:615,625` | 0.1-0.2 | Low (marginal) |
| Incremental tokenizer `DecodeStream::step` | `tokenizer.rs:115` | 0.1-0.3 | None (already optimal) |
| SSE frame construction + HTTP/1.1 chunk headers | axum | 0.05-0.1 | None (protocol floor) |

Note on prior audit claim: the auditor initially flagged "double-buffer
is defeated on HTTP because `process_token` blocks on `.item_i32()`".
**This is wrong.** Commit `f6be5f6` queues step N+1's `async_eval` BEFORE
the `.item_i32()` materialize, so the GPU queue stays one-deep regardless
of caller. HTTP benefits from the double-buffer equally. The 3.4 ms gap
is per-step Rust/tokio overhead, not GPU idle.

Not pursued — no single lever gives a ≥5% win; total potential 2-3 ms is
distributed across ≥3 architectural changes (move scheduler to tokio task,
merge detokenize into scheduler tick, streaming JSON encoder). Each would
need its own matched A/B bench and carries its own regression risk. Grind
work, not a 大招.

### mlx_lm source comparison (Qwen3.5 generator)

Question: does `mlx-lm`'s generator have any Qwen3.5-specific technique we
haven't implemented?

Finding: **no — mlx_lm's `generate_step` uses the exact same
double-buffer pattern we just landed.**

Quoted pattern from `mlx_lm/generate.py`:
- Initial step → `mx.async_eval(y, logprobs)` (no block)
- Main loop: `next_y, next_logprobs = _step(y)` with previous `y` still
  un-materialized
- Only call `y.item()` at yield time (end of iteration)

Matches our `Qwen35StepDriver::decode_token` post-`f6be5f6` structurally.
There is no hidden Qwen3.5-specific trick we missed.

**Incidental finding — mlx_lm issue #932**: Gated DeltaNet shows a 93 → 34
tok/s (2.7x) decode slowdown on non-vocab input embeddings in their own
implementation (`Qwen3.5-35B-A3B`). This is a **known open issue on their
side** with no fix. Confirms the GDR kernel we measured at 6.1 ms/row is
near mlx-lm's own kernel ceiling — this is a genuine kernel-level problem,
not a missing integration on our part.

## Conclusion — /loop is re-closed

Summary of what remains:

| lever | status | next action |
|---|---|---|
| Scalar step-driver c=1 | **+12.7% landed** (`f6be5f6`) | none — matches mlx_lm FFI |
| DFlash batched c≥2 | audited, 2-5% upside ungrindable | defer; low ROI vs effort |
| HTTP c=1 tax (3.4 ms/tok) | audited, no single lever | defer; architectural |
| mlx_lm Qwen3.5 tricks | compared, none missed | done |
| GDR kernel 6.1 ms/row | requires Xcode Metal capture | out-of-/loop; plan ready |
| M5.3a device-resident tensor | architectural unblock | out-of-/loop; plan ready |

**/loop accountability note**: per `CLAUDE.md` guidance "超过 2 次解决不了
同一问题就停下来写清现状", two c=1-upside audits + one source comparison
= three approaches attempted in this /loop iteration, all concluding
medium-or-zero upside within /loop scope. Stopping here and closing the
prompt is the correct call.

## Problems (no code changes this session → no bench run)

- Per `CLAUDE.md` §Benchmarks: this session made zero runtime changes
  (two audits + a web fetch + this markdown). Hence **exempt from the
  "every runtime change requires a bench entry" rule**. The parent win
  (`f6be5f6`) already has its bench entry in
  `2026-04-20-metal-qwen35-decode-double-buffer.md`.
- `cargo clippy --features metal` still fails in this environment (cmake
  profile drift when clippy rebuilds mlx-sys). Pre-existing env issue,
  not introduced this session. Normal `cargo build --release` clean.

## Rule

**When a win reopens a "terminal state" doc, budget exactly one /loop
iteration to confirm whether the reopening reveals a new class of lever
or just extends the grind.** The 2026-04-19 terminal doc was correct in
substance — the only /loop-reachable upside was the scalar double-buffer
(Iter 12, landed). Once that banked, the audit sweep confirmed the
previously-documented ceilings still hold. Don't treat "win found" as
"project reopened" by default; treat it as "one more audit pass to
verify the terminal state is still terminal."

Corollary: the GDR kernel / M5.3a work is the real next arc. Those carry
their own plan docs, their own bench scaffolding, and their own /loop
scope. Don't bundle them with the "continue Metal Qwen3.5 optimization"
prompt.

## Cross-refs

- `docs/experience/wins/2026-04-20-metal-qwen35-decode-double-buffer.md`
  — parent win (step-driver +12.7%, commit `f6be5f6`).
- `docs/experience/wins/2026-04-19-metal-qwen35-final-state.md` — original
  terminal-state doc this session re-confirmed.
- `docs/experience/errors/2026-04-19-dflash-long-prompt-prefill-chunking-desync.md`
  — unrelated latent DFlash prefill bug documented this session (commit
  `d93bb72`), not addressed; flagged for the next session that touches
  scheduler prefill paths.
- `docs/plans/metal-gdr-kernel-xcode-capture.md` — next real lever,
  out of /loop scope.
- `docs/plans/m5.3-device-resident-tensor.md` — architectural unblock
  for the longer-term Metal arc.
- mlx_lm issue #932 (Gated DeltaNet 2.7x slowdown on non-vocab
  embeddings) — external confirmation that the GDR kernel ceiling we hit
  is a genuine upstream-known kernel-level problem, not integration
  deficit on our side.
