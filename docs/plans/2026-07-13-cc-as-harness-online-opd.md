# cc-as-harness online OPD

> Status: Active — 2026-07-13. Supersedes the weak in-process agent loop as the
> default agent-OPD rollout harness.

## Verdict

The in-process ARLE agent loop (8-turn bash/read/replace, `AgentSession::run_turn`)
is a **capability bottleneck** — decoded cases show failures are mostly "ran out
of 8 turns mid-exploration," and the memory records cc gives **2/3 edits vs 0 ever
in the in-house loop**. Train the student under the strong harness it will deploy
under. The cc pipeline already exists offline; the fix is making it the **online
iterative** default **with KV-cache correctness under cc's access pattern**.

## What already exists (reuse, don't rebuild)

`scripts/cc_run.sh` → `arle serve --dump-messages-dir` → `scripts/cc_swe_baseline.py`
(cc drives the served student per task, serial) → `arle train cc-convert`
(`crates/train/src/cc_convert.rs`) → `records.jsonl` → `agent-opd --replay-records`.
This is cc-harness OPD, but **offline / single-pass / rejection-CE only**
(`run_agent_opd_replay` uses `masked_writeback_ce_step`, ignores `--update-strategy`;
`cc_run.sh` collects only *passing* windows).

## Design

**Online loop = per-round serve RESTART.** Each round: serve the round's adapter
(fresh process) with `--dump-messages-dir` → cc over the train tasks → `cc-convert`
→ train one round → save adapter N+1 → next round. Adapter chaining is supported
(`--lora-adapters` load + `--save-lora-adapters`).

### KV-cache correctness (the load-bearing gate)

cc's access pattern is brutal on the cache: **17K-token system prompt + multi-turn
where each request resends the whole growing history**. Grounded issues:

- **#92 stale-prefix-on-weight-epoch (correctness, invisible).** `RadixCache` is
  token-block-keyed with **no epoch/version** (`radix.rs`), prefix-cache default-on.
  A new adapter → cached KV computed under the *old* adapter is silently reused.
  → **Per-round serve restart sidesteps this for free** (fresh cache + weights).
  Hot-reload + epoch-invalidation (tag pages / flush on adapter swap) is the
  perf-optimized follow-up, not the MVP. See
  [[reference_prefix_cache_stale_across_weight_epochs]].
- **Multi-turn re-prefill / seq_len drift** — already fixed (`rewind_on_attach`,
  `infer-core/src/lib.rs:2748`) + tested (`:2707+`). cc's growing history is this
  case. **Gate: add a cc-pattern test** (growing history + tool-result injection +
  large system prompt) confirming no drift on this exact access shape.
- **17K-prompt re-prefill / TTFT 82s** — perf (recurrent-sidecar restore gap), not
  correctness. Defer.
- **#162 unbounded-concurrency KV exhaustion** — cc fires concurrent requests;
  cap with `--max-running-requests ≥2` (cc_run.sh uses 4).

### Token-exactness (the other risk)

`cc-convert` re-renders cc's dumped conversation through the ChatML span renderer
(`chat::render_structured_chatml_with_spans`) and masks assistant spans by byte
overlap (`cc_convert::mask_from_offsets`). Token-exact **only if** the ChatML
renderer matches the checkpoint's Jinja `chat_template` (Qwen: yes today; tool-call
/ whitespace drift breaks it silently — `cc_convert.rs:12-17`). Gate: assert the
re-rendered assistant tokens round-trip against the serve-emitted tokens on a
sample before trusting the mask at scale.

## Tranches

1. **KV correctness gate** — cc-pattern multi-turn prefix test (growing history +
   tool-result + big system prompt) in `infer-core`; confirm per-round-restart =
   fresh cache. *(user priority: "kv cache 也得测试好")*
2. **Online loop orchestrator** — `scripts/cc_opd_loop.sh`: per-round serve-restart
   → cc → convert → train-one-round → chain adapter → periodic held-out eval.
3. **SAO-on-cc (follow-up)** — collect *failing* windows + carry reward through
   `cc-convert`; make `run_agent_opd_replay` honor `--update-strategy`. (cc-harness
   + SAO Phase 1/2 together.)
4. **Perf follow-up** — hot-reload adapter + #92 epoch-invalidation to kill the
   per-round model-reload; recurrent-sidecar prefix restore for the 17K prompt.

## Non-goals (now)

In-process serve of the live student handle (the `Arc<Mutex<LoadedInferenceEngine>>`
→ `ServeHandle` adapter gap) — subprocess serve sidesteps it; revisit only if
per-round reload dominates wall time.
