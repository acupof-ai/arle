# cc-as-harness online OPD

> Status: Active — 2026-07-13. Loop + prefix-reuse VALIDATED (pod, GPU H20). Supersedes the weak in-process agent loop as the
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

## Chaining gap (verified, not assumed)

`--lora-adapters` is **defined but never read** in the agent-opd path (`grep
'\.lora_adapters' train_cli.rs` = ∅); the loader inits fresh zero-B LoRA
(`LoadMode::LoraStudent`), and there is **no adapter-load function** (only
`save_lora_adapters`, no import twin). So a round cannot resume from the prior
round's adapter → the bash loop can't chain. Serve *does* load adapters
(`serve --lora-adapters`, proven by `cc_run.sh`); replay *does* save them
(`save_agent_opd_adapters`). Only the **resume-load** is missing.

## Tranches

1. **KV correctness gate** ✓ — cc-shape multi-page prefix re-match test
   (`agentic_reprefill_cc_large_prompt_shape`, `70e3f5137`); per-round restart =
   fresh cache for #92. *(user priority: "kv cache 也得测试好")*
2. **Adapter resume-load** ✓ — `load_qwen35_lora_adapters` (PEFT registry twin,
   `591a864e8`) + `--lora-adapters` wired into agent-opd train/eval; serve loader
   accepts a PEFT dir (`3b9d0bb7`). **Validated (pod): round-1 `resumed adapter
   from …/adapters_replay`, CC_OPD_EXIT=0.**
3. **Online loop orchestrator** ✓ — `scripts/cc_opd_loop.sh`: per-round
   serve-restart → cc → convert → train-one-round (`--replay-records`) → chain
   adapter → periodic cc held-out eval. **Validated end-to-end (pod, 2 rounds).**
   **CE-only** (replay ignores `--update-strategy`).
4. **Prefix reuse (shared 17K prompt)** ✓ — periodic recurrent-sidecar snapshots
   + partial restore (`312d22c8c`, STRIDE=2048, L3-disk). **Validated (pod):
   needle gate 3/3 exact 115→8000; cross-task 18320 restore `restore_failed 1→0`;
   cc tasks pass at baseline rate.** Kills the per-conversation/per-restart 18K
   re-prefill.
5. **SAO-on-cc (follow-up)** — collect *failing* windows + carry reward through
   `cc-convert`; make `run_agent_opd_replay` honor `--update-strategy`. (cc-harness
   + SAO Phase 1/2 together.)
6. **Perf follow-up** — hot-reload adapter + #92 epoch-invalidation to kill the
   per-round model reload; native fixed-concurrency A/B for sidecar prefill savings.

## Non-goals (now)

In-process serve of the live student handle (the `Arc<Mutex<LoadedInferenceEngine>>`
→ `ServeHandle` adapter gap) — subprocess serve sidesteps it; revisit only if
per-round reload dominates wall time.
