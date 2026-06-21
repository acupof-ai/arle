# Rubric-OPD 27B GPU validation — a day of self-inflicted detours

## Context

Validating rubric-OPD (one-model self-consistency) capability on Qwen3.6-27B-FP8
(8×H20). Over one session the *experiment* was blocked repeatedly by my own
inference errors and config mistakes — none were real model/method bugs. Logged
so the next run skips them.

## Root causes + fixes (in order hit)

1. **Stale binary.** A "fixed" MMA decode kernel had no effect because the pod
   `arle` binary (mtime 10:02) predated the source push (10:16) — it was never
   rebuilt. **Rule:** "no effect after a fix" → suspect a stale binary FIRST;
   compare binary mtime vs source mtime / `strings | grep <symbol>` before
   theorizing about the code.

2. **Prefill-OnceLock log misread → 3 phantom bug chases.** The decode-GEMV path
   log used a single `OnceLock` that fired on the FIRST GEMV — which is a
   **prefill** call (seq_len>16, always scalar by design; MMA is decode-only).
   I read "scalar fallback" as "MMA never engages on decode" and chased THREE
   phantom kernel bugs (K%128 gate, `s_cc_major` runtime-vs-driver API,
   stale-`cudaGetLastError`), all reverted. A `RUST_LOG=info` + a *decode-specific*
   log finally showed `tensor-core MMA path`. **Rule (§0): ALWAYS measure the
   decode path with a decode-specific probe; NEVER infer engagement from a
   shared-OnceLock log or from "all dims pass the gate". Inference ≠ evidence —
   I was wrong 3× in one afternoon.**

3. **grad-checkpointing force-disabled in self-consistency → CE OOM.**
   `train_cli.rs` skipped `set_gradient_checkpointing` when `--self-consistency`
   on the false premise that freeing the judge's VRAM covers the CE activation
   set. It does not — the 27B CE backward OOM'd at `alloc_zeros`. **Fix b0e3e2d9:
   always honor the flag.** **Rule:** freeing *inference* VRAM ≠ fitting *training*
   activations; they are independent.

4. **Under-validated CE config → all 8 seeds crashed.** I "validated" the
   grad-ckpt fix at max-new **512** (writeback-batch 4), then launched 8 seeds at
   max-new **1024** → the `slice_bwd` `[B,S,V]` logits-grad (vocab 248320) doubled
   → `alloc_zeros` OOM on all 8. **Fix:** `--writeback-batch 1` (peak `1×1024·V`
   = HALF the proven `4×512·V`). **Rule:** validate at the REAL workload shape,
   not a smaller smoke shape — the failing path's scaling is shape-specific.

5. **greedy self-consistency is degenerate.** SC with greedy rollouts gives N
   identical samples → majority-vote is a no-op. **Fix:** `--rollout-temperature
   0.7` (distinct accepted 33-41/round confirmed diversity). Also enabled the
   8-seed multi-seed design.

6. **Weight-share review catches (delegated diff).** Reviewing the train-infer
   weight-share agent diff before commit (no codex credits → reviewed by hand)
   found: (a) a borrowed-FP8 `Drop` that `ptr::read+forget`-ed a *copy* while
   drop-glue still freed the original `Arc` → would `cuMemFree` the infer engine's
   live FP8 bytes (double-free); fixed to `clone()+forget` (bump strong count +
   leak). (b) an always-engine-first load order → the `num_slots` clamp sees full
   free VRAM → over-reserves KV → default-path student OOM; fixed to conditional
   (default student-first, shared engine-first). **Rule:** delegated >5-file
   architectural diffs need a hand review of the memory-safety + ordering crux
   before commit; the typecheck doesn't catch a double-free or a num_slots
   regression.

7. **Cosmetic round-label.** `run_rubric_rounds` (called per outer round with
   `rounds:1`) hardcodes "round 0" in its phase-A log → every round logs
   "round 0 phase-A", which read as "stuck on round 0". The outer
   `for round in 0..rounds` (train_cli.rs:1713) uses the correct index for
   `eval_round{N}.jsonl` + the summary. Cosmetic; 1-line fix to thread the global
   round into the display.

## Rule (one line)

§0 to the bone: when a fix "has no effect" or a metric "looks wrong", the bug is
almost always in my *measurement* (stale binary, misread log, under-sized
validation) — measure the actual thing (decode path / real-config CE / per-seed
decode) before touching the model or the method. Inference ≠ evidence.
