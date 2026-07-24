# agent-OPD sweetspot3 band=1 — two band-cut bugs masking a base-prompt length wall

## Context
#173 phase 1: 7-GPU sharded comfort-band profile on staged-sweetspot3 (27B student,
32 tasks, SAMPLES=8, SPEC=off, 600s cc-timeout default). Band came back **1 task**
(< 8 floor). Decoded to ground truth before any recut (CLAUDE.md §0 case-as-fact).

## Root Cause — three findings, all decoded/traced, not inferred
1. **`comfort_band.py --pass-threshold` default = 1.0** (line 177): a sample counts as
   "pass" only at reward==1.0 (all fail_to_pass green). The 27B produces dense PARTIAL
   credit (0.6–0.92) and rarely a perfect 1.0, so competent partial-solvers with real
   in-group variance (std 0.31–0.43) are mislabeled `too-hard(pass=0.00)` and dropped.
   Local recompute: pass@1.0→band=1, pass@0.5→band=5, std>0.05→7.
2. **Token accounting zeroed for timed-out samples** (`train_cli.rs:3450`
   `filter_map(|s| s.cc_input_tokens)`): `cc_input_tokens` is parsed from the `claude`
   CLI stdout `usage` (`cc_harness.rs:341`), which is `None` when the session hits the
   600s wall (`cc=None`). 20/28 tasks had all 8 samples time out → group prompt_tokens=0
   → `avg_traj=0` → trivially passed the `too-long` filter. The kept task survived on
   this zeroing coincidence, not a clean keep.
3. **The real wall — base prompt ~21.5K tokens.** 0-GPU re-derivation from 745 on-disk
   `.tokens.json` sidecars (final-turn = max-prompt per sample, matching cc_convert.rs:185;
   NOT summing turns — that double-counts the resent prefix) shows **27/28 tasks' real
   avg_traj is 22K–28K**. The `claude` CLI injects a ~21.5K-token system preamble
   (measured: turn-1 prompt=21641) — its own system prompt + tool schema, NOT our
   ~250-token `cc_prompt`. Every trajectory starts 21.5K deep, so none fits the
   writeback's 23K `max_update_seq` VRAM wall (update_strategy.rs:653; errors/2026-07-22
   confirms 27B LoRA backward OOMs at seq≈30K, offload/quantized-KV can't save it).

**Net: std>0.05 ∩ ≤23K = 0 tasks.** No pass-threshold or pass-lo/hi tuning recovers a
band — the corpus trajectory length (22–28K) and the writeback wall (23K) are collinear,
separated by the fixed 21.5K CLI base prompt.

## Fix (staged, not all landed)
- comfort_band: `--pass-threshold` default 1.0→0.5 + a reward-variance keep criterion
  (std>0 = non-zero dapo advantage) — the correct signal for dapo, not pass-count.
- Token bug `train_cli.rs:3450`: read tokens from the sidecar (present even on timeout),
  not the CLI usage — separate runtime commit, bench-gated.
- The length wall needs a real decision (base-prompt trim is CLI-owned / not ours;
  raise max_update_seq hits the known 30K OOM; or shorter-task corpus) — deferred to ckl.

## Rule
Before a "corpus too narrow → widen the band" reflex, decode the token/reward ground
truth: a thin band can be a measurement artifact (pass-threshold, zeroed tokens) stacked
on a physical wall (base-prompt length vs writeback VRAM). Recut only after the length
proxy is trustworthy (0-GPU re-derive from sidecars) — tuning pass-lo/hi on garbage
avg_traj wastes a GPU-day. The `claude`-CLI base prompt (~21.5K) is a fixed floor on
every agentic-OPD trajectory; corpus length budgets must be set relative to it.
