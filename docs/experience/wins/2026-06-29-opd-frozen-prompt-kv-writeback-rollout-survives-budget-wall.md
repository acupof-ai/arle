# OPD frozen-prompt-KV writeback — rollout fix survives, payoff blocked by a rollout-budget wall (case-decoded)

## Context

Goal: measure the frozen-prompt-KV masked-CE writeback payoff end-to-end on the
H20 box (8×H20 97 GB). The frozen-KV path forwards only the *generated* segment
(prompt-prefix KV seeded off-tape) instead of the full ~15k-token trajectory,
targeting the full-trajectory writeback OOM (baseline `forward_hidden_states
~2144 s`, peak ~88–95 GB, OOM `cuda alloc_zeros failed` at ~90 GB — see
[`docs/plans/2026-06-25-opd-storage-tiering-and-writeback-oom.md`](../../plans/2026-06-25-opd-storage-tiering-and-writeback-oom.md)).

The run blocker (the `materialized state len != DecodeRow.kv_seq_len` assert at
agentic re-prefill) was fixed at HEAD by `1b0f0459` (`restore_recurrent_sidecar`:
`release_recurrent` before `acquire` + `set_seq_len(matched_len)`), locked by the
regression test `6848375f`. HEAD `d7a87ec9` also carries the frozen-KV feature
(gated `ARLE_OPD_WRITEBACK_FROZEN_PROMPT_KV`, Gate-A-exact: loss ≤1e-5, LoRA grads
≤1e-4 vs the full path — `crates/train/tests/test_frozen_prompt_kv_writeback.rs`)
and ckl's FP8 KV-cache feature.

- **Build:** clean local HEAD `d7a87ec9` synced to `/host/arle-ckl-aopd` (git
  archive tarball over the drifted pod tree), `--features cuda`, **BUILD_EXIT=0**.
  Binary symbol-verified: frozen-KV phase strings (`[masked-writeback-frozen]
  seq_len=`, `offload_checkpoints=`), env gate `ARLE_OPD_WRITEBACK_FROZEN_PROMPT_KV`,
  rollout-fix `set_seq_len(matched_len)` in source. The HEAD `kernels.toml`
  (954 lines) carries new HD256 FP8 prefill kernels absent from the warm
  `/host/arle-build` tree (808 lines) → forced a real recompile; TileLang AOT
  regen needed a working python (the aopd venv's tilelang crashed with the
  tvm-ffi `__ffi_repr__ already registered` double-registration error; the
  `/host/arle-build` venv's tilelang 0.1.11 / tvm 0.25.dev0 imported clean and
  built the kernels via `INFER_TILELANG_PYTHON`).
- **Run:** GPU 4 (clean, 0 MiB), persistent tmux, `exec -a arleCKL`,
  `ARLE_OPD_WRITEBACK_FROZEN_PROMPT_KV=1`, student `/host/Qwen3.6-27B-FP8`,
  `--share-frozen-base`, `--lora-layer-start 32 --rollout-num-slots 1
  --samples-per-prompt 4 --writeback-cap 1 --rounds 1`.

## What Worked (the two payoff preconditions are SOLID)

1. **Rollout SURVIVES the agentic context — the `1b0f0459` fix holds.** Zero
   `materialized state len` assertions across base-eval (3 held-out tasks) + the
   training rollouts, every run, through dozens of multi-turn re-prefills. The
   rollout blocker is genuinely fixed.
2. **Frozen-KV writeback path is compiled in and Gate-A-exact.** Binary carries
   the frozen path; the local Gate-A test proves the gen-segment-only forward
   reproduces the full-trajectory CE loss (≤1e-5) and LoRA grads (≤1e-4).

## The wall (case-as-fact — decoded the actual rollout tokens, NOT inferred)

The writeback payoff could **not** be exercised because **no rollout produced an
accept** (`passed=0 → trained_pairs=0 → writeback never fires`) under achievable
configs. This is a **rollout generation-budget wall**, not a frozen-KV or
rollout-fix wall — the clean-HEAD binary's rollouts are *verbose-reasoning-first*
where the prior (`fkv2b`) binary led with short tool calls.

| run | `--max-turns` | `--max-tokens` | rollout-phase VRAM peak | decoded rollout behavior | accept | writeback |
|---|---:|---:|---:|---|---:|---|
| **fkv3** (brief config) | 16 | **768** | 36.8 GB | 4/4 samples `turns=0` — `turn0 fin=length`: model writes a prose plan, **exhausts 768 tokens before emitting any `<tool_call>`** (`tool_call` literal count = 0 whole run) | 0 | not fired |
| **fkv4** | 16 | **4096** | 40.1 GB | sample 0 `turns=16` (engaged — budget fix worked), **hits the 16-turn ceiling one edit short** (at turn 15 it had stated the exact root cause); samples 1–3 `turns=0` Stop (temp=1.0 chose prose-then-stop) | 0 | not fired |
| **fkv5** | **30** | 4096 | 43.8 GB | sample 0 `turns=30 MaxTurns` — **navigated into the WRONG task's staged repo** (`cd .../ansible__ansible-0ea40e0`, an *eval* task, from turn 0) and spent all 30 turns there, about to fix `0ea40e0`'s `HASH_BEHAVIOUR` dict-merge bug instead of `f327e65`'s keyword-validation bug → `git diff` on its own repo empty → no edits; samples 1–3 `turns=0` Stop | 0 | not fired |

All three: base-eval held-out `pass_rate=0.3333 (1/3)` (task `0ea40e0` passes via a
short tool call; the other two are beyond the base model in budget). VRAM peak
tracked the rollout/eval engine only — the writeback spike to ~90 GB never
occurred, confirming the writeback never ran.

**Decoded cases (case-as-fact, per-turn tokens):**

- *fkv4 sample 0* (`turns=16`): navigated *correctly* — grep'd to
  `_collection_finder.py` → `validate_collection_name` → `is_valid_collection_name`,
  and by turn 15 had stated the exact root cause ("checks if the namespace or
  collection name is a Python keyword") — but spent all 16 turns exploring and
  **ran out of turns before the edit step**. The prior `fkv2b` binary solved this
  same task in **19 turns**.
- *fkv5 sample 0* (`turns=30`): from **turn 0** the model `cd
  /host/aopd_work_fkv5/ansible__ansible-0ea40e0` — the wrong task's repo (`0ea40e0`
  is an *eval* task, staged into the shared work-root by base eval). It explored
  `0ea40e0`'s tree for all 30 turns and at turn 29 was about to apply
  `0ea40e0`'s fix (`result = dict(a); result.update(b)`), never touching its own
  task `f327e65`. Even a perfect edit there would leave `f327e65`'s `git diff`
  empty → no_edits. So `--max-turns 30` did NOT yield an accept; the failure is
  rollout *navigation*, not budget.

## Measured payoff — NOT obtained (writeback never fired)

Across all three runs `passed=0 → trained_pairs=0 → writeback never invoked`, so
the headline payoff numbers (forward-gen-segment time vs baseline ~2144 s; peak
writeback VRAM vs ~90 GB; `mean_loss`; held-out Δ) **could not be measured**. The
single training task on this binary produced zero accepts under every achievable
rollout budget. The prior `fkv2b` run (different binary) DID reach the writeback
on this task (`[masked-writeback-frozen] seq_len=14581 gen_start=960
gen_len=13621`) but OOM'd on `cuda alloc_zeros` with the rollout engine resident
— so the frozen-KV gen-segment path forwards `gen_len≈13.6k` (not "a few hundred"
as hoped; the agentic generated segment is large), and a clean-GPU re-measure of
that path is the remaining open item once a passing rollout exists.

## Rule

- **The frozen-KV writeback payoff is gated on an *accepted rollout*; without
  one, the path can't be exercised — and accept-rate is a rollout
  budget/capability/navigation property, not a writeback property.** Before
  spending a heavyweight run on the writeback win, first clear the rollout-accept
  bar with a *cheap* probe (1 task, decode the per-turn tokens): `--max-tokens`
  large enough to reach a `<tool_call>` before truncation (`turn0 fin=length` =
  too small), `--max-turns` large enough to finish the edit (this binary needs
  ≥19; the brief's 16 is short), and a work-root where the model can't `cd` into
  a *sibling* task's staged repo (fkv5 sample 0 worked the wrong task's tree for
  all 30 turns → empty diff). The brief's `--max-turns 16 --max-tokens 768`
  produces 0 accepts on Qwen3.6-27B-FP8 with the clean-HEAD binary.
- **A `passed=0 / no-writeback` round is a rollout case to decode, never a
  frozen-KV KILL.** Decoding the per-turn tokens (`fin=length` / `MaxTurns` /
  `Stop` / which dir it `cd`'d into) pins the wall precisely; the rollout-fix
  (`1b0f0459`) and frozen-KV (Gate-A-exact) preconditions both stayed green
  throughout — neither was the blocker.
- **The frozen-KV gen-segment is large, not "a few hundred tokens."** `fkv2b`
  measured `gen_len≈13.6k` on a ~14.6k trajectory — the agentic *generated*
  portion dominates. The win vs the full-trajectory forward is real but bounded
  (~1k prompt-prefix saved here); the clean-GPU re-measure of forward-gen time +
  peak VRAM is deferred until a passing rollout exists to fire the writeback.
- **Note on sample size:** even a measured Δ here would be a 1-train-task /
  3-held-out-task signal — below the multi-seed≥5 + Wilson-CI bar; it would be a
  first value signal only, never a default-flip license.

Claude-Session: https://claude.ai/code/session_01Vsoud3oabdLDppvb274bCr
