# cc-harness online OPD + shared-prefix reuse — pod-validated

> Status: Shipped — 2026-07-13 (H20). Correctness-gated; guidellm perf A/B pending-remote.

## Context

Made the cc-harness (Claude-Code drives the served student) the online iterative
OPD rollout, replacing the weak 8-turn in-process loop, and fixed the shared 17K
system-prompt re-prefill. Plan:
cc-as-harness online OPD.

## What worked

**Online loop end-to-end** (`scripts/cc_opd_loop.sh`, CC_OPD_EXIT=0, 2 rounds):
per-round serve-restart → cc rollout → `cc-convert` → `agent-opd --replay-records`
train → **adapter resume** → cc held-out eval. cc drove the 27B to a passing edit
(`228 passed`) where the native 8-turn loop stalls at "no edits / ran out of turns".

Two chaining bugs found + fixed by case-level pod debugging:
- **resume-load missing** — `--lora-adapters` was defined but never read + no
  import twin of the save. Added `load_qwen35_lora_adapters` (reuses the PEFT
  registry, `591a864e8`) + wired it into agent-opd train/eval.
- **serve loader took a file, got a dir** — `load_student_lora_update` did
  `fs::read` on the adapter dir; now resolves `<dir>/adapter_model.safetensors`
  (`3b9d0bb7`). The name parser already accepted PEFT + raw conventions.

**Shared-prefix reuse** (`312d22c8c`) — Qwen3.6 is hybrid (full-attn + GDN
linear-attn); the recurrent state only restores at a *snapshotted* boundary.
Evidence (`ARLE_KVDRIFT_DEBUG`): within a conversation the sidecar already HITs;
the miss was cross-conversation — a new task radix-matches the shared 17K prompt
at `matched_len=18320`, but saves land only at each request's own end (≥21440),
never at the shared boundary → full 18K recompute. Fix = MambaRadixCache pattern:
periodic recurrent snapshots every 2048 tokens during long prefills (→ tier, L3
disk, survives serve restart), restore probes the largest boundary `B ≤
matched_len` and re-prefills only `[B..prompt]` (≤2048). Boundary block
re-prefills fresh → dodges the vLLM "one-block-too-many" corruption.

**Correctness-gated on GPU** (the load-bearing step for a hybrid recurrent-cache
change — silent wrong-output risk):
- Needle correct-inference gate: **3/3 exact at every length 115→8000**
  (`TEMPLATE=qwen3_nonthink RAW=1`); the 8000-tok repeats fire partial-restore and
  still retrieve exactly → partial-restore does not corrupt output.
- Cross-task reuse: the `matched_len=18320` `restore_failed` went **1 → 0** — the
  boundary that fell back to full recompute now restores cleanly. cc tasks pass at
  the pre-sidecar baseline rate (1/2, byte-passing edits) → KV not corrupted.
- The sub-`21440` restore at 18320 succeeding proves the ≤18320 periodic snapshot
  functions (no request-end save is ≤18320).

## Rule

- **A hybrid (attn + recurrent/SSM) prefix-cache change is correctness-critical,
  not perf-only** — the recurrent state is a fold over the prefix and only
  restores at snapshotted boundaries; a wrong boundary silently corrupts output.
  Gate it with the needle correct-inference ladder (3× same-config) + a real
  reuse run that must still pass, before trusting it. Reuse across conversations
  needs a snapshot at the *shared* boundary (MambaRadixCache), not just the
  request's own end.
- **cc-harness = capability multiplier for OPD rollouts**: cc gets the 27B to
  edit + pass where the in-process 8-turn loop stalls. Train under the harness the
  student deploys under.
- **Debug the chain link-by-link at case level**: two "resume broken" failures
  were both trivial path/format conventions (dir-vs-file, PEFT-vs-raw naming),
  found by quoting the exact serve/train log line, not by re-reasoning the design.
