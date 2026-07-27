# DSpark test-time training is single-process-only; every DSpark model is multiproc, 2026-07-27

## Context

The plan's Phase 5 ISO license gate needs an ISO-off `prob_match_alpha = 0/0.5/1`
spectrum-drift sweep on a real DSpark head — the premise being "an unconstrained
pure-PG acceptance update leaves the head near-isospectral." H20 verification
(HEAD `b922bde52`) cleared the build and every unit gate, and two output-path
gaps I'd missed were fixed along the way (`--dspark-prob-match-alpha` `327d172e4`,
the ISO-off `SpectrumProbe` `3e883e127`). The sweep still could not run.

## Root cause

The DSpark train sidecar and `--dspark-markov-init` are wired **only on the
single-process serve path** — `on_engine_loaded` (serve.rs) is the sole site that
loads the head and spawns `spawn_dspark_train_sidecar`. Under multiproc TP the
coordinator (`run_config` → `serve_multiproc`) returns before that hook is ever
constructed, and `serve_multiproc.rs` has zero sidecar wiring.

**And every DSpark-capable CUDA model is multiproc.**
`cuda_model_takes_multiproc_serve` (loaded.rs:1916) is true for `Dsv4 | Qwen35`
— i.e. DSv4-Flash **and** Qwen3.5/3.6. Only `Qwen3Dense` serves single-process,
and the staged seeded heads are all DSv4-Flash vocab (248320). So on current pod
assets the sweep is structurally unrunnable: the one serving mode the staged
heads support is exactly the mode the trainer isn't wired into.

Verified empirically: served DSv4-Flash-FP8 TP=4 with `--dspark-train
--dspark-markov-init --dspark-prob-match-alpha 0`, drove 40+ requests → 0
`dspark_train: step=` lines, 0 `spectrum_drift` lines, no sidecar start. The
flags were silently inert.

## Fix (this tranche)

`b922bde52`: `run_config` fails fast when `--dspark-train` / `--dspark-markov-init`
meet a multiproc model, instead of serving with the sidecar silently dead —
Phase 3's rule that a no-op flag must reject, not no-op. This does NOT unblock the
sweep; it makes the blocker loud.

## Rule

**A test-time-training sidecar must be wired into every serving mode its target
model actually uses, or it is dead on arrival.** DSpark training was built and
unit-tested on the single-process path, but the models it drafts for (DSv4,
Qwen3.6 MoE) all run multiproc TP — so the feature never ran end-to-end on its
real substrate. Before claiming a training path works, confirm the *serving mode
of the production model* reaches the trainer, not just that the trainer passes
its own unit tests.

## Open — the premise is unmeasured, not disproven

To run the ISO premise sweep, one of:
1. **Wire the DSpark sidecar + head-init into the multiproc rank-0/coordinator
   path** (`serve_multiproc.rs`) — a bounded new feature; the trainer already
   runs on a CPU sidecar independent of TP, so it's a plumbing job (route rank-0
   experiences + head-sync through the coordinator). This is the general fix and
   unblocks all DSpark test-time training under TP.
2. **Stage a dense-Qwen3 (single-process) DSpark substrate** — smaller, off the
   validated Qwen3.6-MoE substrate, so a weaker premise signal.

Until then Phase 5's H20 license is **unmeasured** (not failed): the ISO A/B and
Agent-RFT ISO (#32) stay gated. Do not treat "unrun" as "premise holds."
