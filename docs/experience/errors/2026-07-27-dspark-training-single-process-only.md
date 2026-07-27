# DSpark test-time training is single-process-only; multiproc TP silently no-ops it, 2026-07-27

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

## Open — the premise IS runnable single-GPU; multiproc is separate future work

**Correction to the first framing:** the ISO premise sweep is NOT structurally
blocked. `world_size <= 1` → `bind_relay_and_spawn_workers` returns `None` →
single-process serve → `on_engine_loaded` fires → the sidecar runs
(serve_multiproc.rs:111). Qwen3.6-27B-FP8 (~27 GB) fits one H20 (96 GB) at
`WORLD_SIZE=1`, and the ISO-**off** premise sweep needs **no seeded head**: a
cold-grown head (w2=0, w1 Xavier-init) trains ISO-off, and `SpectrumProbe`
captures w1's nonzero base spectrum → a real drift signal. The `iso_without_seed`
guard only fires when ISO is ON, so ISO-off cold is allowed. The devops agent hit
the multiproc wall only because the staged *seeded heads* were DSv4-vocab
(forcing TP) — an asset artifact, not a necessity.

So the sweep runs today: serve Qwen3.6-27B-FP8 single-GPU, `--spec-type dspark
--dspark-train --dspark-prob-match-alpha {0,0.5,1}`, no `--dspark-markov-init`, no
`--dspark-train-iso`; read w1 `spectrum_drift`.

**The multiproc gap is real but SEPARATE — it is not the #32 blocker.** Wiring the
DSpark sidecar into the multiproc rank-0 worker (route rank-0's process-global
experience buffer + an mpsc weight-swap drained between lockstep ticks — the
draft is TP-collective but rank-0's chain is authoritative, dsv4.rs:1801/1835, so
only rank-0's head matters) is future work for TP-scale DSpark *training*. It is
not needed to measure the ISO premise. Filed as its own follow-up, not gated on
the sweep.

Until the sweep runs, Phase 5's H20 license is **unmeasured** (not failed): the
ISO A/B and Agent-RFT ISO (#32) stay gated. Do not treat "unrun" as "premise
holds."
