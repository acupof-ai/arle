# agent-OPD MTP lockstep coordinator tears down on a single GPU

## Context

Validating the new `gpu_busy_frac` timer on the pod: `SMOKE=1 GPU=1
scripts/agent_opd_curve.sh` on one H20, ThinkingCap-Qwen3.6-27B-FP8 + `SPEC=mtp`
+ `--share-frozen-base`. Build clean (BUILD_EXIT=0), `gpu_busy_secs`/`gpu_busy_frac`
emit correctly — but the run is degenerate: all **32/32 round-0 rollouts return
`API Error: 500 coordinator lockstep loop closed`**, `completion_tokens=0`, so the
emitted `gpu_busy_frac ≈ 0.0–0.03` reflects a **dead serve**, not an
agent-latency-idle-bound rollout. Reporting "≪1 → mega-rollout worth building"
from it would be a case-as-fact false conclusion.

## Root Cause

`infer_server::coordinator` (`coordinator.rs:165`) — the lockstep coordinator's
120 s ack-watchdog fired at tick #1823648 (min_acked 4 behind) and **deliberately
tore down** ("tearing down instead of hanging forever"), correlated with the first
real rollout inference. MTP spec-decode reuses `tp_lockstep_proposal/accept`, which
is designed for **TP≥2**; on **`world_size == 1`** (single GPU) the coordinator
waits for cross-rank acks that never arrive → watchdog teardown → every subsequent
submission 500s.

**Same class as the V100 DSpark TP=1 lockstep-stall KILL** (`docs/baselines.md`
§V100: "the TP lockstep mechanism … on TP=1 the coordinator stalls waiting for
cross-rank acks that never arrive. Needs a TP=1 fast path or the lockstep disabled
when world_size=1"). MTP on single-GPU agent-OPD hits the identical wall.

Separately, `scripts/agent_opd_curve.sh` model fetch was dead — `huggingface-cli
download` is removed in current `huggingface_hub` — masked here by pointing
`STUDENT_MODEL` at a local copy. Fixed to `hf download` the same day.

## Fix

- **Measurement (now):** run single-GPU agent-OPD with **`SPEC=off`** — the
  `gpu_busy_frac` / reward-bearing-ratio measurement is independent of spec-decode
  (spec is a decode-speed lever, not a rollout-structure one), so dropping MTP
  sidesteps the lockstep and lets the serve generate.
- **Engine (follow-up, unbuilt):** a `world_size == 1` fast path for the
  MTP/DSpark lockstep — skip the coordinator when single-rank, or disable the
  ack-watchdog during single-GPU rollout. The proper fix, shared with the V100
  DSpark case.
- **Tooling (done):** `agent_opd_curve.sh` `huggingface-cli download` → `hf download`.

## Rule

MTP/DSpark lockstep has no `world_size == 1` fast path — **single-GPU OPD rollout
must run `SPEC=off`** until one lands. A metric can emit correctly on a dead serve;
gate any `gpu_busy_frac` / ratio conclusion on `completion_tokens > 0` first
(case-as-fact: a degenerate value from a torn-down coordinator is not a measurement).
Pod symbol check caveat: under this release profile (`lto="thin"`, `codegen-units=1`)
short json-key/symbol strings are split `movabs` immediates in `.text`, not
contiguous `.rodata`, so `strings … | grep <short-symbol>` **false-negatives** —
verify a landed change by 8-byte chunks or by running the binary, not full-string grep.
