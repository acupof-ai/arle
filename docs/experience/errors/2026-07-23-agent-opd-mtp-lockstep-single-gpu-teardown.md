# agent-OPD single-GPU serve tears down: local relay driver blocked acks

## Context

Validating the new `gpu_busy_frac` timer on the pod: `SMOKE=1
scripts/agent_opd_curve.sh` on one H20, ThinkingCap-Qwen3.6-27B-FP8. Two runs,
both dead the same way: all round-0 rollouts return `API Error: 500 coordinator
lockstep loop closed`, `completion_tokens=0`, so the emitted `gpu_busy_frac ≈ 0`
reflects a dead serve, not a measurement.

## Root Cause

Two hypotheses, resolved by a controlled A/B:

1. **REFUTED — "MTP lockstep is TP≥2-only" (first run, SPEC=mtp).** The SPEC=off
   control run tore down with a byte-identical signature (tick ~1.82M, exactly
   `TICK_WINDOW=4` acks behind, 120 s watchdog at `coordinator.rs:183`). Spec
   decode is not the trigger.
2. **CONFIRMED at the code level — ack liveness coupled to a blocking submit.**
   Single-GPU (TP=1) serve = `coordinator_local_router` → in-process
   `serve_handle_relay_driver`, the only ack sender. Its `TickAdmissions` arm
   called `serve.submit_streaming()` inline — which blocks in `handle_rx.recv()`
   until the engine thread drains the submission. One long engine step
   (first-touch JIT / giant prefill chunk, or a genuine engine hang — which of
   the two the pod hit is still unconfirmed) → the driver blocks submitting the
   next request → acks stop → coordinator races ahead exactly `TICK_WINDOW=4`
   (matches the log) → 120 s `ACK_STALL_TIMEOUT` → deliberate teardown → every
   subsequent submission 500s.

Separately, `scripts/agent_opd_curve.sh` model fetch was dead — `huggingface-cli
download` is removed in current `huggingface_hub`; fixed to `hf download`.

## Fix

- **Landed:** `serve_handle_relay_driver` split into a pump (recv → ack, never
  blocks) and a dedicated `arle-local-relay-submitter` thread that does all
  engine-touching work (submit / cancel / stats). Acks are liveness, not
  admission. Regression gate:
  `infer_server::tests::relay_driver_acks_ticks_while_engine_is_wedged`.
- **Resolved by the rerun (busytimer-fixed, HEAD 55684cc66):** the long engine
  step was neither JIT nor a hang — the serve survived, decoded real turns,
  then died on **device KV-pool exhaustion, fatal at submit**
  (`TokenKVPool: out of pages`, `execution.rs:310`): 8 concurrent CC sessions
  × ~31K-token **prompts** (decoded sidecars: generations are 12–96 tokens,
  coherent; the 31K is the CC harness preamble) ≈ 248K ≈ the 250K pool.
  **Landed (a9d0c5412):** seam hook `kv_device_free_pages()` + a device-budget
  gate in `allocate_for_plan` — device exhaustion now sheds/parks (#162/#164
  path) instead of killing the engine thread. Gate:
  `infer_core::tests::device_pool_exhaustion_degrades_instead_of_fatal`.
- **Not implicated:** the TP≥2 lockstep path (`serve_multiproc.rs` workers ack
  after each engine step; pacing there is intentional and untouched); #48
  FP8-temp>0 (decoded generations are clean at temp=1.0).

## Rule

A flow-control ack path must never contain a call that blocks on the thing it
paces — in the local relay topology the driver acks, so every engine-blocking
call belongs on a separate thread. A metric can emit correctly on a dead serve:
gate any `gpu_busy_frac` / ratio conclusion on `completion_tokens > 0` first
(case-as-fact). A plausible mechanism ("MTP lockstep is TP≥2-only") is not a
root cause until a controlled variable flip (SPEC=off) confirms it — here the
control refuted it. Pod symbol-check caveat: under `lto="thin"` +
`codegen-units=1`, short strings are split `movabs` immediates in `.text`, so
`strings … | grep` false-negatives — verify by 8-byte chunks or by running the
binary.
