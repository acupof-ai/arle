# Real Ctrl-C (per-request cancel token) + warm-load parity for the in-process CLI

## Context

Two in-process CLI defects, both root-caused from source (no daemon / transport
change — the CLI already shares the scheduler runtime with `metal_serve`):

- **Dead Ctrl-C** — SIGINT didn't interrupt generation; the REPL hung until
  `max_tokens`. Root cause:
  [`errors/2026-05-31-dead-ctrlc-no-cancel-token-threadscope-wedge.md`](../errors/2026-05-31-dead-ctrlc-no-cancel-token-threadscope-wedge.md).
- **Cold in-process load** — `loaded.rs` never pinned weights
  (`mlx::set_wired_limit`), unlike `metal_serve` (which auto-pins). The
  in-process `arle` path missed the warm-load that drops c=1 p99 86→15 ms.

## What Worked

**Cancel token** — `Arc<AtomicBool>` threaded end-to-end:
`repl install_ctrlc_handler` → agent interruptible API (changed `&AtomicBool`
→ `Arc<AtomicBool>`) → `CompletionRequest.cancel` → `IncomingRequest.cancel`
(new field; `cancel: None` at all other ~13 construction sites incl. HTTP, which
keeps socket-close) → Metal runtime per-request state. New `cancel_requested()`
= `delta_closed() || cancel.load(Relaxed)` replaces bare `delta_closed()` at the
runtime reap / prefill-chunk / decode pre-checks. On Ctrl-C the runtime stops
generation → `delta_tx` closes → the agent's `thread::scope` worker returns →
`join()` unwedges. CUDA path: additive field, `cancel: None`, compiles
(`cargo check --features cuda,no-cuda` green); no CUDA tick-loop wiring (out of
scope, concurrently refactored).

**Warm-load** — moved `auto_wired_limit_bytes` from the `metal_serve` *binary*
into a lib module `infer/src/backend/metal/wired_limit.rs` (reachable as
`infer::backend::metal::auto_wired_limit_bytes`); `loaded.rs::load` now builds
`MetalBackendOptions{ runtime_limits.wired_limit_bytes: auto_wired_limit_bytes(..) }`
so the in-process CLI pins weights exactly like `metal_serve`. (The `0` arg to
`spawn_metal_scheduler_handle_from_path` is `max_waiting`, **not** the wired
limit — pinning flows through `MetalRuntimeLimits::apply` → `mlx::set_wired_limit_bytes`.)

Tests: 34 `chat` + 106 `http_server` + 33 `agent` + 10 metal-runtime lib green;
`metal_serve` + `arle` build clean (metal,no-cuda); cross-backend check green.

## Verification — Ctrl-C is real (behavioral, not just compiles)

Drove the live `arle` REPL (Qwen3.6 Metal, M4 Pro): started a 4000-token
generation, fired SIGINT mid-stream. Token streaming was flowing ~383 B/s;
**post-SIGINT growth = 0 B over the following 2 s** — generation halted within
~1 s, stopped at 256/4000 tokens. Without the fix it would have streamed the
full 4000 tokens (~48 s more). Confirmed `metal_serve` still serves with no
regression (warm c=1 decode 73.5 tok/s, unchanged from the tranche-1 baseline —
cancel adds one atomic load per tick; warm-load only affects the in-process
path).

Warm-load TTFT impact and the prefill-vs-mlx-lm gap are quantified in the A/B
entry (separate, in progress).

## Rule

Closing in-process CLI gaps (Ctrl-C, cold load) is a request-contract +
runtime-loop change, not a transport/daemon change — the CLI already runs the
full scheduler in-process. Thread cancel to the producer loop and verify the
stream actually stops; pin weights in *every* load path, not just the server bin.
