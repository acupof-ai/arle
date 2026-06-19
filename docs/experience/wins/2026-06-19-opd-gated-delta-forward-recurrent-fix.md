# OPD gated-delta device forward — route to recurrent, fixes sm_90 chunk-WGMMA hang

## Context
Commit `08e505d7` added a CUDA gated-delta-rule (GDR) forward for the OPD
autograd path (`cuda_linear_attention_forward_device`, backend_cuda.rs). On
H20 (sm_90) the OPD student forward **hung >120s at GPU 100%**. Measured
(tmux1, GPU7, per-stage timing): conv/prepare/cumsum/a/solve complete in ms,
the wedge is `gated_delta_rule_prefill_chunk_recompute_cuda` — a real
launch-then-sync **deadlock** (FFI smoke confirmed it is not a stub).

## Root cause
The autograd GDR forward called the 7 `gated_delta_rule_prefill_chunk_*`
FullRow-WGMMA stages **unconditionally**, bypassing the hang-gate the
inference path installs at `gdr_prefill_batch.cu:152`
(`seq_len > 32 || !gdr_chunkwise_prefill_enabled()` → recurrent). That gate
exists precisely because those chunk-WGMMA kernels deadlock on sm_90
(`errors/2026-05-30`, `errors/2026-06-04`). The autograd path had no gate and
no fallback. Static review (Workflow H1) and runtime measurement (tmux1)
double-confirmed the same root cause from opposite directions; a purely-static
reviewer (codex) could not see the hang — it needed the gate-bypass structural
insight plus the on-GPU measurement.

## What Worked
Mirror the inference dispatch instead of fixing the WGMMA codegen:
- `backend_cuda.rs:3623` — autograd GDR forward defaults to the device
  **recurrent** kernel; the chunk-WGMMA path is enabled only under
  `seq_len <= 32 && ARLE_GDR_CHUNKWISE_PREFILL=1`, matching inference.
- `backend_cuda.rs:3834` — the recurrent forward writes its state into
  `chunk_state` at each 64-token chunk boundary, preserving the chunked-scan
  backward's chunk-boundary input contract.
- `linear_attention.cu:533` — the backward reads the forward-saved BF16
  `qkv_conv` instead of reconstructing q/k/v from `preact`+silu, aligning the
  backward with the recurrent forward trajectory.
- `test_linear_attention.rs:337` — the CUDA grad-check no longer silently
  `return Ok(())` on non-CUDA boxes; it emits per-stage timing and per-gradient
  rel/abs error (closes the "pass-as-skip never ran on a GPU" validation gap).

## Validation (correctness LICENSED)
GPU7, before the box was reclaimed:
- `cargo test -p autograd --release --features cuda --test test_linear_attention -- --nocapture` → **5/5 PASS**; log routes through `gdr_recurrent`, never enters `gdr_recompute` (hang gone).
- `cargo test -p autograd --release --features cuda` → all pass, **EXIT 0**.
- Local: `cargo check`, `cargo clippy -D warnings`, `git diff --check` clean.

## Pending-remote (perf A/B)
The wall-clock win (device recurrent vs the host GDR recompute that dominates
the current A/B's 56.6s base_backward / 233s step) is **not yet measured** —
the H20 box is in use by ckl. Perf license deferred: re-measure base_backward
+ step time on the recurrent binary, A/B vs the host-path baseline, before any
throughput claim. This entry is `pending-remote` on the perf axis;
correctness is licensed above.

## Rule
A device forward that reuses kernels which the inference path **gates off** for
a known hang must inherit the same gate + fallback, not call them raw. When a
runtime hang resists static review, the gate-bypass structural insight (which
guard does the working path have that this one skips?) plus an on-GPU
per-stage timing run localize it together — neither alone suffices.
