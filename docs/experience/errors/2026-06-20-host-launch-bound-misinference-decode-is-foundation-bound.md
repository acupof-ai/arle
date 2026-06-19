# "Host-launch-bound" was a mis-inference — DSv4 B=1 decode is foundation-bound (per-step ctx.sync + cross-process lockstep), proven by the graph −41% I already had

## Context

Re-investigating the DSv4-Flash B=1 forward (38 t/s / 26ms vs the historical
no-spec 44 t/s / 22.4ms). The stage profiler showed per-stage host launch time
(`host_ms`) summing to ~22ms ≈ 85% of the 26ms wall (mla_attn alone 18ms host).
I re-framed: "decode is HOST-LAUNCH-BOUND — the host launches ~350 kernels/token,
the GPU overlaps behind it; a CUDA graph should remove the launches and win."

## Root Cause (verified in code)

That re-frame is WRONG, and I had the controlled experiment refuting it the whole
time: my own `errors/2026-06-19-dsv4-decode-graph-clean-replay-still-regresses`
(graph −41% even with a clean 0-alloc replay). Three code facts prove the wall is
NOT the launches:

1. **`crates/infer-cuda/src/ops.rs:467`** — `argmax_into` ends in `ctx.sync()` (a
   full `cuStreamSynchronize`), called from `sample_cuda_token`
   (`executor.rs:1002`) at the end of EVERY decode step. The host blocks on the
   whole GPU chain every token → it never runs ahead → host-launch latency is
   overlapped/hidden, not the wall.
2. **`crates/infer-cuda/src/tp.rs:297`** — the per-layer TP AllReduce "runs in
   place on the compute stream … no cross-stream event is needed." Eager already
   inline-serializes comm; there is no comm/compute overlap for a graph to lose
   (my NCCL-serialized-in-graph hypothesis was also wrong).
3. **`crates/cli/src/serve_multiproc.rs:27`** — TP4 is 4 OS processes with a
   per-tick cross-process `TickAdmissions` relay broadcast + a 4-way GPU barrier
   straggler (`custom_all_reduce.cuh:198-216`). Per-token, cross-process,
   unremovable by any per-worker graph.

So the 26ms wall = the serial GPU chain (43× attn→AR→MoE→AR) + the per-step
ctx.sync + the per-tick cross-process coordination. The ~22ms host-launch
OVERLAPS under it (hidden). The profiler `host_ms` is launch time that overlaps
the GPU in the eager (unprofiled) run — comparing it to the clean wall and
concluding "host-bound" is the inference error; the graph experiment is the
control, and it says foundation-bound.

## Fix (corrected levers, code-grounded — quantify with nsys before picking)

The wall is the foundation, so the levers attack the foundation, not the launches:
- `ops.rs:467` per-step `ctx.sync()` → device-side sampling (token stays on
  device, position advanced by-reference) so the host CAN run ahead — necessary
  before any graph can pipeline.
- `serve_multiproc.rs` 4-process TP → single-process TP (threads, NCCL across
  threads) removes the per-tick relay + cross-PROCESS barrier. The biggest item.
- `tp.rs:297` inline AllReduce → real comm/compute overlap (separate comm stream
  + events) — there is currently NONE.
- MTP / 2-head MTP — amortize the per-token foundation over >1 emitted token.
nsys must quantify the 26ms into {GPU-compute, inline-AllReduce, cross-process
barrier straggler, ctx.sync} to rank these. Do NOT re-attempt the per-worker
decode graph (3 kill records + this analysis converge).

## Rule

A profiler `host_ms ≈ wall` does NOT prove host-launch-bound — in an async
pipeline the launch time OVERLAPS the GPU, so it can be fully hidden. The
DECIDING evidence is the controlled experiment: a CUDA graph removes host
launches; if it doesn't win, the launches weren't the wall. I had that experiment
(−41%) in my own errors log and re-inferred past it. **Controlled experiment >
profiler inference** — when a profiler reading tempts a bottleneck re-frame, run
(or recall) the experiment that isolates it before re-framing. This is the third
KV/perf flip-flop this session; each was a measurement/inference taken without
its isolating control (see also `2026-06-20-dsv4-kv-reuse-metric-fix-ineffective`).
