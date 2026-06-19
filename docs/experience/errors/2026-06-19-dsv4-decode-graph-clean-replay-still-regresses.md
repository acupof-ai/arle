# DSv4 decode CUDA graph regresses −41% EVEN with a clean (0-alloc) replay — launches are not the B=1 bottleneck

## Context
DSv4-Flash B=1 no-spec decode on 8×H20 TP4. Foundation = 26 ms/step (38 tok/s).
nsys host-trace: GPU kernels ~1 ms/step, `cudaLaunchKernel` ~1.3 ms/step,
`cuStreamSynchronize` rare → ~20 ms/step is NON-CUDA host. Hypothesis (revisited,
on the belief that prior graph kills were just bad implementations): a CLEAN
capture-once-replay CUDA decode graph removes the per-step host/launch
orchestration → big win.

## What was built (the "done well" attempt)
A full persistent-scratch rewrite of `forward_tokens_decode_graph` (attention +
MoE + tail + whole-step captured bodies): every per-step `DeviceVec::zeros` /
`HiddenStates::uninit` moved out to persistent fields on the slot
`CapturedDecodeGraph`; added `head_hidden_from_stream_into` / `gen_mhc_params_into`
scratch variants. **Confirmed clean:** the graph.rs audit dropped from **176
alloc-node warnings → 0** (0 alloc, 0 host-coupled nodes). Capture-once, replay.

## A/B (same binary, same shell, two env flips, steady-state 3×512 essay)
| | tok/s | ms/step | alloc-warns | output |
|---|---:|---:|---:|---|
| `DECODE_GRAPH=0` (eager) | 38.1 | 26.22 | 0 | coherent |
| `DECODE_GRAPH=1` (clean replay) | **22.3** | 44.76 | **0** | coherent |

The clean-replay graph is **−41% (+18.5 ms/step)**, byte-coherent.

## Root cause / Rule
- **B=1 decode launches are NOT the bottleneck.** A graph that removes them
  cleanly (0 alloc/host nodes, proven) still regresses. The ~1.3 ms/step of
  `cudaLaunchKernel` is dwarfed by what the graph CANNOT remove and by what it
  ADDS: per-step graph-input update (new token / position / KV pointers injected
  into the captured nodes) + the TP4 multiproc coordinator↔worker round-trip +
  the engine-loop per-step host orchestration. Net +18.5 ms.
- **The ~20 ms/step foundation cost is multiproc IPC + per-step host orchestration**,
  not kernel launches and not GPU compute (GPU is ~1 ms = 3% active). A per-worker
  CUDA graph cannot touch the cross-process coordination.
- **This supersedes the (mistakenly deleted) 2026-06-08 entry with stronger evidence:
  the wash is not an artifact of a bad graph impl — a measured-clean (0-alloc)
  replay regresses −41%.** Do NOT default-on `ARLE_DSV4_DECODE_GRAPH`; do not
  re-attempt the decode graph as a B=1 lever without first removing the multiproc /
  host-orchestration cost.
- **Lever that DID move B=1/c8: MTP multi-token draft** (amortization, tok/step>1) —
  enabled + made default (`spec-type=auto`, dt=3): no-spec 38 → MTP 41 (essay) /
  c8 51.7 → 62.8 (+22%). The foundation (ms/step) lever, if pursued, is the
  multiproc IPC + engine-loop host path, not the kernels.

## Validated-method caveat (for the record)
CUDA graphs ARE industry-standard for decode (vLLM/TRT-LLM/SGLang) — but those are
single-process or tightly-fused-device-resident loops. The regression here is
specific to ARLE's multiproc TP4 + per-step host orchestration, which the graph
doesn't address. "Validated method fails = our bug" held: the bug was assuming the
bottleneck was launches; the measurement (clean-graph still −41%) corrected it.
