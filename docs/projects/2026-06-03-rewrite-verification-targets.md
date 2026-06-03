# infer/ rewrite — verification targets ("done" definition)

**Branch:** `arch/ideal-inference-engine`
**Companion:** [`infer-clean-rewrite-plan.md`](2026-06-03-infer-clean-rewrite-plan.md),
[`aipc-pivot-and-northstar.md`](2026-06-03-aipc-pivot-and-northstar.md)

The rewrite is "done" when the crate-split engine (`infer-plan` / `infer-seam` /
`infer-core` / `infer-{metal,cuda,...}` / `infer-server`) **replaces** `infer/src`
and clears every gate below. No half-states: old `infer/src` is deleted only after
the new tree passes its matching gates.

## G0 — structural (compiler + grep, every commit, local)

- `infer-core` contains **zero** backend tokens: `grep -rinE 'cuda|metal|torch|cudarc|tilelang|nccl|device' crates/infer-core/src` is empty.
- `infer-core` depends only on `{infer-plan, infer-seam, anyhow, log}` — no backend crate.
- `cargo check -p infer-plan -p infer-seam -p infer-core` green on the Mac (no GPU).
- Adding a backend = a new `infer-<backend>` crate; `infer-core` does not recompile against it.

## G1 — engine-core correctness (CPU, local, every step)

- `cargo test -p infer-core` green, including: overlap (NotReady→Ready), priority
  admission, pool-full hold/admit, prompt-length reject, preempt-requeue,
  finish-frees-slot, radix prefix reuse (once ported), chunked-prefill (once ported).
- A CPU mock executor + mock KvPool drive every scheduling path with no GPU.

## G2 — backend parity vs current tree (per hardware, before cutover)

Each backend, on its matching hardware (see
[[project_infer_rewrite_and_verification_routing]]), must match the **current
`main` tree** within tolerance:

| Backend | Hardware | Gates |
|---|---|---|
| **Metal** (primary) | local Mac, Qwen3.6-35B-A3B-4bit MoE | `greedy_consistency`; `e2e` parity vs current; needle/coherence on a long prompt |
| **CUDA** | V100 (Qwen3/3.5) | `e2e` + `kv_precision_parity` (BF16 vs INT8/FP8/TQ4) |
| **CUDA / FlashMLA** | H20 (DSv4) | DSv4 `e2e` + needle retrieval to ≥2047; FlashMLA decode parity |

Must-preserve hard-won behaviors (regression test each on port): DSv4 output
inverse-RoPE; hybrid partial-prefix→MISS downgrade; chunked-prefill 3 hit modes;
decode retract heuristic; FP8-KV known routing; TileLang warp23 NaN bypass;
Metal wired-limit auto-pin; prefill-cap-8 multi-shape default.

## G3 — AI PC north-star (the new acceptance metric, local Metal)

Replaces tok/s sweeps as the headline. An **agent-workflow bench harness** runs a
representative multi-turn agent task (tool calls + code edits + retrieval) against
the engine and gates on **both** axes:

- **Task axis:** end-to-end task completion time; per-turn TTFT (interactivity);
  cross-turn KV-reuse hit rate. Must not regress vs the current Metal path.
- **OS-impact axis (PASS/FAIL — "不影响 OS 使用"):** while the workflow runs, a
  concurrent foreground-responsiveness probe must stay under an input-latency
  threshold; peak memory must stay under the wired-limit headroom; no CPU
  busy-spin (scheduler tick must not peg a core — cf. the H5 `cuEventQuery` spin).
  `ResourceGovernor` is the mechanism; this gate proves it works.

## G4 — performance no-regression (per backend, pending-remote where needed)

- `bench_guidellm` on each backend vs the latest pre-rewrite baseline: TTFT / ITL /
  output-throughput Δ% within noise at the binding SLO shape (CLAUDE.md §Benchmarks).
- Remote legs (V100/CUDA, H20/DSv4) run via Codex-in-tmux on the pod; stub
  `wins/` with `pending-remote` until the run lands, then attach Δ%.

## Cutover rule

Delete `infer/src` and flip `infer` to the new crates in **one tranche** only after
G0–G3 pass on Metal locally and G2/G4 pass on V100 + H20. Until then the new tree
lives beside the old; no parallel old+new serving paths ship
([[feedback_no_half_states]]).
