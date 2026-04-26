# Metal Qwen3.5 dense QK-gate fix — bench pending remote

Status: **pending-remote**

## Goal

Confirm that the explicit `qwen35_compiled_set_qk_gate` flag (commit
`25681a2 fix(metal): qk-gate explicit instead of n_gdr heuristic`) does
not regress decode/prefill on real Qwen3.5 checkpoints. The heuristic
swap is logically a no-op for every previously working configuration —
Qwen3.5 with GDR keeps `gate=true`, Qwen3 keeps `gate=false` — but a
dedicated bench tick confirms the C++ compiled prefill/step path ships
identical numbers.

## Hypothesis

No measurable delta on `Qwen3.5-0.8B` (`scripts/bench_guidellm.sh
metal-qwen35-0p8b`) vs the 2026-04-26 packed-GGUF baseline at
`docs/experience/wins/2026-04-26-bench-metal-qwen35-0p8b-packed-gguf-local.md`.

## Why not local now

The fix landed off a Metal regression in `cli_tiny_fixture_live`, which
exercised a synthetic 2-layer 32-hidden checkpoint. A real
`scripts/bench_guidellm.sh` run requires the standard
`models/Qwen3.5-0.8B` weight set; on this Mac the next regular Qwen3.5
sweep covers it without extra setup.

## Linked entries

- Errors: `docs/experience/errors/2026-04-26-metal-qwen35-dense-qk-gate-heuristic.md`
- Prior bench: `docs/experience/wins/2026-04-26-bench-metal-qwen35-0p8b-packed-gguf-local.md`

## Acceptance

| Metric | Pre (2026-04-26 baseline) | Post | Δ% |
|--------|--------------------------|------|-----|
| TTFT | _pending_ | _pending_ | _pending_ |
| Decode tok/s | _pending_ | _pending_ | _pending_ |
| RSS | _pending_ | _pending_ | _pending_ |

Mark this entry resolved by replacing the table when the next sweep
lands and cross-linking from the wins README.
