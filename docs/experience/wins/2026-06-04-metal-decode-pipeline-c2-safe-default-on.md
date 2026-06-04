# Metal cross-step decode pipeline → DEFAULT ON: c≥2 provably fall-safe, c=1 win + bit-identity hold (matched A/B)

## Context

The cross-step decode pipeline landed opt-in (`INFER_METAL_PIPELINE=1`, default
OFF) in `ca7cea72`
([`2026-06-04-metal-rewrite-decode-pipeline-recovery.md`](2026-06-04-metal-rewrite-decode-pipeline-recovery.md)):
it recovers the rewrite Metal decode regression at c=1 (Qwen3.6 +12.9%,
Qwen3.5-0.8B +25%, greedy bit-identical) by holding the decode session one step
ahead. The commit deferred the default flip "pending a c≥2 fall-to-HEAD
validation" — the gate this entry clears.

Goal: make the pipeline the DEFAULT so the c=1 local Metal focus gets the win
without a flag, **only if c≥2 is provably safe**. The single-slot-greedy
per-step gating (`pipeline_decode_enabled()` + `pending_matches_live_slot`) must
keep c≥2 / non-greedy / recycled-slot off the fast path.

## Params / Env

- M4 Pro, MLX (rewrite `infer-metal`, `Engine<MetalExecutor, MetalKvPool>`).
- Models: `mlx-community/Qwen3.5-0.8B-MLX-4bit` (fast), canonical
  `mlx-community/Qwen3.6-35B-A3B-4bit` (MoE, ~19 GB).
- Harness: `agent-bench` — c=1 via `bench_agent_workflow_metal_qwen3{5_08b,6_canonical}`
  (`pure_decode_tok_s` split); c≥2 via the new `drive_concurrent` +
  `metal_c2_fall_to_head_*` tests (submit 2 greedy requests at once,
  `num_slots=4`). Matched A/B = same binary, same shell, env-only flip.
- Runtime probe: new `infer_metal::pipeline_fast_path_hits()` atomic counter —
  proves which decode path each step took (fast-path firings).

## Results

### Step 1 — c≥2 is fall-SAFE (the gate)

**Key architectural finding:** the rewrite Metal executor accepts **exactly one
row per plan** (`executor.rs` `submit`: `row_count == 1` guard, pre-dates the
pipeline). The engine planner (`infer-core/planner.rs`) **does** batch ≥2 rows
into one tick when ≥2 requests are active (CPU test
`concurrent_plan_batches_multiple_rows`: max_rows ≥ 2). So a genuine concurrent
plan is multi-row and hits that guard **before any pipeline logic** — it does
not "fall to a HEAD decode path", it errors LOUD (never silently mis-decodes).

Matched A/B, two greedy requests submitted at once, pipeline OFF vs ON, both
models:

| run | env | step_error | pipeline_hits_delta | fingerprint |
|-----|-----|-----------|---------------------|-------------|
| Qwen3.5 | OFF | `…supports exactly one prefill or decode row, got 2` | 0 | `0x0a99c907b6f64763` |
| Qwen3.5 | ON  | `…got 2` (identical) | 0 | `0x0a99c907b6f64763` |
| Qwen3.6 | OFF | `…got 2` | 0 | `0x0a99c907b6f64763` |
| post-flip default (no env) | ON | `…got 2` | 0 | `0x0a99c907b6f64763` |
| post-flip opt-out (`=0`)   | OFF | `…got 2` | 0 | `0x0a99c907b6f64763` |

The pipeline flag has **zero effect at c≥2**: identical loud error, identical
(empty) output, fast path fires 0× concurrently. No hang, no panic. The flip
cannot newly break c≥2 because c≥2 is a uniform error-out regardless of the flag
(a pre-existing single-row constraint, not a pipeline behavior).

### Step 2 — c=1 win + bit-identity hold post-flip (matched A/B, same binary)

`pure_decode_tok_s`, default-on now means no env = ON, opt-out = `INFER_METAL_PIPELINE=0`:

| model | OFF (HEAD) | ON (default) | Δ |
|-------|-----------|--------------|---|
| Qwen3.5-0.8B (3 runs) | 245.1 / 244.6 / 243.6 → **244.4** | 294.8 / 294.4 / 292.5 → **293.9** | **+20.3%** |
| Qwen3.6-35B-A3B-4bit (2 runs) | 70.2 / 73.8 → **72.0** | 82.3 / 76.6 → **79.5** | **+10.4%** |

- `pipeline fast path LIVE` + `decode pipeline = true` fire only in the ON arm.
- ON's worst beats OFF's best in both models (293.9>>244, 76.6>73.8) — win is
  above the 10% Metal noise floor.
- Greedy **bit-identical** (c=1, cross-process): both arms emit fingerprint
  `0xdb74ad23d24c46f4` for a fixed prompt → 32 tokens. ON shows
  `pipeline_hits_delta=30` (cold-seed + tail don't fast-path), OFF `=0`.

### Gates

- `cargo test -p infer-metal --features metal`: 6/6.
- `cargo test -p infer-core -p infer-seam`: 28 + 0, green.
- `cargo test -p agent-bench` (default, no-metal): 6/6 (new CPU plan-shape test
  included).
- clippy `-D warnings` clean on `infer-metal` (metal + default) and `agent-bench`.
- `CUDARC_CUDA_VERSION=12060 cargo check -p infer-cuda --features cuda,no-cuda`:
  green — seam untouched, infer-cuda not modified.

## Change

`pipeline_decode_enabled()` flipped to default ON: env absent → `true`;
`INFER_METAL_PIPELINE=0`/`false` → `false` (opt-out). Single-slot-greedy
per-step gating unchanged. Added a test-readable `PIPELINE_FAST_PATH_HITS`
atomic + `pipeline_fast_path_hits()` (one relaxed counter on an already-rare
event; production-harmless). `agent-bench` gained `drive_concurrent` +
`ConcurrentResult` (c≥2 driver, FNV fingerprint) and the c≥2 / c=1-fingerprint
validation tests.

## Rule

- **"Falls to HEAD at c≥2" was the wrong mental model — verify the actual code
  path.** The rewrite Metal executor has NO multi-slot decode: a concurrent plan
  errors at the `row_count == 1` guard, it does not run a HEAD decode. The
  default-flip is safe because c≥2 is a uniform loud error with or without the
  pipeline, proven by a matched A/B (identical error string + 0 fast-path hits +
  identical fingerprint), not because a fall-through path was exercised.
- **A default flip needs the runtime probe to show the changed path fires (c=1
  ON: 30 hits) AND stays silent where it must (c≥2: 0 hits; c=1 OFF: 0 hits).**
- Matched A/B (same binary, env-only flip, ≥3/≥2 iterations) gave a tighter,
  more trustworthy delta (+20.3% / +10.4%) than the commit's cross-day numbers.
