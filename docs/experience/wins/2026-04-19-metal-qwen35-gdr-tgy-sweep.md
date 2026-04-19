---
name: Metal Qwen3.5 GDR threadgroup_y causes odd-c valleys
description: GDR decode kernel tg_y=4 (default) creates wave-pack valleys at c=3/5/7; tg_y=2 flattens them with only a ~2% cost at even c. Actionable default-flip candidate.
type: project
---

# GDR kernel `threadgroup_y` controls the c=3/c=5 odd-c valleys

**Date**: 2026-04-19
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA)
**Model**: `mlx-community/Qwen3.5-4B-MLX-4bit` (Hv=32, Dv=128, 24 GDR layers)
**Binary**: `metal_serve` on HEAD `6b4f305`
**Bench**: `scripts/bench_throughput.py` synthetic N=24, max_tokens=128, temp=0
**Knob**: `AGENT_INFER_QWEN35_CPP_DECODE_GDR_TG_Y` (default 4)

## Context

`2026-04-19-metal-qwen35-concurrency-sweep-c3-c5-valleys.md` found that
c=3 and c=5 underperform surrounding concurrencies. The GDR decode kernel
(`mlx_qwen35_model.cpp:88`) dispatches grid `(32, Dv, B*Hv)` with
threadgroup `(32, tg_y, 1)` — tg_y sets how much of the Dv axis each
threadgroup covers. Hypothesis: at the default tg_y=4, the resulting
threadgroup count per GDR layer (`1024 · B`) packs into GPU waves in a
way that creates fill-holes at specific odd B. tg_y=2 doubles the TG
count per B and should reshuffle that packing.

## What Worked

Swept c=1..8 at three tg_y values in the same thermal-matched session.

### Throughput (tok/s)

| c | tg_y=2 | tg_y=4 (default) | tg_y=8 |
|---|--------|------------------|--------|
| 1 | 73.0   | 70.9             | 74.4   |
| 2 | 127.8  | 130.2            | 133.7  |
| **3** | **154.4** ← +10.0% | 140.4 | 139.7 |
| 4 | 157.1  | **160.3**        | 147.6  |
| **5** | **156.4** ← +4.7%  | 149.4 | 154.2 |
| 6 | 157.8  | **160.1**        | 154.2  |
| **7** | **162.7** ← +2.2%  | 159.2 | 153.7 |
| 8 | 158.3  | 153.5            | **163.3** |
| **mean** | **143.4** | 140.4 | 140.1 |

### Findings

1. **tg_y=2 flattens the odd-c valleys.** c=3 +10%, c=5 +5%, c=7 +2%.
   The tg_y=4 "valley" structure is entirely a packing artifact.
2. **tg_y=2 costs ~2% at even c** (c=2/4/6/8 all drop a little).
   Net arithmetic mean across c=1..8 is **+2.1% at tg_y=2 over default**.
3. **tg_y=8 wins only at c=1/2/8** — it moves the valleys to different c
   values (c=3, c=4). Not a clean win.
4. **Per-c optimum is parity-sensitive**: odd B→tg_y=2, even B→tg_y=4.
   Dynamic selection based on `ctx.batch_size % 2` would capture nearly
   all per-c peaks (peak-of-peaks mean = 145.7 tok/s, only ~1.6% above
   fixed tg_y=2).

## Ceiling shape at tg_y=2

c≥3 sits in a 154–163 tok/s band — essentially flat. The "GDR kernel
6.1 ms/row" ceiling from the final-state doc is the same underlying
limit, but with the wave-packing artifact removed the curve monotonically
approaches it instead of dipping.

## Proposed follow-ups (for next /loop iter)

**Cheap, low-risk win**: change the decode-path default from 4 to 2 in
`qwen35_cpp_gdr_threadgroup_y` (`crates/mlx-sys/src/mlx_qwen35_model.cpp:80`).
One hardcoded constant; prefill default stays at 4 (untested for prefill
and the cost profile is different with S>1).

**Slightly better**: parity-adaptive tg_y — `B % 2 == 1 && B > 1 → 2`,
else 4. Requires passing `batch_size` into the helper. ~5-line change.
Captures +1-2% over fixed tg_y=2.

Both options are bench-only changes — no scheduler, no trait shape, no
state invariants touched.

## Rule

**GPU-kernel tile constants deserve a c-sweep before being baked.** A
single c=4 bench looks great at tg_y=4 (peak 160 tok/s), but every odd c
loses 5-10% to wave-packing artifacts. Sweep odd AND even c values at
multiple tile shapes when landing any new Metal kernel default —
power-of-2 benching misses the most common failure mode.

## Cross-references

- `2026-04-19-metal-qwen35-concurrency-sweep-c3-c5-valleys.md` — c-sweep at default tg_y that surfaced the valleys
- `2026-04-19-metal-qwen35-final-state.md` — prior c=1/2/4/8 ceilings at tg_y=4
- `crates/mlx-sys/src/mlx_qwen35_model.cpp:79-85` — `qwen35_cpp_gdr_threadgroup_y` helper and env knobs
- `crates/mlx-sys/src/mlx_qwen35_model.cpp:800-823` — kernel dispatch with the tg_y value
