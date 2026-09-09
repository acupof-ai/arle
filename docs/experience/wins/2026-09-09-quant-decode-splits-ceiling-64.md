# Quantized decode split ceiling 16 → 64 — CUDA, 2026-09-09

> Status: pending-remote (no CUDA device on the authoring machine; the change
> is complete and type-checked, the measurement is not)

## Goal

Decode tok/s at B=1 on the FP8-KV paged decode lane, Qwen3.8-27B on H20.

## Hypothesis

The quantized paged decode kernel's grid is `(kv_heads * splits, batch,
q_tiles)`. The split selector (`quant_decode_num_splits`,
`crates/infer-cuda/src/qwen35_attention.rs`) is occupancy-driven
(`sm_count / (batch * kv_heads)`, floor 8) but was clamped at 16 by three
sites: the kernel's `kMaxSplits` (`paged_attention_quantized_fa3.cu`), the
pool workspace (`paged_kv.rs`), and the Rust clamp. On H20 (78 SMs, 4 KV
heads) B=1 picks 20 splits and was clamped to 16 — 64 blocks, under one
wave. Raising the ceiling to 64 lets B=1 run 20 splits (80 blocks). B=8 is
unchanged by construction (floor 8 binds), which is the clean-attribution
control arm.

Direction and rough magnitude come from tileRL's matched A/B on a
comparable split-KV decode kernel (16 → 64 splits at B=1: +1.7% / +1.1% /
+1.9% at d512 / d8192 / d32768; B=8 unchanged —
`tileRL/docs/experience/wins/2026-08-28-decode-split-by-occupancy.md`,
read-only reference). Our selector moves 16 → 20, not 16 → 64, so the
magnitude is unknown and this entry is the gate.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8000 \
  --model Qwen3.8-27B \
  --prompts-jsonl bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1,8 \
  --requests-per-concurrency 16 \
  --max-tokens 214 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output bench-output/quant-splits-64/bench
```

- Baseline: archived pre-change binary (commit before this one), same flags
- Treatment: this commit, same flags
- Server flags: `--kv-cache-dtype fp8`, otherwise default
- Trials: ≥3 per arm, same shell, side by side (spec §3.1)
- Unit gate: `cargo test -p infer-cuda --release --features cuda
  quant_decode_num_splits` (links only on a CUDA host; on Mac the infer-cuda
  test binary does not link — pre-existing, `_cublas_init` et al.)

## Environment

- Host / GPU: H20 (sm_90, 78 SMs)
- Model / dtype: Qwen3.8-27B, FP8 KV cache
- TP / slots / KV: TP=1, default slots, paged KV page_size 16
- Required card: any sm_90 H20; the selector reads SM count from the device

## Results

| concurrency | arm | completed | errors | decode tok/s | TTFT p50/p99 ms | ITL p50/p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | | | | | | — |
| 1 | treatment | | | | | | |
| 8 | baseline | | | | | | — |
| 8 | treatment | | | | | | |

Expected: c=1 treatment ≥ baseline (tileRL: +1.1–1.9% at 16→64; ours is
16→20); c=8 identical within noise (floor 8 binds both arms). Also record
free VRAM at boot both arms: the quantized-attn workspace is sized for 64
splits now (4× the old allocation, one per model, tens of MB at this shape).

Correctness gate before timing: needle ladder ×3
(`scripts/needle_gate.py` + `scripts/lever_gate.sh`), 18/18 exact, every
length deterministic.

Raw artifacts: `<json>`, `<csv>`, `<server log>`.

## Change list

- `crates/cuda-kernels/csrc/attention/paged_attention_quantized_fa3.cu`:
  `kMaxSplits` 16 → 64
- `crates/cuda-kernels/src/paged_kv.rs`: workspace sized for 64 splits
- `crates/infer-cuda/src/qwen35_attention.rs`: selector extracted as
  `quant_decode_num_splits`, clamp 16 → 64 (`QUANT_DECODE_MAX_SPLITS`)
- `crates/infer-cuda/src/executor/qwen35.rs`: stale comment fixed (it named a
  nonexistent `choose_decode_num_splits`; the real decider is the selector)

## Rule

A split count is an occupancy knob before it is a scan-length knob. Size the
decode grid against the SM count of the card you are on, and let the ceiling
follow the grid arithmetic — a clamp below the occupancy pick re-introduces
the under-fill the selector exists to prevent.
