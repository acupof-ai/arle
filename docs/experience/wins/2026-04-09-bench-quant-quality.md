# 2026-04-09 · Weight Quantization Quality & Throughput Benchmark

## Context

First systematic quality + throughput comparison across all quantization formats on Qwen3-4B.
L4 GPU (23GB VRAM), num_slots=1, greedy decode, 100 token generation.

## Results

| Format | group_size | Speed (tok/s) | vs BF16 | Model Size | Quality |
|--------|-----------|--------------|---------|------------|---------|
| BF16   | -         | 30.3         | baseline | 7.6 GB     | ✅ Coherent |
| W8A16  | 128       | 50.4         | +66%    | 4.2 GB     | ✅ Matches BF16 |
| W4A16  | 128       | 50.8         | +68%    | 2.5 GB*    | ✅ Matches BF16 |
| W4A16  | 32        | 48.5         | +60%    | 2.7 GB*    | ⚠️ Off-topic but coherent |
| W2A16  | 32        | 50.8         | +68%    | 1.8 GB*    | ❌ Repetitive garbage |

*On-disk size. Runtime memory is higher due to INT4/INT2 → INT8 unpack at load.

## What Worked

- W4-g128 achieves same quality as W8 with half the disk size
- All quantized formats get ~50 tok/s (+66% over BF16) via W8 GEMV kernel
- INT4/INT2 → INT8 unpack workaround is reliable and correct
- concat_rows dummy allocation for quantized weights saves ~3.5 GB VRAM

## Update: Native W4 GEMV Enabled (same session)

Root cause found: single-request decode path used BF16-only `ops::gemv` instead of
quantized-aware dispatch. Native W4 GEMV now works for all paths.

| Format | Speed (tok/s) | vs BF16 | vs W8 |
|--------|--------------|---------|-------|
| BF16   | 30.4         | baseline | -     |
| W8A16  | 51.4         | +69%    | baseline |
| W4A16 (native, g128) | 72.3 | +138% | +41% |

## Limitations

- Scheduler's paged attention path produces numerically divergent greedy output vs single-request engine (pre-existing)
- Marlin kernel integration remains an option for further W4 prefill optimization

## Rule

For W4A16: use group_size=128, not 32. Quality matches W8 with half the model size.
W2 is not viable for 4B models — too much precision loss.
