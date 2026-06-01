# Metal RSS Accounting Note

## TL;DR

The README chart now reports **process RSS high-water** for the whole request
run. In the 2026-06-01 retest after `71494395`, ARLE reached **11.44-15.47
GiB** and mlx-lm stayed around **7.23-7.24 GiB**.

The older **2.2 GiB** and **2.5-4.1 GiB** ARLE numbers were request-window
current-RSS samples. They are useful for explaining macOS / MLX residency
accounting, but they are not the headline memory number for the README chart.

| Measurement | ARLE | mlx-lm | Meaning |
|---|---:|---:|---|
| request current RSS, old sweep | 2.5-4.1 GiB | 7.23-7.24 GiB | sampled during streaming; can miss prior high-water pages |
| process RSS high-water, current README sweep | 11.44-15.47 GiB | 7.23-7.24 GiB | conservative process-level peak for the run |

## What The Low Number Meant

The old sweep sampled:

```python
psutil.Process(pid).memory_info().rss
```

during each streaming request. On Apple unified memory, that current RSS sample
does not mean the full model consumes only that amount of physical memory.

The model was still loaded and addressable:

- ARLE loaded Qwen3.6 safetensors through MLX mmap-backed arrays.
- `metal_serve` no longer pinned MLX pages by default with `set_wired_limit`.
- macOS can reclaim or re-account non-wired mmap / Metal-managed pages.
- c=1 decode keeps a small KV working set, and Qwen3.6 MoE touches sparse
  expert subsets per token.

So the low RSS result was a residency/accounting observation, not proof of a
2-4 GiB total model footprint.

## Current README Reading

The current chart uses process RSS high-water because it is harder to
misinterpret. It says:

- ARLE TTFT/TPOT are close to mlx-lm after the KV-boundary clear fix.
- ARLE process RSS high-water is **higher** than mlx-lm on this retest.
- The earlier low-RSS explanation remains an accounting note, not a benchmark
  headline.

## Why `--auto-wired-limit` Exists

`--auto-wired-limit` asks MLX to keep roughly the model bytes plus headroom
wired/resident. That spends memory to reduce pageout-driven latency tails. It
is useful for p99-sensitive dedicated serving, but it is not required for the
model to be loaded or for inference to work.

Default mode keeps pages non-wired so macOS can reclaim them. That is better for
mixed desktop use, but it can expose more latency variance under memory
pressure.

## More Solid Follow-up

A full memory proof should add `vmmap -summary`, MLX allocator / wired counters,
and a memory-pressure p50/p95/p99 A/B with and without `--auto-wired-limit`.
Until then, README memory claims should use the high-water chart.
