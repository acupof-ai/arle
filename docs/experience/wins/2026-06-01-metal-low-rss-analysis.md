# Metal RSS Accounting Note

## TL;DR

The README chart now uses **cumulative process RSS high-water** from the same
n=5 ARLE-vs-mlx-lm run. On 2026-06-01 that is:

| Metric | ARLE | mlx-lm | Meaning |
|---|---:|---:|---|
| README RSS high-water | 14.54 -> 17.41 GiB | 14.79 -> 14.81 GiB | conservative process RSS peak by target prompt length |
| raw per-request RSS peak at ARLE 12k | 2.31-15.04 GiB | 14.72-14.77 GiB | current residency can fall after macOS reclaim |

The low ARLE samples are real process RSS samples, but they are **not** proof
that the full Qwen3.6 model footprint is 2-3 GiB.

## Why RSS Can Look Small

ARLE loads Qwen3.6 through MLX mmap-backed arrays. With no explicit wired limit,
macOS can reclaim or re-account non-wired mmap / Metal-managed pages after the
weights have been faulted and used. That changes current process RSS without
unloading the model or making inference impossible.

This is why the same n=5 run can show ARLE request-window RSS falling at 12k
while the server still completes 256-token generations. Current RSS is
residency accounting, not model-size accounting.

## Why The README Uses High-Water

High-water is the safer README metric because it answers: "how much process RSS
did this run touch at least once?" It does not let later reclaim turn a memory
panel into a fake 12k memory reduction.

The raw JSON still keeps per-request RSS peaks for follow-up memory diagnostics:
`docs/experience/wins/assets/2026-06-01-readme-metal-vs-mlxlm-chat-essay-avg.json`.

## Why `--auto-wired-limit` Exists

`--auto-wired-limit` asks MLX to keep roughly the model bytes plus headroom
wired/resident. That spends memory to reduce pageout-driven p99 latency tails.
It is useful for dedicated serving, but it is not required for loading weights
or running inference.

Default mode leaves pages non-wired so the OS can reclaim them. That is better
for mixed desktop use, but it makes current RSS and latency more sensitive to
memory pressure.

## Follow-up

A full memory proof should add `vmmap -summary`, MLX allocator counters, and a
memory-pressure A/B with and without `--auto-wired-limit`. Until then, README
claims should stay on process RSS high-water plus raw RSS samples.
