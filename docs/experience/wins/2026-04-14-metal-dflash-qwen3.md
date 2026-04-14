# 2026-04-14 · Qwen3 Metal DFlash benchmark on M4 Pro

## Context

User asked for a real Metal DFlash runtime validation and a benchmark that
shows a meaningful throughput win, not just a compile-only proof.

Environment:

- Machine: Apple M4 Pro
- OS: macOS 25.3.0 (`Darwin 25.3.0`)
- Build: `cargo run --release -p infer --no-default-features --features metal,no-cuda`
- Target model: `mlx-community/Qwen3-4B-bf16`
- Draft model: `z-lab/Qwen3-4B-DFlash-b16`

## What Worked

Two things mattered:

1. The Metal DFlash path ran end-to-end against the real HuggingFace target
   and draft checkpoints with no correctness/runtime crash.
2. The draft model's native `block_size=16` was the throughput sweet spot.
   Forcing `--speculative-tokens 8` hid the gain and, on common workloads,
   regressed generation speed badly.

The useful benchmark is generation-heavy (`prompt=20`, `generation=256`):

| Variant | Prompt TPS | Gen TPS | Repo E2E TPS | TTFT |
| --- | ---: | ---: | ---: | ---: |
| Baseline target-only | 249.3 | 25.9 | 25.7 | 80.2 ms |
| DFlash, default block (`16`) | 236.3 | 152.0 | 144.8 | 84.6 ms |
| Delta | -5.2% | +487.0% | +463.4% | +5.5% |

That is the first locally verified Metal DFlash win: the decode-heavy /
generation-heavy path is substantially faster on real weights, while TTFT
stays roughly flat.

One caution from the same session: a more balanced `128/128` workload still
regressed (`27.4 -> 20.2 tok/s` generation), so the current implementation is
not yet a universal win. The confirmed local rule is therefore "use DFlash for
generation-heavy workloads first, and leave `speculative_tokens` at the draft
default unless a benchmark says otherwise."

## Raw Data

Before (`prompt=20`, `generation=256`, baseline):

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":25.900872172327126,"p50":25.811637792808927,"p99":26.210284634052705},"load_ms":1044.857333,"model":"mlx-community/Qwen3-4B-bf16","peak_rss_mb":7821.546875,"prompt_tokens":20,"prompt_tokens_requested":20,"prompt_tps":{"mean":249.3312970185276,"p50":250.63023164762322,"p99":252.45432943396585},"quantization":"bf16","repo_e2e_tps":{"mean":25.692312730786536,"p50":25.60084557544868,"p99":25.99787890394507},"timed_runs":3,"total_time_ms":{"mean":9964.809014,"p50":9999.669708000001,"p99":10047.800167},"ttft_ms":{"mean":80.22798599999999,"p50":79.798833,"p99":81.662875},"warmup_runs":1}
```

After (`prompt=20`, `generation=256`, DFlash at draft-default `block_size=16`):

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":152.03747317812528,"p50":153.75458485961784,"p99":154.28476205944847},"load_ms":4158.467709,"model":"mlx-community/Qwen3-4B-bf16","peak_rss_mb":8849.5625,"prompt_tokens":20,"prompt_tokens_requested":20,"prompt_tps":{"mean":236.29451728643303,"p50":236.26535169562445,"p99":237.24850946251445},"quantization":"bf16","repo_e2e_tps":{"mean":144.75902925380805,"p50":146.345022320337,"p99":146.7957175037615},"timed_runs":3,"total_time_ms":{"mean":1769.019903,"p50":1749.290792,"p99":1813.848833},"ttft_ms":{"mean":84.641028,"p50":84.650584,"p99":84.972708},"warmup_runs":1}
```

Mis-tuned control (`prompt=20`, `generation=256`, DFlash forced to `block_size=8`):

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":17.352594256980904,"p50":17.304685452073812,"p99":17.537820106793056},"load_ms":3768.5530830000002,"model":"mlx-community/Qwen3-4B-bf16","peak_rss_mb":8860.84375,"prompt_tokens":20,"prompt_tokens_requested":20,"prompt_tps":{"mean":237.36916051229957,"p50":239.56124453887205,"p99":244.10697685568047},"quantization":"bf16","repo_e2e_tps":{"mean":17.25399191525372,"p50":17.202876942962625,"p99":17.439931931128175},"timed_runs":3,"total_time_ms":{"mean":14838.062319666666,"p50":14881.231834,"p99":14953.99875},"ttft_ms":{"mean":84.322625,"p50":83.485958,"p99":87.55062500000001},"warmup_runs":1}
```

## Rule

For Metal DFlash, never lock `speculative_tokens` to an arbitrary lower value
before benchmarking. The draft model's native block size is part of the
performance contract; overriding it downward can erase the win completely.
