# 2026-04-15 · Metal DFlash rerun and server reality check

## Context

Reran Metal DFlash on current `main` for two reasons:

1. verify the direct Qwen3 DFlash win still holds on the current Metal path
2. measure what DFlash actually does inside `metal_serve`, where DFlash still
   uses the legacy serial runtime instead of the live scheduler runtime

The same session also exposed and fixed a local-cache DX bug: DFlash draft
checkpoints in the HuggingFace snapshot cache commonly do not ship tokenizer
files and expose their weights as symlinks into the blob store. The old
`resolve_model_path()` logic therefore missed an already-cached local draft
checkpoint and fell back to network repo probing. The fix adds
`resolve_weighted_model_path()` plus a symlink-aware local weighted-checkpoint
probe.

Environment:

- Commit under test: post-`hf_hub` weighted-model local-cache fix on 2026-04-15
- Machine: Apple M4 Pro
- OS: macOS 26.3.1(a)
- Build:
  `cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve --bin metal_bench`
- Artifacts:
  - `/tmp/metal-dflash-2026-04-15/qwen3-baseline-20-256.json`
  - `/tmp/metal-dflash-2026-04-15/qwen3-dflash-20-256-localdraft.json`
  - `/tmp/metal-dflash-2026-04-15/qwen3-dflash-serve-focused-small.json`

## What Worked

### Direct DFlash is still a real decode-heavy win

`mlx-community/Qwen3-4B-bf16`, `prompt=20`, `generation=256`, `warmup=1`, `runs=3`

| Variant | Prompt TPS | Gen TPS | Repo E2E TPS | TTFT |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `237.2` | `24.5` | `24.3` | `84.4 ms` |
| DFlash | `237.2` | `154.4` | `146.9` | `84.3 ms` |
| Delta | `+0.0%` | `+529.1%` | `+503.5%` | `-0.1%` |

Interpretation:

- the original Metal DFlash story still holds on the current tree
- the win is decode-heavy and single-request-shaped
- TTFT stays flat; the gain is almost entirely decode throughput

### DFlash in `metal_serve` helps single-request decode, not concurrency

Focused server check on the same Qwen3 target and a `20 / 256`-style prompt:

| Variant | C | Throughput | TTFT p50 | ITL p50 |
| --- | ---: | ---: | ---: | ---: |
| Baseline `metal_serve` | `1` | `23.29 tok/s` | `1514 ms` | `36.98 ms` |
| DFlash `metal_serve` | `1` | `27.81 tok/s` | `107 ms` | `0.01 ms` |
| Baseline `metal_serve` | `4` | `23.30 tok/s` | `3559 ms` | `147.32 ms` |
| DFlash `metal_serve` | `4` | `24.45 tok/s` | `15767 ms` | `0.04 ms` |

Interpretation:

- at `C=1`, DFlash does help the server path on this decode-heavy workload
- at `C=4`, aggregate throughput is still nearly flat because the server path is
  still serial under DFlash
- DFlash does **not** fix serving concurrency; it only speeds up the decode work
  inside each serial request

The very small DFlash `ITL p50` is expected here. DFlash accepts token blocks,
so the HTTP stream emits accepted content in bursts; that makes inter-chunk time
look tiny even though end-to-end concurrency remains queue-shaped.

### The local-cache DX fix is real

After the `hf_hub` fix, this now resolves from the local HuggingFace cache
without probing the network:

```bash
./target/release/metal_bench \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt-tokens 20 \
  --generation-tokens 32 \
  --warmup 0 \
  --runs 1 \
  --json
```

That command completed locally and produced:

- `generation_tps = 151.0`
- `repo_e2e_tps = 88.6`
- `ttft = 149.3 ms`

## Raw Data

### Direct baseline

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":24.543886131173654,"p50":24.671403230751572,"p99":24.89424746209346},"load_ms":5036.867375,"model":"mlx-community/Qwen3-4B-bf16","peak_rss_mb":2484.921875,"prompt_tokens":20,"prompt_tokens_requested":20,"prompt_tps":{"mean":237.16060802379835,"p50":234.8059726045532,"p99":248.88920127249082},"quantization":"bf16","repo_e2e_tps":{"mean":24.34671446758244,"p50":24.481810639506996,"p99":24.683497521100172},"timed_runs":3,"total_time_ms":{"mean":10516.877236333334,"p50":10456.742917,"p99":10722.587084},"ttft_ms":{"mean":84.44506966666667,"p50":85.176709,"p99":87.801458},"warmup_runs":1}
```

### Direct DFlash

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":154.4094522997508,"p50":154.36870210744934,"p99":154.60062935246933},"load_ms":2430.131625,"model":"mlx-community/Qwen3-4B-bf16","peak_rss_mb":8512.96875,"prompt_tokens":20,"prompt_tokens_requested":20,"prompt_tps":{"mean":237.19790281381748,"p50":237.49255267884652,"p99":240.21402877792892},"quantization":"bf16","repo_e2e_tps":{"mean":146.9357040903301,"p50":146.90857162318264,"p99":147.00893524328185},"timed_runs":3,"total_time_ms":{"mean":1742.258861333333,"p50":1742.5804170000001,"p99":1742.805417},"ttft_ms":{"mean":84.32786133333333,"p50":84.213167,"p99":85.511333},"warmup_runs":1}
```

### Focused `metal_serve` baseline vs DFlash

```json
{
  "baseline": {
    "results": [
      {
        "concurrency": 1,
        "requests": 2,
        "throughput_tps": 23.28573982470752,
        "ttft_p50_ms": 1514.309287071228,
        "itl_p50_ms": 36.98110580444336
      },
      {
        "concurrency": 4,
        "requests": 4,
        "throughput_tps": 23.29675614961886,
        "ttft_p50_ms": 3558.847188949585,
        "itl_p50_ms": 147.3175287246704
      }
    ]
  },
  "dflash": {
    "results": [
      {
        "concurrency": 1,
        "requests": 2,
        "throughput_tps": 27.812215515941247,
        "ttft_p50_ms": 107.19513893127441,
        "itl_p50_ms": 0.014424324035644531
      },
      {
        "concurrency": 4,
        "requests": 4,
        "throughput_tps": 24.450207541716257,
        "ttft_p50_ms": 15767.078518867493,
        "itl_p50_ms": 0.040411949157714844
      }
    ]
  }
}
```

## Rule

For Metal DFlash, keep two truths separate:

- DFlash is already a strong decode-heavy optimization for a single request
- DFlash inside `metal_serve` is **not** a serving-throughput solution while it
  still runs on the legacy serial runtime

If server concurrency is the goal, the next work remains the same:
cross-request batched decode in the live Metal scheduler runtime.
