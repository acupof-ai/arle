# Qwen3.6 DFlash Prefill Seed And Verify Fastpath

## Goal

- Optimization / alignment: close two concrete gaps against `dflash-mlx` on
  Metal Qwen3.6 DFlash:
  full-prefix `target_hidden` seeding during prefill, and mask-free verify
  fast-path dispatch for exact 2-pass SDPA.

## Hypothesis

- If Qwen3.6 DFlash seeds the first draft block from the full captured prefill
  context instead of the terminal token/chunk, and verify stops forcing an
  additive mask when there is no left padding, local Metal DFlash should spend
  less time in verify and recover some throughput even if acceptance stays
  workload-limited.

## Command

```bash
cargo +stable fmt --all
```

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact
```

```bash
cargo +stable clippy -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench --bin metal_request -- -D warnings
```

```bash
target/release/metal_request \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt hi \
  --raw-prompt \
  --max-new-tokens 2 \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

```bash
target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --use-step-driver \
  --ignore-eos \
  --prompt-tokens 20 \
  --generation-tokens 1024 \
  --warmup 1 \
  --runs 3 \
  --json
```

```bash
target/release/metal_bench \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --use-step-driver \
  --ignore-eos \
  --prompt-tokens 20 \
  --generation-tokens 1024 \
  --warmup 1 \
  --runs 3 \
  --json
```

```bash
QWEN35_DFLASH_PROFILE=1 \
target/release/metal_request \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompt 'The function f satisfies the functional equation f(x) + f(y) = f(x + y) - xy - 1 for all real numbers x and y. If f(1) = 1, then find all integers n such that f(n) = n. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.' \
  --raw-prompt \
  --max-new-tokens 128 \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

```bash
PYTHONPATH=/tmp/dflash-mlx \
python3 -m benchmark.benchmark \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --draft z-lab/Qwen3.6-35B-A3B-DFlash \
  --prompt 'The function f satisfies the functional equation f(x) + f(y) = f(x + y) - xy - 1 for all real numbers x and y. If f(1) = 1, then find all integers n such that f(n) = n. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.' \
  --max-tokens 128 \
  --repeat 1 \
  --cooldown 0
```

## Environment

- **Backend:** Metal
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Draft:** `z-lab/Qwen3.6-35B-A3B-DFlash`
- **Hardware:** `Apple M4 Pro`
- **Memory:** `48 GB unified memory`
- **OS:** `macOS 26.3.1 (25D771280a)`
- **Feature set:** `--no-default-features --features metal,no-cuda`
- **Commit base while measuring:** `8d1880c`
- **Execution mode:** strict serial for local Metal benches (`baseline -> DFlash`)

## Results

Validation:

- `metal_bench` unit test: `1 passed`
- `clippy -D warnings` on `metal_bench` + `metal_request`: passed
- `metal_request` smoke: `exit 0`, `Output tokens: 2`

Strict-serial local 1024-token benchmark after the change:

| mode | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 69.82 | 69.49 | 67.51 |
| DFlash now | 32.52 | 32.44 | 75.12 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate | blocks |
|---:|---:|---:|---:|
| 16 | 2.43 | 58.89% | 421 |

Same-session before/after for the DFlash lane:

| snapshot | gen tok/s | repo e2e tok/s | TTFT ms | avg accepted inputs | acceptance rate |
|---|---:|---:|---:|---:|---:|
| before | 30.09 | 30.00 | 95.24 | 2.44 | 58.98% |
| now | 32.52 | 32.44 | 75.12 | 2.43 | 58.89% |

Direct-path profile anchor on a longer reasoning prompt:

| metric | value |
|---|---:|
| `draft μ` | `0.1 ms` |
| `verify μ` | `103.3 ms` |
| `eval μ` | `0.7 ms` |
| `matched K̄` | `1.70 / 15` |
| `tok/block` | `2.70` |
| `p1` | `66%` |
| `p2` | `46%` |
| `p3` | `32%` |
| `Gen TPS` | `35.8 tok/s` |
| `TTFT` | `183.5 ms` |

`dflash-mlx` comparison on the same machine, same target/draft pair, reasoning prompt, `max_tokens=128`, `repeat=1`:

| runtime | baseline tok/s | DFlash tok/s | acceptance |
|---|---:|---:|---:|
| `dflash-mlx` | 75.15 | 44.20 | 89.06% |

## Problems

- The local `metal_bench` synthetic prompt (`" benchmark throughput"` repeated
  to 20 tokens) still lands in a low-acceptance regime for Qwen3.6 DFlash.
- This slice improved verify-side cost, but did not move acceptance depth on
  the synthetic long-generation harness.
- `dflash-mlx` on this M4 Pro still did not beat baseline on the short
  128-token reasoning benchmark, so the README M5 Max speedups are not
  directly portable to this machine/workload.
- `dflash-benchmark` expects to run under a git repo; when launched from a
  plain temp directory it fails at `git rev-parse --short HEAD`.

## Learnings

- Qwen3.6/Qwen3.5 DFlash prefill must accumulate captured hidden state across
  the whole prompt, not just the terminal token or terminal chunk.
- Packed verify should not build an additive mask when every row has zero left
  padding; that mask blocks the exact 2-pass verify kernel for no benefit.
- Allowing the native single-row `verify_block_summary` path to use the same
  mask-free 2-pass kernel recovers verify throughput without bringing back the
  old packed `B=1` plumbing.
- On this machine, verify-side cost reductions are measurable, but acceptance
  still dominates the remaining gap versus baseline.

## Δ vs baseline

- **Prior committed local snapshot:** [2026-04-22-bench-guidellm-qwen36-dflash-suppress-mask-token.md](./2026-04-22-bench-guidellm-qwen36-dflash-suppress-mask-token.md)
- **Same-session before snapshot:** local pre-change DFlash run from the same
  benchmark window, shown in the results table above

| metric | prior committed local DFlash | now | Δ% |
|---|---:|---:|---:|
| generation_tps | 33.33 | 32.52 | -2.4% |
| repo_e2e_tps | 33.25 | 32.44 | -2.4% |
| TTFT ms | 75.66 | 75.12 | -0.7% |

## Rule

- For Qwen3.6/Qwen3.5 Metal DFlash, treat prefill hidden capture as a
  full-context accumulator and treat zero-left-padding packed verify as
  mask-free. Do not regress either path back to terminal-only seed capture or
  unconditional additive verify masks.
