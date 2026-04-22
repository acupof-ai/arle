# Qwen3.6 DFlash Verify QMM Kernel

## Goal

- Optimization / alignment: replace the fake `prefer_verify_m16` fast-path
  with a real fixed-`M=16` verify-qmm Metal kernel, shared by compiled
  Qwen3.5/Qwen3.6 linears and the Qwen3.6 MoE block.

## Hypothesis

- If Metal Qwen3.6 verify uses one real `M=16` quantized-matmul kernel instead
  of reshaping to `[16, H]` and falling straight back to stock
  `quantized_matmul`, single-row DFlash should spend less time inside verify,
  acceptance should stop paying as much fixed linear overhead, and long
  generation throughput should recover.

## Command

```bash
cargo +stable build -p infer --release --no-default-features --features metal,no-cuda --bin metal_request --bin metal_bench
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
  --prompt ' benchmark throughput benchmark throughput benchmark throughput benchmark throughput benchmark throughput benchmark throughput benchmark throughput benchmark throughput benchmark throughput benchmark throughput' \
  --raw-prompt \
  --max-new-tokens 256 \
  --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

System trace:

```bash
sample <metal_request_pid> 2 5 -file /tmp/qwen36_verifyqmm_success.sample.txt
```

## Environment

- **Backend:** Metal
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Draft:** `z-lab/Qwen3.6-35B-A3B-DFlash`
- **Hardware:** `Apple M4 Pro`
- **Memory:** `48 GB unified memory`
- **OS:** `macOS 26.3.1 (25D771280a)`
- **Feature set:** `--no-default-features --features metal,no-cuda`
- **Execution mode:** strict serial for the `1024` benchmark (`baseline -> DFlash`)

## Results

Validation:

- `metal_bench` unit test: `1 passed`
- `clippy -D warnings` on `metal_bench` + `metal_request`: passed
- `metal_request` smoke: `exit 0`, `Output tokens: 2`

Strict-serial `1024` benchmark after the change:

| mode | gen tok/s | repo e2e tok/s | TTFT ms |
|---|---:|---:|---:|
| baseline | 83.49 | 83.04 | 66.43 |
| DFlash now | 35.90 | 35.80 | 76.95 |

Additional DFlash stats:

| block_size | avg accepted inputs | acceptance rate | blocks |
|---:|---:|---:|---:|
| 16 | 2.57 | 61.13% | 398 |

Same-harness before/after for the DFlash lane:

| snapshot | gen tok/s | repo e2e tok/s | TTFT ms | avg accepted inputs | acceptance rate |
|---|---:|---:|---:|---:|---:|
| before | 32.52 | 32.44 | 75.12 | 2.43 | 58.89% |
| now | 35.90 | 35.80 | 76.95 | 2.57 | 61.13% |

Direct synthetic-profile anchor after the change:

| metric | value |
|---|---:|
| `draft μ` | `0.1 ms` |
| `verify μ` | `72.1 ms` |
| `eval μ` | `0.6 ms` |
| `matched K̄` | `3.24 / 15` |
| `tok/block` | `4.24` |
| `p1` | `100%` |
| `p2` | `100%` |
| `p3` | `100%` |
| `p4` | `16%` |
| `Gen TPS` | `57.5 tok/s` |
| `TTFT` | `66.5 ms` |

System `sample` on the successful synthetic run:

- hot stack still anchored at
  `qwen35_dflash_speculative_block -> verify_block_summary -> qwen35_compiled_verify_block_summary`
- the new `verify_quantized_matmul_cpp(...)` is now directly visible in the
  verify stack
- `MTLCommandQueue` submission is still present, but the previous failed
  pipelined kernel experiment was removed; the committed tree keeps only the
  stable single-kernel `mma2big` path

## Problems

- This slice improves verify fixed cost, but DFlash is still below the plain
  baseline on the `1024` harness.
- A pipelined verify-qmm variant was prototyped and traced, but it stalled the
  request path on this machine; that branch was deleted instead of kept behind
  a default-off switch.
- The main remaining gap is still acceptance depth. Even after this change the
  average block only commits `2.57` tokens.

## Learnings

- `prefer_verify_m16` only matters if it actually lands on a different kernel.
  A reshape-to-2D-only path looks clean in code but leaves most of the verify
  cost untouched.
- The clean version is one bridge helper, `verify_quantized_matmul_cpp(...)`,
  shared by compiled Qwen3.5/Qwen3.6 linears and the MoE helper. That keeps
  the verify path flat and avoids per-caller special casing.
- Deletion was the right cleanup here: the failed pipelined variant complicated
  the bridge, produced a worse runtime story, and added no shippable value.

## Δ vs baseline

- **Prior committed local snapshot:** [2026-04-22-bench-guidellm-qwen36-dflash-prefill-seed-and-verify-fastpath.md](./2026-04-22-bench-guidellm-qwen36-dflash-prefill-seed-and-verify-fastpath.md)

| metric | prior committed local DFlash | now | Δ% |
|---|---:|---:|---:|
| generation_tps | 32.52 | 35.90 | +10.4% |
| repo_e2e_tps | 32.44 | 35.80 | +10.4% |
| TTFT ms | 75.12 | 76.95 | +2.4% |

## Rule

- For Qwen3.6/Qwen3.5 Metal DFlash verify, keep one canonical `M=16`
  verify-qmm kernel path and delete failed speculative kernel variants instead
  of parking them behind dormant flags.
