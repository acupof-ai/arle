# CPU matmul naive baseline at Qwen3-0.6B production shapes — AMD Ryzen 7 3700X, 2026-05-19

## Goal

- **(diagnosis)** Establish the GFLOPs/s ceiling of `autograd::backend::cpu_matmul_forward` (the naive triple-nested-loop scalar reference) at **every distinct matmul shape an OPD forward pass touches on Qwen3-0.6B** (seq=4 = 3 prompt + 1 rollout, hidden=1024, intermediate=3072, num_heads=16, head_dim=128, vocab=151936, num_hidden_layers=28).
- Provide an independent wall-clock framing complementary to the tiny-shape phase profile in [`2026-05-19-bench-opd-step-cpu-tiny-profile.md`](2026-05-19-bench-opd-step-cpu-tiny-profile.md): production-shape matmul **is the wall-clock ground truth** for OPD CPU substrate, not the tiny smoke window.

## Hypothesis

- Naive `cpu_matmul_forward` is sub-2 GFLOPs/s on Zen 2 (no SIMD, no parallel, column-strided inner loop on B, no cache blocking). At Qwen3-0.6B shapes this should place a **production OPD CPU step in the tens-of-seconds range**, while tiny-shape phase profiling completes ~1000 steps/sec — a >10,000× wall-clock gap silently hidden by per-step framing.
- If confirmed, a drop-in cached / SIMD / parallel matmul kernel (matrixmultiply, gemm, hand-rolled cache-blocked + std::thread) is the dominant ROI lever for OPD CPU.

## Command

```bash
mkdir -p bench-output/2026-05-19-cpu-matmul-naive-qwen3-06b && \
  cargo run -p autograd --example cpu_matmul_microbench --release \
  | tee bench-output/2026-05-19-cpu-matmul-naive-qwen3-06b/cpu_matmul_naive.txt
```

Benchmark constants (in `crates/autograd/examples/cpu_matmul_microbench.rs`):

```text
backend = cpu_matmul_forward (naive triple-loop scalar)
warmup_runs = 1
measured_runs = 5
seed_a = 0xA110_C5
seed_b = 0xB770_C5
```

Shape catalog (per layer × `per_forward_count`):

| name | M | K | N | per_forward_count |
|---|---:|---:|---:|---:|
| `q_proj` | 4 | 1024 | 2048 | 28 |
| `k_proj` | 4 | 1024 | 1024 | 28 |
| `v_proj` | 4 | 1024 | 1024 | 28 |
| `o_proj` | 4 | 2048 | 1024 | 28 |
| `gate_proj` | 4 | 1024 | 3072 | 28 |
| `up_proj` | 4 | 1024 | 3072 | 28 |
| `down_proj` | 4 | 3072 | 1024 | 28 |
| `lm_head` | 4 | 1024 | 151936 | 1 |

## Environment

| Item | Value |
|---|---|
| Backend | CPU `cpu_matmul_forward` (naive triple-loop) |
| CPU | AMD Ryzen 7 3700X 8-Core Processor, 8C/16T, Zen 2 |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Runtime substrate commit | `16430a7` (matmul code unchanged since `4da07c9`) |
| Feature set | `cargo run -p autograd --example cpu_matmul_microbench --release` |
| Non-default flags | none |

## Results

### Per-shape median (5 runs, warmup=1)

| shape | FMAs | median (s) | GFLOPs/s | ×/fwd | fwd cost (s) |
|---|---:|---:|---:|---:|---:|
| `q_proj    [4,1024] @ [1024,2048]`   | 8.389e6 | 0.042127 | **0.398** | 28 | 1.179564 |
| `k_proj    [4,1024] @ [1024,1024]`   | 4.194e6 | 0.021061 | **0.398** | 28 | 0.589695 |
| `v_proj    [4,1024] @ [1024,1024]`   | 4.194e6 | 0.020998 | **0.399** | 28 | 0.587945 |
| `o_proj    [4,2048] @ [2048,1024]`   | 8.389e6 | 0.043031 | **0.390** | 28 | 1.204868 |
| `gate_proj [4,1024] @ [1024,3072]`   | 1.258e7 | 0.060108 | **0.419** | 28 | 1.683031 |
| `up_proj   [4,1024] @ [1024,3072]`   | 1.258e7 | 0.060163 | **0.418** | 28 | 1.684571 |
| `down_proj [4,3072] @ [3072,1024]`   | 1.258e7 | 0.073041 | **0.345** | 28 | 2.045154 |
| `lm_head   [4,1024] @ [1024,151936]` | 6.223e8 | 0.881362 | **1.412** |  1 | 0.881362 |

### Aggregate cost per OPD step

| Quantity | Value |
|---|---:|
| Matmul cost per single forward (28 layers + lm_head) | **9.856 s** |
| Matmul cost per OPD step (≈3 forwards: rollout student + teacher + tape-on student) | **29.569 s** |
| Backward adds ~2× forward matmul cost (transpose + 2 sgemm per matmul) — projected total step | **~60–90 s** |

### Wall-clock framing contrast (SOLID §0)

| Framing | Source | Result |
|---|---|---|
| Tiny shape (hidden=16, vocab=16) | [`2026-05-19-bench-opd-step-cpu-tiny-profile.md`](2026-05-19-bench-opd-step-cpu-tiny-profile.md) | **1173 steps/sec** (0.85 ms/step) |
| Production shape (Qwen3-0.6B, projected from naive matmul only) | this bench | **~0.011–0.017 steps/sec** (60–90 s/step) |

The tiny-shape framing under-represents production wall-clock by **~5 orders of magnitude**. Phase-percentage attribution at tiny shape is still a useful taxonomy, but **fix-priority must be set by production-shape wall-clock**, not tiny-step percentages.

## Problems

- **Single-CPU-machine numbers.** Bench runs on one workstation (Zen 2, 8C). Numbers on Zen 3/4, Intel Sapphire Rapids, ARM Graviton will differ. Treat absolute GFLOPs as a Zen 2 anchor, not a universal claim.
- **Naive impl has no AVX2.** rustc 1.95 with `-C opt-level=3` does emit some autovectorisation, but the column-strided B-access pattern (`b[(inner * n) + col]` at `crates/autograd/src/backend.rs:1489`) defeats most SIMD. The 0.4 GFLOPs result is consistent with that.
- **Backward not measured.** Backward matmul cost projected as 2× forward (two sgemm per backward, plus a host transpose). Will be confirmed by the next tranche.
- **No host allocator pressure isolated.** Each call allocates the output `Vec<f32>`. At the largest shape (`lm_head`, 2.4 MB) this is small relative to the matmul body, but at intermediate shapes it could be a noticeable share — not separated in this run.

## Learnings

1. **Wall-clock framing flipped the priority.** Tiny-shape phase % (codex B2 entry) said *backward 31% / rollout-student-forward 30% / teacher 18% / student 17%* — a reasonable taxonomy but a misleading optimisation queue. At production shape, every one of those phases is dominated by `cpu_matmul_forward`. Single-variable fix = matmul kernel swap.
2. **0.4 GFLOPs/s is sub-1% of Zen 2 single-core SIMD peak.** Even a single-threaded `matrixmultiply::sgemm` (which uses AVX2 + register/cache blocking) should reach 20–30 GFLOPs/s on Zen 2 — **50–75× per-call speedup**. Parallel across 8 cores: 150–200 GFLOPs/s, ~400× per-call.
3. **Why naive matmul ruins cache.** Inner loop `acc += a[row*k+inner] * b[inner*n+col]` strides B by `n` per inner step (column-major access on a row-major buffer). For `n=3072` (gate/up_proj), each FMA touches a new 12 KB stride — bypasses L1d (256 KB per L1) cache utility entirely. Reordering loops (`row → inner → col`) and tiling on `k` would alone give multiple-×.
4. **License-or-kill (SOLID §0):** the matmul-swap hypothesis now has wall-clock evidence at production shape, not just source-grep inference. Next experiment is a **single-variable** A/B: same `crates/autograd/examples/cpu_matmul_microbench.rs`, swap `cpu_matmul_forward` body for a candidate (matrixmultiply / hand-rolled blocked / rayon-parallel blocked) and compare the same table. Any candidate that fails to deliver ≥10× on at least 6 of the 8 shapes is killed.

## Rule

For any OPD/train CPU perf claim:
- Tiny-shape framings (hidden ≤ 64, vocab ≤ 1024) are valid for **correctness invariants and structural % taxonomy**, never for **wall-clock priority decisions**.
- Wall-clock priority requires a production-shape (Qwen3-0.6B or larger) component bench *or* a full-step probe at the production shape. The component bench in this entry is the cheaper substitute when the full-step probe is memory-blocked on the dev box.
- Drop-in single-op replacement experiments must be benched against the exact same shape catalog committed under `crates/autograd/examples/cpu_matmul_microbench.rs:QWEN3_06B_SHAPES`. The full-step probe is the eventual confirmation; the microbench is the SOLID licensing gate for whether the full-step run is worth its memory cost.

## Artefacts

- Bench source: `crates/autograd/examples/cpu_matmul_microbench.rs`
- Raw: `bench-output/2026-05-19-cpu-matmul-naive-qwen3-06b/cpu_matmul_naive.txt`
- Raw sha256: `6a229eb96bb39c24ea06bfaabf3be9d993ca8530fcc2d82d438bebe16dc135c7`
- Companion (tiny-shape phase profile under the same head): [`2026-05-19-bench-opd-step-cpu-tiny-profile.md`](2026-05-19-bench-opd-step-cpu-tiny-profile.md)
