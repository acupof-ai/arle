# Training speed benchmark — `train_multi_turn` CPU vs Metal

## Goal

Establish a repeatable baseline for the stepwise GRPO trainer's throughput on
CPU and Metal, and (M4.8 smoke) confirm that reward trajectories climb under
a short training run. Future acceleration work (e.g. device-resident tensors,
GPU-side softmax / RMSNorm) has numbers to beat.

## Hypothesis

- At **TinyLM default scale** (d_model=64, seq_len=10), Metal will be slower
  than CPU: every matmul round-trips host↔GPU, and matmul kernel time is in
  the noise relative to FFI setup.
- At **4×-scaled config** (d_model=256, d_ff=512, 4 layers, seq_len=18),
  Metal should pull ahead — matmul work per call becomes large enough to
  amortize transfer overhead.

Both hypotheses confirmed below.

## Params

Seed 42, same args both backends:

| Config | iters | group | prompt_len | agent | turns | obs | d_model | n_layers | d_head | n_heads | d_ff |
|--------|------:|------:|-----------:|------:|------:|----:|--------:|---------:|-------:|--------:|-----:|
| default | 50 | 8 | 4 | 2 | 2 | 2 | 64 | 2 | 32 | 2 | 128 |
| scaled  | 20 | 4 | 8 | 4 | 2 | 2 | 256 | 4 | 64 | 4 | 512 |

## Env

- Host: Apple Silicon Mac (Darwin 25.3.0), MLX via `crates/mlx-sys`
- Build: `cargo build --release` (CPU) / `--features metal` (Metal)
- Commit: `5ad7a8c`-era train crate (post M5 backend trait + M4.5/M4.6/#12)
- `TensorStore::with_backend(Arc<dyn Backend>)` dispatch; forward matmul
  goes through `backend().matmul_forward(...)`. Backward stays CPU.

## Results

### Default (TinyLM-scale) — 50 iters × group 8, seq_len 10

| Backend | Wall | iter/s | episode/s | token/s | Final KL | Mean reward (last 5 iters) | Eval pass@1 @ 49 |
|---------|-----:|-------:|----------:|--------:|---------:|---------------------------:|-----------------:|
| CPU     | 4.80 s | 10.41 | 83.30 | 833.0 | 0.0102 | 0.044 | 0.0000 |
| Metal   | 9.32 s |  5.36 | 42.91 | 429.1 | 0.0102 | 0.044 | 0.0000 |

CPU wins 1.9× here — the TinyLM forward has O(10²) matmul work per call.
FFI + MLX graph construction overhead dominates.

### Scaled (d_model=256, 4 layers) — 20 iters × group 4, seq_len 18

| Backend | Wall | iter/s | episode/s | token/s |
|---------|-----:|-------:|----------:|--------:|
| CPU     | 57.00 s | 0.35 | 1.40 | 25.3 |
| Metal   | 40.69 s | 0.49 | 1.97 | 35.4 |

Metal wins 1.40× at this scale. Reward trajectory bit-identical (same sampling
seeds, Metal/CPU matmul parity ≤1e-3). Loss path and KL unchanged.

## Self-evolve smoke (M4.8)

Same default config — 50 iters, group 8, eval every 25:

- `mean_reward` trajectory drifts from 0.031 (iter 0) to 0.044 (mean of last 5)
  with isolated peaks at 0.094. Monotonic climb is **not** expected at this
  scale — the policy is tiny, GRPO variance is high, and a reward-per-token
  signal makes exact-match pass@1 a stretch. The trajectory nudging up at
  all at a 50-iter budget is the smoke-test positive signal.
- Eval `pass@1` stays 0.0000 throughout — exact-sequence match requires
  every response token to land; the trainer optimizes a **fraction-matched**
  reward, not strict pass/fail. The 24h full self-evolve benchmark
  (Plan §6.3 acceptance P24 ≥ P0 × 1.3, absolute ≥ 0.3) is deferred until
  we have the Arithmetic/ToolSuccess verifiers driving harder tasks.

## Problems

- **Metal TinyLM slowdown** was expected. Device-resident tensors + a
  batched `mx::matmul` path would amortize the upload, but that's the full
  Option-A→B tensor rewrite in the plan and is not on the critical path.
  [MLX lazy-eval docs](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
  recommend tens-to-thousands of ops per `eval()` — our per-call `matmul→eval
  →readback` constructs 1-op graphs, hitting MLX's degenerate path. Design
  direction captured in
  [`docs/plans/rust-agent-rl-single-node.md`](../../plans/rust-agent-rl-single-node.md)
  §7.2 M5.3 annotation (commit `8080273`).
- **Greedy eval pass@1 = 0** is a reward-shape artifact, not a regression.
  Documented here so future readers don't mistake it for a training bug.

## Learnings

- Transfer overhead crossover for Metal matmul sits **somewhere between
  d_model=64 and d_model=256** for this workload. Training payloads below
  the crossover should stay on CPU.
- CPU/Metal forward parity verified empirically: identical mean_reward and
  KL trajectories across the 50-iter default run (same sampling seeds).
- Training speed is now **reported every run** — the bench line
  `bench: wall … | iter/s … | episode/s … | token/s …` lives in the
  trainer. Future optimizations diff against this baseline directly.

## Rule

Before claiming any training-speed improvement, re-run both configs from
this doc with the same seeds and compare the `bench:` line. Don't take
"Metal should be faster" on faith at TinyLM scale — it isn't.
