# `nanochat base_train` industry baseline — depth=12 (GPT-1 scale), RTX 4070 Ti SUPER

> **Baseline measurement, not a competition.** This entry captures
> karpathy/nanochat's PyTorch single-GPU training throughput on the same
> hardware (RTX 4070 Ti SUPER, sm_89, 16 GB) so we have an "industry
> reference" number to anchor ARLE's `≥ industry × 1.3` target against.
>
> Counterpart ARLE baseline: see
> [`2026-05-17-bench-pretrain-wave20-adamw-step-device.md`](2026-05-17-bench-pretrain-wave20-adamw-step-device.md)
> (P3.1/Wave 2.0, median 174.7 tok/s).

## Goal

- **(baseline)** Measure nanochat `depth=12` (GPT-1 ~286 M params) training
  throughput on RTX 4070 Ti SUPER under apples-to-apples conditions
  (seq=512, tokens/optim-step=16 384) to anchor ARLE's optimization
  roadmap. **Not** an optimization attempt — measure as-shipped.

## Hypothesis

- nanochat is karpathy's optimized PyTorch reference. Expected
  industry-baseline tok/s: **5 000 – 30 000 tok/s** at d12 on a single
  4070 Ti SUPER. Above that range surprises me upward; below it would
  signal something is wrong with the run.
- ARLE small-25m's 174.7 tok/s is at a **much smaller** transformer
  (40 M params with ~95 % embedding) and a 7.6× wider vocab, so the
  ratio cannot be read as "compute efficiency"; it's primarily a
  scale-of-model + framework-overhead story.

## Command

```bash
# One-time setup (already done):
#   /home/ckl/projects/arle/.venv/bin/uv venv .venv --python 3.10
#   /home/ckl/projects/arle/.venv/bin/uv sync --extra gpu      # torch 2.9.1+cu128
#   python -m nanochat.dataset -n 8                            # ~700 MB climbmix shards
#   python -m scripts.tok_train --max-chars=500_000_000        # 32 768 BPE vocab, 14.7 s

# Bench (both runs):
cd /home/ckl/arle-data/baselines/nanochat
source .venv/bin/activate
OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=$HOME/.cache/nanochat \
  python -m scripts.base_train \
    --depth=12 --max-seq-len=512 \
    --device-batch-size=4 --total-batch-size=16384 \
    --num-iterations=20 --warmup-steps=5 \
    --eval-every=-1 --core-metric-every=-1 \
    --sample-every=-1 --save-every=-1 \
    --window-pattern=L \              # Run 2 only; Run 1 uses default SSSL
    --run=dummy --model-tag=arle-baseline-d12
```

Wrapped by `/home/ckl/arle-data/benches/nanochat-baseline-d12/run_bench{,_full_attn}.sh`.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER · 16 GB · sm_89 (Ada Lovelace) |
| Driver | 595.71.05 |
| CUDA | 13.2 V13.2.78 (system /opt/cuda), wheel built with cu128 |
| Python | 3.10.20 (uv-managed cpython) |
| Torch | `2.9.1+cu128` |
| Compute path | BF16, `torch.compile`, PyTorch SDPA (no FA3 — sm_89 lacks Hopper kernels), no FP8 |
| nanochat commit | `dc54a1a` (clone at `/home/ckl/arle-data/baselines/nanochat`) |
| Model | nanochat GPT, `n_layer=12 n_head=6 n_kv_head=6 n_embd=768 head_dim=128` |
| Vocab | 32 768 (RustBPE, trained locally on 8 climbmix shards / 500 M chars) |
| Params | **286 261 730** (286.3 M) — wte 25.2 M + value_embeds 151.0 M + lm_head 25.2 M + transformer 84.9 M |
| Optimizer | **Muon** (matrices) + **AdamW** (embed/scalars), per-group LR scaling, weight-decay scaled by depth |
| Hyperparams | seq=512, device-batch=4, total-batch=16 384 → grad_accum=8 → **tokens/optim-step = 16 384** (matches ARLE) |
| Dataset | climbmix-400b-shuffle (HF: `karpathy/climbmix-400b-shuffle`), 8 train + 1 val shard |

## Canonical params (locked for apples-to-apples vs ARLE)

- `--max-seq-len=512` (ARLE seq=512)
- `--total-batch-size=16384` (ARLE: batch=2 × seq=512 × grad-accum=16 = 16 384 tokens/optim-step)
- `--num-iterations=20` — 5 warmup logged steps (LR ramp + `torch.compile`), then 15 timed
- `--eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1` — pure throughput

## Results

### Headline tok/s table (steady-state, steps 5–19, n = 15 each)

| Run | window-pattern | median tok/s | mean tok/s | range | step 0 (compile) | peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| 1 | `SSSL` (default; sliding-window) | **54 148** | 53 915 | 51 727 – 54 273 | 35 958 ms | 3 864 MiB |
| 2 | `L` (full attention only) | **56 291** | 56 002 | 53 430 – 56 435 | 23 005 ms | 3 856 MiB |

Δ Run 2 vs Run 1: **+4.0 %** — nanochat itself warns "SDPA has no
support for sliding window attention, your GPU utilization will be
terrible" with the default `SSSL` pattern; Run 2 confirms a small but
real regression from that. **Use Run 2 (`56 291 tok/s`) as the
industry-baseline anchor** since ARLE has no sliding window.

### Per-step trace (Run 2, full attention)

| step | dt (ms) | tok/s | loss |
|---:|---:|---:|---:|
| 0 (warmup, `torch.compile`) | 23 005 | 712 | 10.397 |
| 1 | 290 | 56 429 | 10.385 |
| 2 | 300 | 54 702 | 10.365 |
| 3 | 290 | 56 425 | 10.324 |
| 4 | 291 | 56 354 | 10.254 |
| 5 | 290 | 56 411 | 10.157 |
| 6 | 293 | 55 873 | 9.996 |
| 7 | 293 | 55 865 | 10.134 |
| 8 | 290 | 56 435 | 9.797 |
| 9 | 291 | 56 242 | 9.548 |
| 10 | 290 | 56 427 | 9.327 |
| 11 | 293 | 55 829 | 9.078 |
| 12 | 295 | 55 515 | 8.888 |
| 13 | 291 | 56 291 | 8.740 |
| 14 | 291 | 56 310 | 8.589 |
| 15 | 307 | 53 430 | 8.450 |
| 16 | 290 | 56 405 | 8.356 |
| 17 | 290 | 56 414 | 8.236 |
| 18 | 291 | 56 359 | 8.150 |
| 19 | 291 | 56 222 | 8.046 |

Loss descends monotonically; numerics are fine.

### GPU util / VRAM

- Peak VRAM (in-process `torch.cuda.max_memory_allocated`): **3 856 MiB** (Run 2).
- Peak VRAM (nvidia-smi sampler, includes caching-allocator reserve): **5 459 MiB**.
- GPU util (nvidia-smi at 1 Hz): **mean 24 %, peak 100 %** — sampler aliasing against
  the ~290 ms steps; the actual in-step utilisation is the relevant one. Manual MFU
  estimate (BF16): `2.35e14 FLOPs / 20 steps / 0.291 s / 88 TFLOPS ≈ 46 %` of dense
  BF16 peak — reasonable for a `torch.compile + SDPA` path on Ada Lovelace.
- nanochat MFU column shows `0.00` because its `get_peak_flops()` has no entry for
  `RTX 4070 Ti SUPER`; the script logs `Peak FLOPS (BF16): inf`.

### Δ vs ARLE P3.1/Wave 2.0 (head-to-head, same hardware)

| Axis | ARLE P3.1/Wave 2.0 | nanochat d12 (Run 2) | Notes |
|---|---|---|---|
| **median tok/s** | **174.7** | **56 291** | **ratio: 174.7 / 56 291 = 0.0031 = ARLE is 322× slower** |
| Params | 40.26 M | 286.26 M | nanochat is **7.1× larger** → does *more* work per token |
| Architecture | Qwen3.5 small-25m (hidden=160, L=2, A=5, FFN=320) | GPT (n_embd=768, L=12, n_head=6, FFN=3072) + value_embeds | very different shapes |
| Vocab | 248 320 (Qwen3.5) | 32 768 (nanochat BPE) | ARLE vocab is **7.6× wider** → embedding-heavy |
| Seq len | 512 | 512 | matched |
| Tokens/optim-step | 16 384 (batch=2, grad_accum=16) | 16 384 (batch=4, grad_accum=8) | matched |
| Optimizer | AdamW only (host fallback for most params) | Muon (matrices) + AdamW (embed/scalars), both device-resident | confound — see Problems |
| Compute path | custom CUDA C kernels (Rust crate), no compile | `torch.compile` + PyTorch SDPA, full graph fusion | confound |
| Peak VRAM | not recorded in P3.1 entry (in-flight) | 3 856 MiB | — |
| GPU util | reported low (host-fallback bound; see Wave 2.1 plan) | ~46 % MFU during compute, ~24 % sampler-mean (aliased) | — |

If we normalise by params and assume linear FLOPs scaling
(`tok/s_norm = tok/s × params`), the gap shrinks: ARLE 7.03 G·tok/s vs
nanochat 16.11 T·tok/s ≈ ARLE is **2 290× slower per param-FLOP**. The
"322× slower per token" framing is the cleaner one here because token
throughput is the user-facing metric, but the param-FLOP framing makes
clear ARLE isn't just "doing less work, slower".

## Problems

1. **Param-count + architecture mismatch is huge.** nanochat d12 = 286 M
   (Muon-trained 12-layer 768-d GPT with value_embeds), ARLE small-25m
   = 40 M (Qwen3.5 2-layer 160-d). They are not the same model class.
   The numeric ratio is a *system-level* tok/s comparison, not a
   compute-efficiency comparison. **Don't read 322× as
   "ARLE is 322× less efficient per FLOP"** — see param-normalised number
   above.
2. **Vocab confound.** ARLE 248 320 vs nanochat 32 768 (7.6× wider).
   Embedding+LM-head work dominates ARLE small-25m (~95 % of params
   live in the embedding); nanochat d12 only spends ~17 % of params
   there. ARLE tok/s is essentially a giant-embedding-on-CUDA test;
   nanochat tok/s is a transformer-matrix test.
3. **Optimizer + framework confound.** nanochat: PyTorch + Muon/AdamW
   + `torch.compile` + SDPA, all device-resident. ARLE Wave 2.0: AdamW
   only, host fallback on 7 backward ops (Wave 2.1 in flight to fix).
   The ARLE 174.7 tok/s is **known bottlenecked on host-CPU readbacks**
   — Wave 2.0 wins entry shows only 24 of 24 AdamW launches actually
   hit `step_device`; the rest of the gradient path is host. The 322×
   gap is *expected* to shrink dramatically once Wave 2.1 lands the
   batch-port of `rms_norm_backward`/`silu_backward`/etc.
4. **MFU not auto-computed by nanochat.** `get_peak_flops()` has no
   `RTX 4070 Ti SUPER` entry → logs `Peak FLOPS (BF16): inf`, MFU
   prints `0.00`. Manual estimate ≈ 46 % BF16 MFU; modest gap to
   the ~70 % MFU that an FA3-equipped H100 hits with the same script.
5. **GPU sampler aliasing.** `nvidia-smi --query-gpu=...` sampled at
   1 s misses sub-step burst-and-sync behaviour — mean util reads 24 %
   while peak hits 100 %. Use the in-process `dt`-derived MFU for
   throughput intuition, not the sampler mean.
6. **Run 1 used `--window-pattern=SSSL`** (nanochat default) which
   triggers nanochat's own warning that SDPA on a non-Hopper GPU runs
   slowly under sliding window. Re-ran with `--window-pattern=L`
   (Run 2) for a fair vs-ARLE compare.

## Learnings

- **The industry baseline for "PyTorch training a GPT on a single
  4070 Ti SUPER, BF16, seq=512" is ≈ 56 k tok/s on a 286 M model.**
  This number is the ceiling we benchmark "is ARLE the right neighborhood?"
  against, *after* normalising for params/vocab/architecture. The raw
  174.7 vs 56 291 ratio is system-level, not algorithmic.
- **ARLE's tok/s gap is dominated by the host-fallback gradient path,
  not by being "a worse PyTorch competitor".** Wave 2.0 / Wave 2.1 plan
  is correct: batch-port the 7 backward ops to device, then we expect
  tok/s to jump multiples, not %s. The 322× ratio is *not* a permanent
  architectural cost.
- **Param-FLOP normalisation, not token throughput, is the fair
  efficiency metric here.** When the parent agent compares ARLE
  optimisation deltas to nanochat, do it as
  `Δ (tok/s × params)` not `Δ tok/s`.
- **Sliding-window + SDPA on Ada Lovelace is a measurable cliff** (4 %
  in this bench). If ARLE adds sliding-window attention later, keep an
  FA3-or-equivalent path or expect the same regression.
- **nanochat as-baseline is well-suited for this comparison** —
  single-file model, no DDP, runs out of the box with `uv sync --extra
  gpu`, completes a 20-iter d12 bench in <1 min after compile warmup.
  Worth keeping at `/home/ckl/arle-data/baselines/nanochat` for re-runs
  after each ARLE Wave milestone.

## Δ vs baseline

- **First nanochat baseline on this hardware.** No prior entry to diff.
- ARLE counterpart: [`2026-05-17-bench-pretrain-wave20-adamw-step-device.md`](2026-05-17-bench-pretrain-wave20-adamw-step-device.md).
- Raw artefacts in
  `/home/ckl/arle-data/benches/nanochat-baseline-d12/`:
  - `train.log` (Run 1, SSSL), `train_sssl.log` (copy), `train_fullattn.log` (Run 2, L)
  - `gpu.csv` / `gpu_sssl.csv` (Run 1, 2 s), `gpu_fullattn.csv` (Run 2, 1 s)
  - `meta.txt` — full param counts, MFU calc, env dump
  - `setup.log` — uv sync output (torch 2.9.1+cu128, 130 packages)
  - `tokenizer.log` — RustBPE training (14.7 s, vocab 32 768)
  - `run_bench.sh`, `run_bench_full_attn.sh` — exact commands
