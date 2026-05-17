# `arle train pretrain` baseline — Qwen3.5-arch 40M from-scratch CUDA, RTX 4070 Ti SUPER

## Goal (type: baseline)

Establish the **token-throughput baseline** for scratch-pretraining a
Qwen3.5-family hybrid linear-attention model (`--preset small-25m`,
actually 40.26 M params with this tokenizer) on a single consumer CUDA
GPU (RTX 4070 Ti SUPER, sm_89, 16 GB), so subsequent optimizations
(device-resident AdamW step, device-side softmax/CE, Liger FusedLinearCE,
Muon, bf16, CUDA-graph training step) can be A/B'd against a fixed
reference number.

**Headline metric**: `tok_per_sec` reported by `arle train pretrain`
(post-grad-accum, so `tokens_per_step / wall_per_step`).

## Hypothesis

Going in we expected ~1–5 K tok/s. ARLE's `Backend` contract is
**host-authoritative gradients** by design (see
`crates/autograd/src/backend.rs:6` and the M5.3a DeviceHandle contract
in `crates/autograd/AGENTS.md`), so the AdamW step + several CPU-only
ops (`softmax`, `log_softmax`, `cross_entropy`, `gather`, `mean`,
`rope`, `embedding`) force a host readback **every step**. With a 248 070
vocab the cross-entropy path materializes `[B, S, V]` logits in fp32,
which at `batch=8 seq=512` is 4 GB — and the gradient is another 4 GB.
We expected the GPU to be heavily under-utilized.

## Command

```bash
# CUDA 13.2 + g++-14 + tilelang env
CUDA_HOME=/opt/cuda CARGO_TARGET_DIR=/tmp/arle-target-cuda \
NVCC_CCBIN=g++-14 CC=gcc-14 CXX=g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo build --release -p agent-infer --features cli,cuda --bin arle

# Bench (v2, fits in VRAM)
/tmp/arle-target-cuda/release/arle train pretrain \
  --backend cuda \
  --corpus /home/ckl/arle-data/pretrain/corpus.txt \
  --tokenizer /home/ckl/arle-data/models/Qwen3.5-0.8B/tokenizer.json \
  --preset small-25m --model-family qwen35 \
  --steps 200 --batch 2 --seq 512 --grad-accum-steps 16 \
  --lr 3e-4 --log-every 5 --save-every 200 \
  --out /home/ckl/arle-data/benches/pretrain-25m-cuda-baseline-v2/run
```

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER · 16.0 GB VRAM · sm_89 |
| CUDA / nvcc | 13.2 V13.2.78 |
| Host compiler | g++-14 (nvcc 13.2 + GCC 16.1.1 stdlib mismatch; pin via `NVCC_CCBIN=g++-14`) |
| Driver | 595.71.05 (CUDA 13.2 runtime) |
| OS / kernel | Linux 7.0.3-1-cachyos |
| CPU / RAM | AMD Ryzen 7 3700X 8C16T · 31.3 GB |
| Rust toolchain | 1.95.0-x86_64-unknown-linux-gnu |
| cudarc | 0.19.7 (bumped from 0.18.2 this run, commit `7d5e696`) |
| ARLE commit | `f0a5a23` (`docs(research): tiny-LLM speedrun prior-art map`) |
| Features | `cli,cuda` |
| Model | Qwen3.5-family `small-25m` preset → vocab=248070, hidden=160, layers=2, heads=5, kv_heads=5, head_dim=32, ffn=320, max_pos=512, tie_embed=true |
| Params | **40 255 328 (40.26 M)** — vocab embedding ≈ 39.7 M dominates |
| Tokenizer | `mlx-community`/Qwen team `Qwen/Qwen3.5-0.8B` (248 070 vocab, BPE, downloaded from ModelScope) |
| Corpus | 43.6 MB plain text (alpaca-gpt4-en 52 K rows + gsm8k-main 7 473 rows) → 9 750 488 tokens after BPE |
| Hyperparams | steps=200, batch=2, seq=512, grad_accum=16, **effective batch=32, tokens/step=16 384**, lr=3e-4 cosine, AdamW (host) |

## Results

**Step 1 measurement** (terminated after 1 logged step — see §Problems,
1 step is already conclusive at this throughput):

| Metric | Value |
|---|---|
| `tok_per_sec` | **78.6** |
| `ms_per_step` | **208 418 ms** (= 208.4 s/step) |
| `loss` | 12.437 (random init, expected) |
| `grad_norm` | 0.772 |
| projected 200-step wall | **11.6 h** |

**GPU sampler** (256 samples × 2 s = 512 s wall covering setup + step 1):

| Metric | Value |
|---|---|
| `peak memory.used` | 5 675 MiB |
| `avg  memory.used` | 4 423 MiB |
| `peak utilization.gpu` | 100 % (brief, ≤2 s) |
| `avg  utilization.gpu` | **12.43 %** |

Raw artefacts:
- `/home/ckl/arle-data/benches/pretrain-25m-cuda-baseline-v2/train.log`
- `/home/ckl/arle-data/benches/pretrain-25m-cuda-baseline-v2/gpu.csv` (256 rows of `mem_mib,util_pct,power_w`)

## Problems

1. **OOM at `batch=8 seq=512`** (first attempt): `cuda alloc_zeros failed`
   on step 0 / 1. Cause: cross-entropy materializes `[B, S, V]` logits in
   fp32 = `8 × 512 × 248 070 × 4 B` ≈ **4.06 GB**, plus its gradient (another
   4 GB), plus log-softmax intermediates → ~12 GB just for the CE path on
   top of weights/grads/AdamW (614 MiB). 4070 Ti SUPER's 16 GB doesn't have
   the headroom. **This is the Liger FusedLinearCE case study**
   (`docs/research/2026-05-17-tiny-llm-speedrun-prior-art.md` §3.1).
2. **GPU utilization 12 %** — pipeline is host-bound. Forward matmul on
   CUDA → readback to CPU for `softmax` / `cross_entropy` / `gather` →
   host AdamW step (`backend.rs` per-param upload + compute + download).
   This is the M5.3b op-coverage gap called out in
   [`crates/autograd/AGENTS.md`](../../crates/autograd/AGENTS.md) §DeviceHandle
   contract → "When an op forces a host readback".
3. **Vocab = 248 070** dominates the param count (39.7 M of 40.26 M).
   `tie_embed=true` saves the LM head, but the `B × S × V` logit
   materialization is what blocks batch growth, not the param count.
   Smaller-vocab tokenizers (e.g. a custom 32 K BPE trained on the corpus
   like nanochat does) would change the regime.

## Δ vs baseline

First run for this configuration; no prior. Future optimizations cite
this entry as the `78.6 tok/s` baseline. Headline number to beat in the
next wins entry: **`tok_per_sec ≥ 786`** (10×) for the same
`--preset small-25m --model-family qwen35 --seq 512 --grad-accum=16`
shape on the same hardware. The 11.6 h projected wall is **derived**,
not the headline.

## Learnings

1. **Token throughput, not wall-clock, is the headline.** Wall is a
   derived figure (`steps × tokens_per_step / tok_per_sec`). Optimizing
   "10 h → 1 h" is just shorthand for "tok/s ≥ 10×".
2. **Pipeline is firmly host-bound at this stack version.** GPU util
   averages 12 %; peak only fires for matmul windows. Optimizations that
   *don't* move ops onto the device are wasted motion. License-or-kill
   gate for every next optimization: does it raise the avg GPU util?
3. **`small-25m` preset is vocab-embedding-dominated** with the
   Qwen3.5-0.8B tokenizer (248 K). For scaling-law work that wants
   transformer-FLOPs to dominate (not embedding lookups), we need either
   a 32 K BPE retrained on the corpus, OR a much wider/deeper transformer
   that doesn't have vocab-eclipse on params.
4. **The 248 K vocab × fp32 CE logits is a hard memory wall** that
   blocks batch growth on 16 GB. FusedLinearCE is therefore not just a
   throughput win — it unblocks the only realistic path to fitting
   `batch ≥ 8` for this shape.

## Optimization roadmap (cited from this baseline)

Ranked by expected `tok/s` gain. Each row lands as its own wins entry
citing this `78.6 tok/s` baseline + a Δ% row.

| Step | Lever | Expected tok/s gain | Cumulative tok/s |
|---|---|---|---|
| 1 | `Backend::optim_adamw_step` + device-resident grads (kill AdamW PCIe roundtrip per param per step) | 3–5× | 250–400 |
| 2 | Device-side `softmax` / `log_softmax` / `cross_entropy` / `gather` (kill `[B,S,V]` readback) | 2–3× | 500–1 200 |
| 3 | FusedLinearCE (Liger port, skip `[B,S,V]` materialization, unlock `batch=8–16`) | 1.5× + 4× memory | 750–1 800 |
| 4 | bf16 activations + grads (fp32 master weights kept) | 1.5–2× | 1 100–3 600 |
| 5 | Muon optimizer for hidden matrices (NewtonSchulz, AdamW kept on embeddings/norms) | 1.4× wallclock + ~20 % fewer tokens at same loss | 1 500–5 000 |
| 6 | CUDA-graph capture for the training step (needs packed-seq fixed shapes) | 1.2–1.5× | 1 800–7 500 |

**Stop-rule**: when `tok_per_sec ≥ 10× baseline` (≥ 786) AND average
GPU util ≥ 60 %, the host-bound regime is closed. Past that, further
gains require kernel-level work (FlashAttention-style fused
attention-bwd, larger effective batch, etc.).

## Rule

For tiny-LLM (≤ 100 M) CUDA training benches on consumer GPUs with
host-authoritative gradients: **always measure `tok/s` first, GPU util
second**. If `util < 30 %` the dominant cost is PCIe roundtrip, not
compute — kernel optimization at that point is wasted motion. Move ops
onto the device or rewrite the training step before tuning kernels.
