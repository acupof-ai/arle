<p align="center">
 <img src="docs/assets/caret-counter-lockup.svg" height="56" alt="arle">
</p>

<p align="center">
 <b>Pure-Rust LLM engine: serving, agents, and on-policy distillation — on Apple Silicon and NVIDIA. No Python on the hot path.</b>
</p>

<p align="center">
 <sub>35B MoE at <b>85 tok/s</b> on a MacBook · <b>bit-identical</b> speculative decode · OPD lifts 4B student <b>+27pp</b> on MATH-500</sub>
</p>

<p align="center">
 <a href="https://cklxx.github.io/arle/"><img src="https://img.shields.io/badge/website-cklxx.github.io%2Farle-D97757?style=flat-square" alt="Website"></a>
 <a href="https://github.com/cklxx/arle/actions/workflows/ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
 <a href="https://github.com/cklxx/arle/actions/workflows/metal-ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/metal-ci.yml/badge.svg" alt="Metal CI"></a>
 <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
 <a href="https://github.com/cklxx/arle/releases"><img src="https://img.shields.io/github/v/release/arle?include_prereleases" alt="Release"></a>
</p>

<p align="center">
 <a href="#quick-start">Quick Start</a> ·
 <a href="#performance">Performance</a> ·
 <a href="#why-arle">Why ARLE</a> ·
 <a href="docs/http-api.md">HTTP API</a> ·
 <a href="docs/support-matrix.md">Support Matrix</a> ·
 <a href="docs/architecture.md">Architecture</a> ·
 <a href="CHANGELOG.md">Changelog</a>
</p>

<p align="center">
 <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

---

## Quick Start

### Install

```bash
# Apple Silicon (Homebrew)
brew install cklxx/tap/arle

# Apple Silicon or Linux x86_64 (one-line installer)
curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh | sh

# Linux + NVIDIA (Docker, no compile needed)
docker run --rm --gpus all -p 8000:8000 -v $PWD/models:/models:ro \
 ghcr.io/cklxx/arle:latest serve --backend cuda --model-path /models/Qwen3.5-4B
```

### Serve

```bash
# NVIDIA CUDA
arle serve --backend cuda --model-path /path/to/Qwen3.5-4B --port 8000

# Apple Silicon (Metal)
arle serve --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit --port 8000

# DSpark block-drafter decode + in-process Markov-head training (CUDA)
arle serve --backend cuda \
 --model-path /path/to/Qwen3.6-27B \
 --spec-type dspark --mtp-draft-model /path/to/dspark-draft \
 --dspark-train --port 8000
```

### Use

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
 model="default",
 messages=[{"role": "user", "content": "Hello from ARLE"}],
).choices[0].message.content)
```

> **Source builds need a backend.** `cargo build --release` alone produces a CLI-only binary.
> Add `--features cuda` (NVIDIA) or `--no-default-features --features metal,no-cuda,cli` (Apple Silicon).
> See [docs/install.md](docs/install.md).

### One binary, four modes

| Command | What it does |
|---|---|
| `arle` | Interactive REPL + local agent (Eli-compatible). |
| `arle run --prompt "…"` | One-shot agent execution. `--no-tools` to disable tools. |
| `arle serve --backend …` | OpenAI-compatible HTTP server. `--spec-type dspark --dspark-train` enables DSpark block-drafter decode + in-process Markov-head training. |
| `arle train opd` | **On-Policy Distillation** — teacher runs on the serving runtime. |
| `arle --doctor` | Backend / hardware / model self-check. |

Full install matrix, uninstall, and build from source: [docs/install.md](docs/install.md) · Examples: [`examples/`](examples/).

---

## Performance

Measured on real hardware, not projected.

### Apple Silicon (M4 Pro, 48 GB, c=1)

A 35B-A3B MoE decodes as fast as the 4B dense — only ~3B params activate per token.

| Model (Metal 4-bit) | Decode | TPOT | TTFT |
|---|---:|---:|---:|
| Qwen3.5-0.8B | **318 tok/s** | 3.2 ms | 0.17 s |
| Qwen3.5-4B | 84 tok/s | 11.9 ms | 0.82 s |
| Qwen3.5-9B | 50 tok/s | 20.0 ms | 1.45 s |
| **Qwen3.6-35B-A3B (MoE)** | **85 tok/s** | 11.7 ms | 1.23 s |

### Speculative decode beats the HBM wall

Qwen3.6-27B (OptiQ 4/8-bit): the model's own NextN/MTP head drafts, the base verifies — **output is bit-identical to greedy**, **12.3 → 17.75 tok/s (+44%)**, past the 15.2 tok/s HBM floor no kernel can reach.

### NVIDIA (one H20)

The real workload: 32K-token multi-turn agent prompts, not a synthetic short prompt.

| Qwen3.6, 1×H20 · per-request decode tok/s | c=1 | c=8 | c=16 |
|---|---:|---:|---:|
| 35B-A3B MoE | 149.3 | 27.7 | 15.1 |
| 27B dense + DSpark drafter | **91.8** | 20.5 | 11.2 |

`decode tok/s` is per-request decode speed (`1000 / ITL mean`); prefill is reported separately as TTFT. Per-request decode falls with concurrency: verify is free only while the GPU has idle compute. Full rows with TTFT in [docs/baselines.md](docs/baselines.md).

### DeepSeek-V4-Flash (8×H20, TP=8/EP=8, FP8 MoE)

With the DSpark block drafter: 50.4% acceptance at c=1, TTFT p50 170 / 444 / 871 ms at c=1/8/16. Speculation only engages at c=1 — by c=8 the drafter contributes under 1% and those points are plain decode. Full rows in [docs/baselines.md](docs/baselines.md).

### DeepSeek-V4-Flash (2×H20, TP=2, W4AFP8 MoE)

NVFP4 checkpoint (E2M1+E8M0) converted to W4AFP8 (INT4+BF16) at load time — 4-bit weights keep the 167 GB model on 2×96 GB H20 (FP8 needs 4×). B=1 decode **37 tok/s**; prefill 1K **0.48s** (2109 tok/s), 4K **1.1s** (3647 tok/s).

### DeepSeek-V4-Flash (4×H20, TP=4, W4AFP8 MoE)

Same conversion, 4-way tensor parallelism. B=1 decode **47.7 tok/s** (1.29× over TP=2); per-GPU efficiency 11.9 tok/s/GPU vs FP8 TP=8's 6.6 tok/s/GPU.

### DeepSeek-V4-Flash (4×H20, TP=4) — c=1 decode CUDA graph, on by default

The 43-layer c=1 decode body is captured into one CUDA graph per slot and
replayed, with zero allocation nodes inside the capture. On 32K-token agent
prompts, per-request decode at c=1: NVFP4 experts **40.8 → 44.2 tok/s**
(ITL p50 24.1 → **22.2 ms**), FP8 experts **52.4 → 59.5 tok/s**. The gate is
c=1-only, so c≥8 and spec decode are byte-for-byte unchanged; MMLU is identical
to the eager arm on all 200 items. `ARLE_DSV4_DECODE_GRAPH=0` selects eager.

### Qwen3.8-27B-NVFP4 (one H20) — 4-bit that is actually smaller

A mixed-precision checkpoint: NVFP4 MLP (group 16) plus per-channel FP8
everywhere else. Only 54% of the parameters are 4-bit, so the file is 23.4 GB
against the FP8 model's 30.9 GB — 24% fewer bytes, not half.

sm_90 has no FP4 tensor core, so any real GEMM has to widen the nibbles first and
the only question is what to widen *to*. Marlin widens to BF16 and runs at 84
TFLOPS; widening to E4M3 and handing the bytes to DeepGEMM runs at 274, 93% of
this card's FP8 peak. Marlin keeps decode, where reading half the bytes wins.

The operand DeepGEMM wants is derived from Marlin's resident layout into scratch
per call, so **no weight is resident twice** — the first version of this kept both
layouts and cost 10 GB more than the FP8 model it exists to beat.

| vs Qwen3.6-27B-FP8, 32K agent prompts | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| Decode (ITL) | **+21.3%** | **+20.7%** | **+13.2%** | **+5.5%** |
| End-to-end | **+5.0%** | **+15.3%** | **+9.1%** | **+1.8%** |

Resident **22.4 GB** against FP8's 29.4, KV pool 1,779,114 tokens against
1,582,506. Same binary, both arms, back to back. GSM8K-shaped accuracy is
unchanged by the prefill path: 188/200 with the arms on against 189/200 with them
off, 196/200 identical answers. Full rows and the per-op ladder in
[docs/baselines.md](docs/baselines.md).

### Against SGLang

Same weights, same GPU, same quantized kernel — SGLang serves a repack of our own checkpoint. Qwen3.6-27B, one H20, 33K prompt, one request at a time.

| | ARLE | SGLang 0.5.13 |
|---|---:|---:|
| Decode, per token | **16.69 ms** | 17.16 ms |
| Prefill, 33K prompt | 25.0 s | **21.0 s** |

Decode is 2.8% faster. Prefill is 19% slower — that is what we are working on. Numbers and method: [docs/baselines.md](docs/baselines.md).

### On-Policy Distillation

Teacher = production server. Student trains on its own rollouts:

- Qwen3.5-4B: MATH-500 **+27pp** (0.518 → 0.792)
- Qwen3.5-27B: Terminal-Bench pass@1 **+5.1pp** (20.5 → 25.6%)

Method and raw data: [benchmarks/README.md](benchmarks/README.md) · [docs/experience/wins/](docs/experience/wins/).

---

## Why ARLE

Agent and RL workloads re-process the same prompt + history + tool output every turn. ARLE fixes this once and shares the fix across serving and training.

**KV stays hot across turns.** Prior-turn KV stays on GPU; prefix pages are shared across requests via the host radix cache, demote to host RAM under memory pressure, and promote back on next hit — no re-prefill.

**Quantized KV on CUDA.** INT8/FP8 paged-KV behind `--kv-cache-dtype`, Qwen3.5/3.6 family only. Correctness-gated, opt-in (default BF16). DSv4 rejects the flag: its MLA KV is already FP8-packed.

**One runtime, three surfaces.** Serving, the local agent, and OPD training run the same Rust + model code. The OPD teacher *is* the production server.

**DSpark trains while serving.** `--spec-type dspark --dspark-train` runs the DSpark block-drafter for faster decode *and* trains its Markov head in-process: the hot path captures (draft, target, accepted) tuples, a background thread runs acceptance-weighted policy gradient + probability matching, and updated weights hot-swap back into the running engine — no restart, no separate training job. Seeds from the loaded checkpoint so acceptance never regresses at startup.

---

## Architecture

```mermaid
flowchart TD
  CLI["CLI orchestration<br/>crates/cli/src/train_cli.rs"]
  OPD["OPD / self-OPD<br/>student rollout + teacher rescoring<br/>KL / reverse-KL / beta-JSD"]
  RUBRIC["Rubric-OPD<br/>sample -> judge -> accepted CE"]
  AGENT["Agent-OPD<br/>tool trajectory -> reward<br/>UpdatePreset"]
  DSPARK["DSpark online training<br/>verify-logit capture -> Markov-head update"]
  STUDENT["Qwen3.5 / Qwen3.6 train model<br/>crates/train/src/qwen35.rs"]
  TEACHER["infer-api or EMA teacher<br/>BF16 logits boundary"]
  LOSS["Loss graph<br/>CE / KL / JSD / weighted PG"]
  TAPE["Autograd Tape<br/>checkpoint / recompute / offload"]
  BACKEND["Backend seam<br/>CPU reference + optional CUDA overrides"]
  OPT["Production host AdamW<br/>FP32 params and moments"]
  SYNC["LoRA D2H<br/>infer-engine re-merge"]
  ART["Immutable model/adapter artifact<br/>publish latest last"]
  CODEC["Trainer-state v2 codec<br/>not wired to production CLI"]

  CLI --> OPD
  CLI --> RUBRIC
  CLI --> AGENT
  CLI --> DSPARK
  OPD --> STUDENT
  OPD --> TEACHER
  RUBRIC --> STUDENT
  AGENT --> STUDENT
  STUDENT --> LOSS
  TEACHER --> LOSS
  LOSS --> TAPE
  TAPE --> BACKEND
  BACKEND --> OPT
  OPT --> SYNC
  OPT --> ART
  CODEC -. "no production call edge" .-> ART
  DSPARK -->|"two BF16 vocab-wide D2H copies"| DSPCPU["CpuBackend + host AdamW"]
  DSPCPU -->|"BF16 saved head / host hot-swap"| SYNC
```

The training stack shares the production model/runtime authority: serving or EMA teachers score student trajectories, one autograd substrate drives OPD and RFT objectives, and LoRA updates merge back into the live inference engine. The current device-residency and restart gaps are documented in the [training architecture and kernel audit](docs/research/2026-07-26-training-architecture-algorithm-kernel-audit.md).

Deep dive: [docs/onboarding.md](docs/onboarding.md) (30 min) · [docs/architecture.md](docs/architecture.md) · [docs/codebase-map.md](docs/codebase-map.md).

---

## Status

| | CUDA | Metal | OPD Train |
|---|---|---|---|
| **Stability** | Stable | Beta | Beta |
| **Models** | Qwen3.5/3.6/3.8, DeepSeek-V4-Flash, GLM-5.2 | Qwen3-dense, Qwen3.5/3.6, Gemma4, DeepSeek-OCR, DiffusionGemma | CUDA models |

Full tiers: [docs/support-matrix.md](docs/support-matrix.md) · [docs/stability-policy.md](docs/stability-policy.md).

---

## Documentation

[HTTP API](docs/http-api.md) · [Support Matrix](docs/support-matrix.md) · [Architecture](docs/architecture.md) · [Codebase Map](docs/codebase-map.md) · [Environment](docs/environment.md) · [Troubleshooting](docs/troubleshooting.md) · [Contributing](CONTRIBUTING.md) · [All docs](docs/index.md)

---

## License

[MIT](LICENSE)
