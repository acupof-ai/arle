# DSv4 From-Scratch Feasibility

Status: design only. No implementation, no GPU run.

Question: can ARLE directly reproduce the DeepSeek V4 architecture from random
init and train it until `core > 0.25`?

Verdict: **KILL direct 1B scratch training on the local 4070 Ti SUPER path**.
The only licensed continuation I would support is a one-shot architecture
verification ladder: 50M convergence first, then 200M scaling sanity, then 1B
only if external compute is explicitly budgeted.

## Current DSv4 Substrate

What exists:

| Surface | Evidence | Meaning |
|---|---|---|
| Spec crate | `crates/deepseek-spec/src/lib.rs:3-13` exports V4 config, tensor-name, route, MTP, and attention summaries; `Shard` is defined at `:29-42`. | DSv4 has a canonical metadata surface. |
| V4 config validation | `crates/deepseek-spec/src/v4.rs:20-69` defines `DeepSeekV4Config`; `:88-163` validates V4-only invariants. | Config truth exists and is separate from V3. |
| Tensor/shard policy | `crates/deepseek-spec/src/v4.rs:165-183` builds tensor-name surfaces; `:183-191` handles global tensor shard policy. | Loader/runtime can reuse spec truth; no train-side equivalent exists yet. |
| Local target | `infer/models/dsv4-mini-1B-init/README.md:16-24` says this is a randomly initialized 1B DSv4 architecture replica. | It is architecture scaffolding, not a pretrained model. |
| Local shape | `config.json:42-83` has hidden 1024, 24 layers, 16 attention heads, 16 routed experts, top-2 routing, MTP=1, vocab 129280. | About 1B total params, active params lower because MoE. |
| Real parameter count | Safetensors header: 1889 tensors, 1,021,129,744 total params, mostly BF16. File size is 2,045,565,224 bytes. | The checkpoint is exactly a 1.0B-class BF16 init. |
| Runtime support | `docs/support-matrix.md:62` says CPU reference smoke exists, but CUDA optimized V4 attention/MoE/MTP kernels remain pending. | Inference substrate is in-progress, not train-ready. |
| CPU reference | `infer/src/model/deepseek/reference.rs:1-6` identifies the slow Rust CPU reference; `:95-100` loads config/safetensors; `:870-894` mmaps safetensors. | Useful correctness path, not a training path. |
| Smoke tests | `infer/tests/dsv4_v4_1b_smoke.rs:20-48` parses config + manifest; `:50-72` full forward is ignored until kernels land. | The verified gate is metadata, not train convergence. |
| Readiness gaps | `docs/projects/2026-05-01-deepseek-v4-readiness.md:72-82` lists missing CUDA loader/runtime dispatch, MLA cache/kernels, MoE routing, block-FP8, MTP support. | Runtime gaps must close before serious DSv4 train. |
| OPD-only pivot | `docs/projects/2026-05-18-opd-only-pivot.md:7-20` deleted scratch pretrain because ARLE was 322x behind measured single-GPU industry throughput. | T17 must not silently reverse product scope. |

Train-side gap:

- `crates/train` still has autograd, `Trainer`, checkpoint codec, tokenizer,
  LoRA, and OPD substrate (`docs/codebase-map.md:133-149`).
- The actual model loaders and OPD path are Qwen3.5-specific:
  `crates/train/src/qwen35_loader.rs` builds `Qwen35Model`, and
  `crates/cli/src/train_cli.rs:219-241` wires OPD through
  `load_qwen35_from_hf_dir`.
- `crates/cli/src/train_cli.rs:663-686` can inspect DSv4 configs for
  `estimate-memory`; `:781-845` has a DSv4 parameter estimator. That is not a
  DSv4 autograd model, loader, optimizer state codec, or train loop.
- Deleted pretrain data surfaces include `convert_dataset`, `download_dataset`,
  generic dataset adapters, and tokenizer training (`docs/projects/2026-05-18-opd-only-pivot.md:51-64`).

Data pipeline grep result:

- Present: eval/capability dataset scripts, OPD JSONL prompts, tokenizer helpers,
  long-context synthetic generation, and CLI memory estimation.
- Absent: maintained RedPajama/SlimPajama/FineWeb pretokenization, sharded
  token stream loader, restartable pretrain dataloader, or DSv4 train corpus
  path.

## Industry Baseline

The public 1B-class baseline is weak on MMLU even after serious pretraining:

| Model | Params | Public training scale | MMLU | HellaSwag | Notes |
|---|---:|---:|---:|---:|---|
| TinyLlama | 1.1B | 3T-token checkpoint | 26.04 5-shot | 60.31 10-shot | HF card reports these Open LLM Leaderboard numbers. |
| Pythia | 1.0B | 300B tokens | not listed on its HF card; OLMo comparison gives HellaSwag 44.7 | 44.7 | Pythia paper is a controlled suite, not a capability-maximized recipe. |
| OLMo | 1.0B | disclosed open pretraining | not in the 1B table; OLMo card uses 1B core-task table | 62.5 | More relevant than nonexistent OpenLLaMA-1B for 1B-class comparison. |
| OpenLLaMA | 3B/7B public family | not 1B | no solid 1B number | 3B v2 card reports HellaSwag columns | Do not cite as a 1B baseline. |

Interpretation:

- `core > 0.25` on MMLU means "above 4-choice random chance", not a meaningful
  reproduction win.
- A 1B from-scratch model trained well enough to be worth discussing should
  target **MMLU 5-shot >=30%** or a clear loss/perplexity scaling curve that
  predicts that range.
- TinyLlama hitting only 26.04 MMLU after a 3T-token public recipe means
  `>25%` is a weak and noisy bar.

## Compute Budget

Chinchilla-style sizing for a 1B dense-equivalent target is roughly 20B tokens
compute-optimal; serious overtraining recipes can go far beyond that. The local
DSv4 init is 1.021B total params and about 0.46B active params/token, but train
state still has to own the full MoE parameter set.

Memory estimate on 16 GB:

| Item | Approx bytes |
|---|---:|
| BF16 params | 2.0 GB |
| BF16/FP32 grads | 2-4 GB |
| AdamW moments | 8.2 GB if f32 m/v |
| master weights / optimizer scratch | 0-4 GB depending implementation |
| activations | workload-dependent; tight even with checkpointing |

So a 1B DSv4 AdamW run on RTX 4070 Ti SUPER is not impossible to allocate in
the abstract, but it is right at the memory edge before sequence activations,
MoE buffers, router state, and checkpoint/resume overhead.

Wall-clock estimate:

| Throughput assumption | 20B tokens | 40B tokens |
|---|---:|---:|
| Current ARLE pretrain evidence, 174.7 tok/s | 31,801 GPU-hours | 63,601 GPU-hours |
| Nanochat measured industry baseline, 56,291 tok/s | 98.7 GPU-hours | 197.4 GPU-hours |

The OPD pivot already recorded the 174.7 vs 56,291 tok/s gap as 322x
(`docs/projects/2026-05-18-opd-only-pivot.md:14-20`,
`docs/support-matrix.md:148-157`). Unless ARLE reintroduces a highly optimized
pretrain stack, a direct 1B target is over the 1000 GPU-hour KILL threshold by
more than 30x.

Cluster condition:

- PASS-if-cluster only if ckl budgets at least a multi-GPU pretrain backend and
  the implementation uses an industry-class train loop, not the current retired
  ARLE pretrain surface.
- The DSv4 architecture itself also needs train-side MLA, MoE routing, MTP, and
  checkpointing. None of those exist in `crates/train` today.

## Architecture Verification Ladder

This is the only SOLID path I recommend.

### Step 1: Tiny DSv4, about 50M

Goal: prove ARLE can express the DSv4 architecture and learn a non-trivial LM
objective at all.

Scope:

- Tiny DSv4 config with the same operator families: sliding-window, CSA/HCA
  where possible, routed MoE, mHC, and MTP stub if cheap.
- Synthetic or tiny real-token corpus.
- Single 4070 Ti SUPER, <=4 hours.

PASS:

- Training loss decreases monotonically enough to reject wiring failure.
- No NaN/inf.
- Checkpoint save/resume roundtrip works.

KILL:

- 50M does not converge.
- Any required DSv4 op has no autograd path and cannot be stubbed without
  changing the architecture question.

### Step 2: Small DSv4, about 200M

Goal: verify the 50M result scales and the MoE/router path is not fake.

PASS:

- Same corpus and schedule family gives predictable loss improvement over 50M.
- Router/expert utilization counters are non-degenerate.
- Memory and step time extrapolate to 1B within a documented bound.

KILL:

- Scaling curve is worse than 50M or unstable.
- Router collapses or expert imbalance needs a new research project.

### Step 3: 1B Target

Only after Step 1 and Step 2 PASS.

PASS:

- External compute budget exists.
- Train-side DSv4 loader/model/autograd path has parity tests against the CPU
  reference for one-token and short-sequence logits.
- Estimated training cost is below a ckl-approved threshold.

KILL:

- Single local 4070 Ti SUPER remains the only hardware plan.
- End-to-end compute remains >1000 GPU-hours.

## Product Framing

This does not reverse the 2026-05-18 OPD-only pivot.

Recommended framing: **one-shot architecture verification**, not a restored
scratch-pretrain product line. The runtime product still wants DSv4 inference
first: CUDA MLA/MoE/MTP kernels are already #1 next-model priority. A tiny
from-scratch ladder is useful only if it de-risks architecture correctness or
future weight-transfer experiments.

Non-goals:

- No `arle train pretrain-dsv4` resurrection.
- No RedPajama/SlimPajama/FineWeb product pipeline until Step 1 and Step 2 PASS.
- No claim that `MMLU >25%` is a meaningful achievement.

## License-Or-Kill

Feasibility PASS:

- Step 1 50M converges to non-trivial LM loss within 4 hours on one RTX 4070 Ti
  SUPER.
- The resulting plan remains a bounded architecture-verification experiment.

Feasibility KILL:

- 50M does not converge.
- The DSv4 operator set requires large new autograd/runtime work before even a
  50M smoke can run.
- The 1B target still needs >1000 GPU-hours.

Current T17 verdict:

- **KILL direct 1B training now**.
- **CONDITIONAL PASS for a 50M architecture convergence smoke** if ckl wants to
  spend one bounded GPU window after explicitly accepting that this is not a
  product-line reversal.

## References

- ARLE DSv4 support: `docs/support-matrix.md`, `docs/projects/2026-05-01-deepseek-v4-readiness.md`
- ARLE pivot: `docs/projects/2026-05-18-opd-only-pivot.md`
- TinyLlama paper: https://arxiv.org/abs/2401.02385
- TinyLlama 3T checkpoint eval: https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
- Pythia paper: https://arxiv.org/abs/2304.01373
- OLMo 1B comparison table: https://huggingface.co/allenai/OLMo-1B
- OpenLLaMA 3B v2 model card: https://huggingface.co/openlm-research/open_llama_3b_v2
- Chinchilla paper: https://arxiv.org/abs/2203.15556
