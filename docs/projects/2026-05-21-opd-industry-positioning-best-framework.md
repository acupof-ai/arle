# 2026-05-21 — Industry OPD positioning: how ARLE becomes "the best OPD framework"

> **Status:** strategy / positioning. Maps the public On-Policy Distillation
> landscape (mid-2025 → 2026), identifies where ARLE already wins, calls out
> the remaining gaps that block "best framework" claims, and ranks the
> concrete next tranches by ROI. Companion to the
> [cycle wrap](2026-05-21-opd-cuda-cycle-wrap.md) and
> [usage manual](2026-05-21-arle-opd-cuda-usage-manual.md).

## What "OPD" means today (industry consensus)

On-Policy Distillation = a **frozen teacher** scores the **student's own
greedy/sampled rollouts** with a dense per-token signal (forward-KL, or
reverse-KL in MiniLLM's variant). The combination buys you:

- **RL-grade error-correction** (student sees its own mistakes, not the
  teacher's curated traces) — addresses the "distribution mismatch" issue
  Agarwal et al. 2024 named.
- **SFT-grade reward density** (every token gets a full teacher
  distribution, not a sparse reward at end-of-sequence) — orders of
  magnitude cheaper than GRPO / PPO / DPO on real budgets.
- **Small-model recovery** without the "compounding error on long chains"
  failure mode that pure SFT distillation exhibits.

The three foundational references in mid-2025 → 2026:

| Lab / paper | Year | Variant | Key contribution |
|---|---:|---|---|
| **Agarwal et al. (DeepMind), GKD** — ["On-Policy Distillation of Language Models"](https://arxiv.org/abs/2306.13649) | 2023→2024 ICLR | forward-KL, λ-interpolated on/off-policy | Defined the mixing-distribution framework. Setting λ=1 = purely on-policy. |
| **Gu et al. (Microsoft / Tsinghua), MiniLLM** — ["MiniLLM: On-Policy Distillation of Large Language Models"](https://arxiv.org/abs/2306.08543) | 2023→2024 | **reverse-KL** to avoid forward-KL's mode-covering pathology | The HuggingFace TRL `MiniLLMTrainer` is a direct implementation. |
| **Thinking Machines Lab (Mira Murati)** — ["On-Policy Distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/) blog post | 2025 Oct | Forward-KL on student rollouts; "RL-parity at a fraction of the cost" framing | Reproduced Qwen3 RL parity via OPD on a fixed compute budget. Cited Qwen 38× — the moment OPD went mainstream for production teams. |

Plus a 2026 survey ([arxiv:2604.00626](https://arxiv.org/pdf/2604.00626))
that catalogs all the variants now in use (GKD / MiniLLM / GOLD /
OPSDL / X-KD / "relaxed on-policy" / self-distill / context-distill).

## Where the industry actually runs OPD today

| Path | Stack | Notes |
|---|---|---|
| **HuggingFace TRL** | `GKDTrainer`, `MiniLLMTrainer` (Python, PyTorch, accelerate) | Most-used open-source path. Same TRL repo people already use for SFT/DPO/PPO. Cross-tokenizer support broken as of 2026 ([issue #4562](https://github.com/huggingface/trl/issues/4562)). |
| **TM Lab / labs with budget** | Likely internal stacks; some open code on HF Spaces ([h4-on-policy-distillation](https://huggingfaceh4-on-policy-distillation.hf.space)) | Production-quality with the Thinking Machines recipe. Closed-source for most labs. |
| **vLLM + verl / OpenRLHF** | Production RL stacks with grad-eval fast paths | Some teams retrofit OPD via the verl distillation tooling rather than using TRL directly. |
| **Axolotl / LLaMA-Factory** | Higher-level recipe stacks | OPD recipes ship as YAMLs on top of HF transformers. |

The common substrate is **Python + PyTorch + HF transformers**. Everyone
inherits PyTorch's perf, PyTorch's memory model, PyTorch's deps. There is
no production-grade pure-Rust OPD substrate that we could find.

## What ARLE has *that no public framework has*

Cross-referencing the ARLE state from today's session against the
landscape:

| Capability | Industry default | ARLE today |
|---|---|---|
| **Pure-Rust serving + training, no Python on hot path** | Doesn't exist publicly | ✓ (this whole repo) |
| **OPD step at <100 ms on Qwen3-0.6B + RTX 4070 Ti SUPER** | PyTorch CUDA hits ~83 ms on a smaller moderate shape; not benchmarked at Qwen3-0.6B in TRL | ✓ 0.164 s/step at production Qwen3-0.6B; **48.5 ms on moderate shape, 1.71× faster than PyTorch CUDA reference** |
| **Single binary, no Python or conda env required** | `pip install -e .[opd]` + 50+ deps | ✓ `cargo build --features cuda` |
| **OPD step bit-equivalent CPU ↔ CUDA (debuggability)** | PyTorch's CPU/CUDA bit-equivalence is best-effort | ✓ relerr 1.276e-6 verified gate |
| **License-or-kill discipline on every perf claim** | Variable — many TRL benchmarks aren't reproducible | ✓ every perf commit has a numerical kill criterion in the wins entry |
| **Same runtime authority across serve / agent / OPD** | Each surface is a separate stack (vLLM serving + TRL training + Open WebUI agent) | ✓ `infer` runtime is shared; teacher inference is the production serving path |
| **Convergence verified end-to-end on a real checkpoint** | TRL's HF Space demo, occasionally published runs | ✓ Qwen3-0.6B real-checkpoint, lr=1e-7, 5000 steps, held-out exact-overlap 50 → 82.8 %, KL still falling |

The differentiator is **production runtime + training share the same code
path**, with no Python-on-the-hot-path tax. Everyone else has a Python
training-time → ONNX/whatever serving-time hop.

## What's missing to credibly claim "best OPD framework"

Ranked by user-visible impact for someone evaluating OPD frameworks
mid-2026:

### Critical gaps (block "best framework" claim)

1. **Tokenizer integration + real-text prompt support.** Currently OPD
   training takes hand-picked token-ID arrays. A user wants to point at
   a JSONL of text prompts. → Add `--prompts-file <jsonl>` that loads
   Qwen3 tokenizer (via tokenizers crate or via PyO3 to the .venv's
   transformers) and tokenizes inline. Single tranche, ~half-day.
2. **Public head-to-head benchmark vs TRL `GKDTrainer`.** Same model,
   same prompts, same hardware → throughput numbers + held-out quality
   side-by-side. Even if we lose on some axis, a transparent comparison
   builds credibility faster than any internal claim. Single tranche,
   ~half-day with the tokenizer in place.
3. **LoRA-only OPD recipe + bench.** Most production OPD runs full-finetune
   only base/instruct-aligned models; everyone else uses LoRA for
   distillation. The `LinearWithLora` substrate exists (used in tests);
   need a dedicated `opd_step_cuda_lora_bench.rs` + wins entry showing
   memory + speed delta. Single tranche, ~half-day.
4. **CHANGELOG.md + announcement-ready blurb.** No external user can
   discover today's 32-commit session unless they read the wrap doc.
   Need a CHANGELOG entry and a homepage-callout. ~1 hour.

### High-value gaps (move ARLE from "competent" to "compelling")

5. **Multi-tokenizer support / cross-tokenizer OPD.** TRL has this broken
   as of 2026 ([#4562](https://github.com/huggingface/trl/issues/4562)).
   Implementing it correctly (Universal-Logit-Distillation / ULD
   alignment) would be a marquee differentiator. Multi-day.
6. **Reverse-KL option** (MiniLLM-style) alongside the current forward-KL.
   Some recipes prefer it. Adds optionality without removing anything.
   ~1 day.
7. **Streaming convergence dashboard.** Live `held-out KL / NLL / overlap`
   tracking via the existing CLI metrics, exported to a local TUI or
   web view. Users want to see the curve update during training.
   ~1-2 days.
8. **Multi-GPU / DDP support.** Today's substrate is single-GPU. For OPD
   at Qwen3-4B or larger, this is mandatory. Substantial — but the
   `crates/cuda-kernels/src/collective.rs` NCCL scaffolding already
   exists for the inference path. Multi-week.

### Long-tail gaps (nice-to-have, not blocking)

9. **bf16 / fp8 OPD path.** Memory savings for larger models; precision-
   gated.
10. **GRPO+OPD hybrid recipe.** Some teams want "OPD warm-up + GRPO
    polish." We retired GRPO 2026-05-18 but could reinstate as an
    optional pipeline.
11. **Distillation cookbook.** Step-by-step "Distill Qwen3-7B → 0.6B"
    walkthrough as a top-level doc.

## Recommended sequencing (next 1-2 weeks)

To claim "best OPD framework" credibly, the minimum viable bundle is:

| # | Tranche | Time | What it unlocks |
|---:|---|---|---|
| 1 | Tokenizer + `--prompts-file <jsonl>` | half-day | Real-text supervision; required for everything below |
| 2 | LoRA-only OPD bench + wins entry | half-day | Production recipe + memory budget on consumer GPU |
| 3 | Head-to-head TRL benchmark | half-day | Public credibility |
| 4 | CHANGELOG + homepage callout | 1 hr | Discovery |
| 5 | Reverse-KL option | 1 day | Optionality vs MiniLLM |

After (1)-(4), ARLE has the **fastest, cleanest, only-pure-Rust OPD
framework with a verified Qwen3-0.6B real-checkpoint result and a
side-by-side TRL benchmark**. That's a defensible "best for this
workload" claim.

(5) and the high-value gaps make it a credible default-choice for any
team doing distillation outside the HF ecosystem.

## Open question — public release packaging

ARLE the project is MIT-licensed and on GitHub. The OPD CUDA work
exists in-tree. To get OPD users to *find* ARLE, the missing pieces are:

- **A 2-paragraph distillable claim** in the README's intro
  (currently OPD is mentioned but as a sub-feature, not the headline)
- **A dedicated `docs/opd.md` landing page** linking to the usage manual
  + benchmarks + Qwen3 result
- **Crates.io publish** of the relevant `train` + `autograd` crates
  so that downstream users can `cargo add` rather than git-clone

These are positioning moves, not engineering moves; left for the user
to decide priority.

## Sources

- Thinking Machines Lab — [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Agarwal et al., DeepMind — [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) (ICLR 2024)
- Gu et al., Microsoft / Tsinghua — [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- HuggingFace TRL — [MiniLLM Trainer docs](https://huggingface.co/docs/trl/main/minillm)
- HuggingFace TRL — [GKD trainer + cross-tokenizer issue #4562](https://github.com/huggingface/trl/issues/4562)
- Survey — ["A Survey of On-Policy Distillation for Large Language Models"](https://arxiv.org/pdf/2604.00626) (2026)
- HuggingFace H4 — [Unlocking On-Policy Distillation for Any Model Family](https://huggingfaceh4-on-policy-distillation.hf.space/)
