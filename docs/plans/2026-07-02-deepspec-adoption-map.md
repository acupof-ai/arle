# DeepSpec adoption map — DSpark component verdicts (C2, #125)

> Status: Active — scouting verdict landed 2026-07-02; vendoring deliberately
> deferred to C4 (see §4). Umbrella: [#123](https://github.com/cklxx/arle/issues/123).

Upstream: [`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec),
MIT, pinned survey commit **`afdfa7c9382a`** (2026-06-30, branch `main`, ~4 MB,
97.9% Python). Full-stack training/eval codebase for DSpark + DFlash + EAGLE-3
draft models. Mechanism survey: [`../architecture-dsv4.md`](../architecture-dsv4.md) §7.

## 1. Checkpoint inventory (HuggingFace, verified 2026-07-02)

| HF repo | Target | Feeds child |
| --- | --- | --- |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` | DSv4-Flash | [#128](https://github.com/cklxx/arle/issues/128) C5 |
| `deepseek-ai/DeepSeek-V4-Pro-DSpark` | V4-Pro | not carried (no V4-Pro lane) |
| `deepseek-ai/dspark_qwen3_{4b,8b,14b}_block7` | Qwen3-dense | [#126](https://github.com/cklxx/arle/issues/126) C3 first target |
| `deepseek-ai/dspark_gemma4_12b_block7` | Gemma4-12B | [#131](https://github.com/cklxx/arle/issues/131) C8 (gated) |

No public head exists for: Qwen3.6/Qwen3.5-MoE ([#129](https://github.com/cklxx/arle/issues/129)
C6 needs a C4 export), Qwen3.5 hybrid ([#130](https://github.com/cklxx/arle/issues/130)),
GLM-5.2. Those are the C4 training-harness customers.

**Premise correction vs the original child briefs:** the V4-Flash head is
public — C4 was scoped as "DSv4-Flash has no public head; train one". C4 is
re-scoped to: adopt `DeepSeek-V4-Flash-DSpark` for C5; the training harness
serves the no-public-head models above + any fine-tune against our serving
distribution if adopted-head acceptance underperforms the paper claim.

## 2. Component verdicts

| Component | Verdict | Rationale |
| --- | --- | --- |
| Draft checkpoints (table above) | **ADOPT** | Official trained artifacts; adopt-official-first (2026-06-06 retro). Acceptance re-measured under our bench spec before any perf claim — paper numbers stay hypothesis. |
| Draft-model forward: parallel backbone + Markov/RNN sequential head + confidence head | **PORT** to `infer-cuda`/`infer-metal` | No Python on the hot path. Tensor contract comes from the checkpoints; architecture reference is the `deepspec/` Python model defs at the pinned commit. Metal side lands as a third `DraftKind` config (C6 pattern). |
| Confidence-scheduled verify-length scheduler | **PORT** (redesign at the seam) | The decision point belongs in `infer-core`'s scheduler where live batch/SLO state lives (survey §7.5) — their Python scheduler is reference semantics only. C1 probes this with draft-logit confidence before any trained head is required. |
| Training pipeline (data prep, multi-GPU trainer) | **REFERENCE now, vendor at C4 start** | Offline tooling, allowed off the hot path. TileKernels precedent: don't-submodule, port-selectively; pin the commit until a child actually consumes the code. |
| Eval harness (9-benchmark suite) | **SKIP** | ARLE's bench-and-trace spec + needle gate are the licensing instruments; their eval numbers serve as cross-checks only. |
| EAGLE-3 / DFlash reference impls | **SKIP as integration; KEEP as A/B baseline** | C3's acceptance gate measures against the vendored EAGLE-3 numbers on the same prompts, not by integrating a second lane (no-half-states). |

## 3. Port contract anchors (for C3/C6 pickup)

Draft config from the reference impl (survey §7.2): `draft_hidden_size~1024`,
`num_draft_layers~5`, `block_size=7` (matches every shipped checkpoint name),
Markov head `markov_rank~256`, target hidden taps from a sparse layer set
(e.g. `[1,9,17,25,33]`). Verify/rollback reuse on CUDA:
`forward_tokens_verify_scheduled` + `Dsv4SpecRingSnapshot`; on Metal:
`DraftKind` + `qwen35_speculative_block`. Exact per-tensor mapping is C3's
first task, read off `deepspec/` model defs + checkpoint safetensors headers.

## 4. Why vendoring is deferred

C3 needs only (a) a checkpoint and (b) the architecture definition — both
consumable from the pin without carrying 4 MB of Python in-tree. `vendor/`
currently holds code we *build* (FlashMLA/DeepGEMM/DeepEP); a Python trainer
enters the tree the day C4 runs it, not before. Re-pin on vendor.
