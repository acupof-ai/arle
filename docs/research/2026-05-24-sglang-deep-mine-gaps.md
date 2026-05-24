---
title: SGLang deep-mine gaps for ARLE after T6
date: 2026-05-24
type: research
status: source-scan + repo-cross-reference; no implementation
owner: codex
related:
  - docs/projects/2026-05-24-opd-mainline-task-backlog.md
  - docs/plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md
  - docs/support-matrix.md
  - docs/codebase-map.md
---

# SGLang Deep-Mine Gaps for ARLE After T6

## 0. SOLID Status

This note is **hypothesis-grade**. Evidence here means official SGLang docs plus
ARLE repo grep and local truth docs. It is not runtime evidence. Every proposed
implementation below has a license-or-kill gate that must be measured before a
performance or correctness claim is made.

T6 already covers:

| T6 gap | Covered axis |
|---|---|
| G1 | Metal KV pool dual-write / prefix reuse hot path |
| G2 | Metal command-buffer or MLX encode replay license |
| G3 | CUDA graph bucket preheat license |
| G4 | Metal top-k/top-p/penalty sampler parity |
| G5 | CUDA HiCache T2 scheduler hot path license |
| G6 | CUDA RadixCache insert e2e validation |
| G7 | Metal MLX_GUARD mutex license |

This doc only lists gaps not already consumed by those G1-G7 items.

## 1. Source Sweep

| SGLang source | Features mined | T6 overlap |
|---|---|---|
| [SGLang docs index](https://docs.sglang.io/) | Runtime feature inventory: RadixAttention, prefix cache, PD disaggregation, speculative decoding, structured outputs, quantization, multi-LoRA. | Reference only |
| [Structured Outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs) | JSON schema, regex, EBNF, structural tags, OpenAI-compatible `response_format`. | New |
| [Tool Parser](https://docs.sglang.io/docs/advanced_features/tool_parser) | Tool-call parser, `/parse_function_call`, `tool_choice` behavior. | New |
| [Reasoning Parser](https://docs.sglang.io/docs/advanced_features/separate_reasoning) | Model-family reasoning extraction for DeepSeek, Qwen3, GPT-OSS, Kimi, etc. | New |
| [Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding) | EAGLE-2/EAGLE-3, FR-Spec token map, DFlash, draft-token knobs. | New, except Metal DFlash exists |
| [Deterministic Inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference) | Batch-invariant deterministic mode with seeds. | New |
| [LoRA Serving](https://docs.sglang.io/docs/advanced_features/lora) | Multi-LoRA adapters in one batch via S-LoRA/Punica-style serving. | New |
| [Quantization](https://docs.sglang.io/docs/advanced_features/quantization) | W8A8 FP8/INT8, ModelOpt FP8/FP4, torchao configs, compressed quantized model behavior. | Partly new |
| [PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) | Separate prefill/decode workers and router integration. | New, larger architecture |
| [Model Gateway](https://docs.sglang.io/advanced_features/router.html) | Worker lifecycle, OpenAI/gRPC routing, PD mode discovery. | New, later scale-out |
| [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design) | L3 storage backends and PD-disaggregation integration. | T6 G5/T11, no duplicate |

## 2. ARLE Cross-Reference

| ARLE surface | Current status |
|---|---|
| `docs/support-matrix.md` | `/v1/responses` is beta and structured outputs are missing; `xgrammar-sys` is scaffold-only with no HTTP, scheduler, sampler, or GPU integration. CUDA spec decode is not shipped throughput-positive. |
| `infer/src/http_server/openai_v1.rs` | `tool_choice` and `response_format` parse permissively but are explicitly no-ops. Streaming tools are rejected. Non-streaming tool calls are parsed after generation from `<tool_call>` blocks. |
| `crates/chat/src/protocol.rs` | Tool prompt rendering and post-hoc `<tool_call>` parsing exist; constrained tool-call decoding does not. |
| `infer/src/sampler.rs` | Sampling params include seed, top-k, top-p, min-p, and penalties; this is not a batch-invariant deterministic serving mode by itself. |
| `infer/src/speculative.rs` and `infer/src/speculative/cuda.rs` | Spec decode substrate exists, but support-matrix status is not shipped. |
| `infer/src/model/qwen35/lora.rs` and `infer/src/model/qwen35/weights.rs` | Qwen3.5 serve can merge one PEFT LoRA adapter into dense weights at load time; this is not request-selectable multi-LoRA batching. |
| `crates/train/src/qwen35_loader.rs`, `crates/train/src/lora.rs` | OPD train-side LoRA is a first-class student path; serve-side multi-adapter selection is not. |
| `docs/codebase-map.md` | TP/PP/EP scaffolds exist, but production scale-out routing and PD split are not wired. |

## 3. Candidate Matrix

| ID | Candidate | ARLE relevance | New vs T6 | First license-or-kill gate | Verdict |
|---|---:|---:|---|---|---|
| T7-A | Structured-output + tool-choice constrained decoding | 10/10 | Yes | CPU-first schema/tool prompt suite reaches 100% valid outputs without post-hoc repair; GPU path adds <=5% greedy overhead for small JSON schema and <=15% for tool-call schema. | Do next after current queue gates |
| T7-B | Responses/tool/reasoning parser parity | 8/10 | Yes | OpenAI Responses + chat fixtures cover non-streaming and streaming function-call deltas plus reasoning split; no regression in existing chat/tool tests. | Pair with T7-A |
| T7-C | Throughput-positive CUDA spec-decode reboot via EAGLE/MTP/adaptive gate | 8/10 | Yes | P5-free GPU run shows >=1.25x wall-clock output tok/s at c=1 and c=4, with greedy output identity or distribution-preserving stochastic acceptance. | GPU-deferred |
| T7-D | OPD teacher FP8/W8A8 compressed-tensors load path | 8/10 | Partly | Qwen3.5/Qwen3.6 FP8 teacher checkpoint loads, emits finite logits, passes 20-step OPD smoke within 16 GB, and does not degrade heldout KL vs BF16 beyond an agreed threshold. | OPD-priority spike |
| T7-E | Batch-invariant deterministic serving mode | 7/10 | Yes | Same prompt, seed, and sampling params produce byte-identical output across single vs mixed batches; overhead <=5% for greedy and <=10% for seeded sampling. | Medium priority |
| T7-F | Multi-LoRA adapter serving | 7/10 | Yes | Two or more adapters can be selected per request in one batch; no-adapter overhead <=5%, active-adapter overhead <=15%, output matches merged-adapter baseline. | After OPD adapter flow stabilizes |
| T7-G | Cache-aware router / worker affinity | 6/10 | Yes | Multi-worker synthetic W3/W4 keeps prefix hit rate >=90% and improves TTFT p50 >=25% vs round-robin. | Later scale-out |
| T7-H | PD disaggregation / prefill-decode split | 5/10 | Yes | At c>=16 with long prompts, TTFT p50 improves >=25% without ITL p99 regression >10%. | Architecture license needed |
| T7-KILL-1 | New HiCache L3 backend deep-dive | n/a | No | Already covered by T6 G5 and T11 storage/transport design. | Kill as duplicate |
| T7-KILL-2 | New CUDA graph/piecewise graph item | n/a | No | Already covered by T6 G3. | Kill as duplicate |
| T7-KILL-3 | New Metal sampler parity item | n/a | No | Already covered by T6 G4. | Kill as duplicate |

## 4. Candidate Details

### T7-A Structured-output + tool-choice constrained decoding

SGLang treats JSON schema, regex, EBNF, structural tags, and tool-choice forcing
as decode-time constraints. ARLE currently accepts `response_format` and
`tool_choice`, but `openai_v1.rs` documents both as no-ops, and tool calls are
post-hoc parsed from generated text.

Why it matters for ARLE: W3/W4 agent workloads depend on valid JSON and tool
arguments. Post-hoc parsing is not enough when the model drifts, truncates a
JSON object, or ignores `tool_choice="required"`.

License-or-kill:

- **PASS:** `crates/xgrammar-sys` drives a CPU-first constrained decoder over
  a tiny vocab fixture and an HTTP-level fixture for `response_format` and
  `tool_choice`. 500-1000 deterministic tool prompts produce 100% schema-valid
  JSON/tool args with no repair pass. Later GPU gate: <=5% overhead for a small
  JSON schema and <=15% overhead for a tool-call schema.
- **KILL:** any design requires per-token logits D2H on the production CUDA or
  Metal path, or pushes overhead above 15% on W3/W4.

Implementation boundary: do not rewrite grammar matching in Rust; continue
using `xgrammar-sys`. Start with CPU/no-cuda tests and a sampler-side bitmask
API, then wire HTTP.

### T7-B Responses/tool/reasoning parser parity

SGLang separates parser concerns: tool-call parsers, reasoning parsers, and
Responses API behavior are explicit runtime surfaces. ARLE already has
`/v1/responses`, prompt-side tools, and post-hoc function-call parsing, but
streaming tool calls are rejected and there is no model-family reasoning split.

Why it matters for ARLE: local agent and OPD eval surfaces need stable event
schemas. Reasoning and function-call deltas are API contracts, not just text
formatting.

License-or-kill:

- **PASS:** chat and Responses fixtures cover non-streaming tool calls,
  streaming function-call deltas, `tool_choice` enforcement, and reasoning split
  for at least Qwen3-style and DeepSeek-style tags. Existing `crates/chat` and
  HTTP tests stay green.
- **KILL:** if model-family parsers become a broad compatibility layer without
  an ARLE consumer, keep only Qwen3/Qwen3.5 and DeepSeek V4.

T7-B should share parser definitions with T7-A so constrained tool decoding and
post-hoc parsing do not diverge.

### T7-C CUDA spec-decode reboot with EAGLE/MTP/adaptive gate

SGLang's spec-decode docs emphasize EAGLE-2/EAGLE-3, FR-Spec vocabulary maps,
and DFlash/NEXT-token families. ARLE has a substantial CUDA spec substrate, but
the support matrix says no CUDA spec-decode mode is shipped
throughput-positive, and older self/external/Medusa attempts were killed or
blocked.

Why it matters for ARLE: if OPD teacher inference remains decode-bound after
T2/T5b, spec decode is still one of the few levers that can reduce wall-clock
rollout cost. But prior failures mean acceptance ratio alone is not evidence.

License-or-kill:

- **PASS:** on GPU after P5, c=1 and c=4 wall-clock output tok/s improve by
  >=1.25x against no-spec, with temperature=0 output identity or a stochastic
  distribution-preserving test. The gate must include target verifier cost,
  draft KV memory, and scheduler overhead.
- **KILL:** acceptance looks good but wall-clock improves <1.25x, or draft KV
  cuts the target KV pool enough to regress long-context admission.

Implementation boundary: no new speculative runtime flag until the gate is met.
Prefer a measured MTP/EAGLE spike over reviving the killed classical path.

### T7-D OPD teacher FP8/W8A8 compressed-tensors load path

SGLang documents W8A8 FP8/INT8 and ModelOpt FP8/FP4 deployment paths. ARLE has
strong quantization coverage, but OPD still has practical teacher-load gaps:
the support matrix is broad, while local OPD plans still call out loader and
quality gates for FP8 or compressed teacher variants.

Why it matters for ARLE: OPD is the main training axis. A smaller/faster
teacher can change whether 4B teacher + student LoRA fits comfortably on the
local 16 GB GPU and whether longer prompts are usable.

License-or-kill:

- **PASS:** target FP8/W8A8 teacher checkpoint loads through the native loader,
  logits are finite, a 20-step OPD smoke reaches train/eval summaries within
  16 GB, and heldout KL does not degrade beyond the task-specific threshold.
- **KILL:** loader succeeds but output quality collapses, repeated-token
  pathology appears, or memory saved is offset by activation/dequant overhead.

Implementation boundary: this is an OPD teacher path first, not a generic
quantization expansion. Do not mix it with unrelated AWQ/GPTQ parity work.

### T7-E Batch-invariant deterministic serving mode

SGLang exposes deterministic inference as a serving mode compatible with
chunked prefill, graph execution, radix cache, and seeded sampling on selected
attention backends. ARLE has deterministic tests and request-level seeds, but
no explicit serving mode that promises batch-invariant output across queue
composition.

Why it matters for ARLE: OPD evals, regression tests, and user-facing tool
loops need reproducible outputs when a request is run alone or co-scheduled.

License-or-kill:

- **PASS:** with `--deterministic` or equivalent request policy, identical
  prompt/seed/sampling params produce byte-identical output across single,
  batched, and mixed warm-prefix runs. Greedy overhead <=5%; seeded sampling
  overhead <=10%.
- **KILL:** deterministic mode requires disabling the production attention path
  or costs >10% on normal serving.

Implementation boundary: start by making the guarantee explicit in tests before
adding knobs. Do not treat `seed` alone as sufficient.

### T7-F Multi-LoRA adapter serving

SGLang serves multiple LoRA adapters inside one batch. ARLE has train-side LoRA
and serve-side single-adapter merge/load paths, but no request-level adapter
selection, adapter cache, or multi-adapter batching.

Why it matters for ARLE: OPD produces adapter checkpoints. Serving those
adapters without materializing a new dense model per student is the natural
closed loop for compare/eval and local agent personalization.

License-or-kill:

- **PASS:** a server can preload or lazy-load at least two adapters, select one
  per request, batch mixed adapter/no-adapter rows, match a merged-adapter
  baseline, and keep no-adapter overhead <=5% / active-adapter overhead <=15%.
- **KILL:** mixed adapters force one forward pass per adapter and throughput
  falls >15% on the target workload.

Implementation boundary: start with Qwen3.5 OPD adapter-only checkpoints; avoid
generic PEFT breadth until the OPD-produced adapter loop works.

### T7-G Cache-aware router / worker affinity

SGLang's gateway/router owns worker lifecycle and PD-aware routing. ARLE has a
single-runtime front door plus distributed scaffolds, but no cache-aware
multi-worker routing policy that preserves RadixCache locality across requests.

Why it matters for ARLE: session and system-prompt affinity directly affects
prefix reuse. Round-robin routing can destroy a cache win that the single
worker would have kept.

License-or-kill:

- **PASS:** in a two-worker local harness, sticky/cache-aware routing keeps
  prefix hit rate >=90% on W3/W4 synthetic sessions and improves TTFT p50
  >=25% vs round-robin without queue p99 regression >10%.
- **KILL:** single-worker saturation dominates or the added router hop erases
  locality gains.

Implementation boundary: this is a scale-out project, not a prerequisite for
single-GPU OPD. Keep it behind T11 and multi-worker runtime decisions.

### T7-H PD disaggregation / prefill-decode split

SGLang treats prefill and decode as separate deployable roles and routes
between them. ARLE currently uses unified engines. T6 focuses on local
scheduler/cache gaps, not a deployment split.

Why it matters for ARLE: long prompts and multi-session agent loads can be
prefill-heavy, while OPD rollouts can be decode-heavy. A split can help if both
phases have independently saturated resources.

License-or-kill:

- **PASS:** at c>=16 with long prompts, separate prefill/decode workers improve
  TTFT p50 >=25% and do not regress ITL p99 >10%. The measurement must include
  KV transfer cost.
- **KILL:** transfer overhead or queue imbalance offsets the split, or the
  single-node HBM budget is still the bottleneck.

Implementation boundary: architectural license required before code. This is
not a code-only follow-up.

## 5. Killed As Duplicate Or Low Signal

| Item | Verdict | Reason |
|---|---|---|
| More HiCache L3 backend exploration | KILL duplicate | T6 G5 owns scheduler hot-path license; T11 owns storage/transport design. |
| Piecewise CUDA graph / CUDA graph bucket work | KILL duplicate | T6 G3 owns the nsys + wall-clock license. |
| Metal top-k/top-p sampler parity | KILL duplicate | T6 G4 owns the code-only sampler gap. |
| Chunked prefill tuning | KILL duplicate | Already present in scheduler policies and T6/T4 observability path. Needs measurement, not a new T7 gap. |
| Generic RL serving integration | KILL low signal | ARLE pivot is OPD only; SGLang RL docs do not add a concrete OPD implementation gap beyond deterministic rollout and teacher serving already listed. |

## 6. Suggested Order After T7

1. T7-A structured/tool constrained decoding, CPU-first.
2. T7-B Responses/tool/reasoning parser parity, sharing T7-A parser contracts.
3. T7-D OPD teacher FP8/W8A8 loader spike, after current dirty loader WIP is
   audited by T8.
4. T7-E deterministic serving mode, because it improves OPD/eval evidence.
5. T7-C spec-decode reboot, GPU-deferred until P5 is idle.
6. T7-F multi-LoRA serving, once OPD adapter checkpoints are stable.
7. T7-G/T7-H scale-out router and PD disaggregation, after T11 transport design.

## 7. References

Official SGLang docs:

- [SGLang documentation index](https://docs.sglang.io/)
- [Structured Outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs)
- [Tool Parser](https://docs.sglang.io/docs/advanced_features/tool_parser)
- [Reasoning Parser](https://docs.sglang.io/docs/advanced_features/separate_reasoning)
- [Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
- [Deterministic Inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference)
- [LoRA Serving](https://docs.sglang.io/docs/advanced_features/lora)
- [Quantization](https://docs.sglang.io/docs/advanced_features/quantization)
- [PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- [SGLang Model Gateway](https://docs.sglang.io/advanced_features/router.html)
- [HiCache System Design](https://docs.sglang.io/docs/advanced_features/hicache_design)

ARLE truth surfaces:

- [docs/projects/2026-05-24-opd-mainline-task-backlog.md](../projects/2026-05-24-opd-mainline-task-backlog.md)
- [docs/plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md](../plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md)
- [docs/support-matrix.md](../support-matrix.md)
- [docs/codebase-map.md](../codebase-map.md)
- [docs/plans/M_xgrammar-ffi-scaffold.md](../plans/M_xgrammar-ffi-scaffold.md)
