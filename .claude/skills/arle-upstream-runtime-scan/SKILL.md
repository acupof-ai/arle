---
name: arle-upstream-runtime-scan
description: Use this skill when ckl asks to ground an ARLE serving/runtime, model-path, benchmark, capacity, Qwen3.5/DeepSeek, scheduler, paged_kv, MLX, autograd, or OPD decision in upstream SGLang/vLLM/TensorRT-LLM evidence before changing local code. It distills BBuf AI-Infra Auto Driven SKILLS into an ARLE-specific source-survey workflow without symlinking or vendoring those repositories.
version: 1.0.0
---

# arle-upstream-runtime-scan

This is an ARLE source-survey skill, not a replacement for local measurement.
Use it to decide what upstream evidence to read before an ARLE runtime change,
then license or kill the local decision with ARLE logs, benches, traces, and
tests.

## Scope

Use for:

- Comparing ARLE against SGLang, vLLM, TensorRT-LLM, or MLX-style serving.
- Choosing a Qwen3.5, Qwen3-Next, DeepSeek V3/V4, MoE, attention, cache,
  sampler, loader, or quantization precedent before local implementation.
- Explaining capacity, KV-cache budget, memory fraction, max tokens,
  continuous batching, prefix-cache, or queue-growth behavior.
- Reading external profiler traces where prefill/decode stage separation
  matters.
- Designing OPD/eval serving baselines that need a fair external reference.

Do not use for:

- Local kernel retuning. Use `kernel-optimization`; BBuf KernelWiki and
  KernelPilot are merged there.
- Pure SGLang PR review, production incident response, or architecture-diagram
  lookup. Those BBuf skills are not ARLE-specific enough.
- Any claim that can be answered by local ARLE code, docs, or measurement
  without upstream context.

## Non-Negotiables

- Start from ARLE truth surfaces: `docs/index.md`, relevant `AGENTS.md`,
  current code, and recent `docs/experience/{wins,errors}`.
- Treat upstream source survey as hypothesis-grade. It may suggest a design or
  risk; it does not prove ARLE behavior.
- Fetch external repos into `/tmp` for inspection when needed. Do not clone,
  symlink, vendor, or copy BBuf repos into `.claude/skills/`.
- Summarize and cite source URLs. Do not paste long source passages or port
  copied text into this repo.
- Preserve ARLE benchmark protocol: matched workload, version/commit capture,
  internal counters when available, and wall-clock framing as ground truth.

## Workflow

### 1. Classify The Question

Pick exactly one primary lane before reading upstream material:

| Lane | Use when | ARLE anchor |
| --- | --- | --- |
| Fair serving benchmark | ckl asks "ARLE vs SGLang/vLLM" or wants a best external baseline | `docs/bench-and-trace-spec.md`, `scripts/bench_guidellm.sh` |
| Capacity / OOM / KV budget | memory pool, max tokens, KV dtype, mem fraction, request capacity | `docs/support-matrix.md`, server logs, `/v1/stats`, `nvidia-smi` |
| Model PR history | Qwen3.5, DeepSeek V4, Qwen3-Next, MoE, loader, sampler, cache path | `infer/src/model/`, `crates/*-spec/`, active project docs |
| Trace triage | external torch profiler or ARLE nsys/ncu points at a stage | `docs/resources/*profiling*`, `scripts/profile_*` |
| OPD/eval baseline | serving correctness or capability numbers need external validation | `scripts/arle_capability_eval.py`, train/eval docs |

If the lane is kernel-local, stop and switch to `kernel-optimization`.

### 2. Fetch Only What You Need

For a fresh upstream scan:

```bash
mkdir -p /tmp/arle-bbuf-scan
git clone --depth 1 https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS /tmp/arle-bbuf-scan/AI-Infra-Auto-Driven-SKILLS
```

If already cloned, use `git -C <repo> fetch --depth 1` and record
`git rev-parse HEAD`. Read only matching skill files or model-history pages.

Relevant source areas:

- `model-pr-optimization-history/SKILL.md`
- `model-pr-optimization-history/{sglang,vllm}/{qwen35,qwen3-core,qwen3-next,deepseek-v4}/`
- `skills/llm-serving-auto-benchmark/SKILL.md`
- `skills/llm-serving-capacity-planner/SKILL.md`
- `skills/llm-torch-profiler-analysis/SKILL.md`
- `skills/model-compute-simulation/SKILL.md`
- `skills/llm-pipeline-analysis/SKILL.md`

Kill these by default for ARLE: `sglang-humanize-review`,
`sglang-prod-incident-triage`, `sglang-sota-humanize-loop`,
`model-architecture-diagram`. They are SGLang/plugin/operator UX skills unless
ckl explicitly asks for that external task.

### 3. Model PR History Pass

Use model PR history when the local task touches model-specific runtime
behavior or an external framework is already faster.

Useful queries from the BBuf checkout:

```bash
python3 model-pr-optimization-history/scripts/query.py --list
python3 model-pr-optimization-history/scripts/query.py --framework sglang --model qwen35 --paths-only
python3 model-pr-optimization-history/scripts/query.py --framework vllm --model deepseek-v4 "fused norm router" --limit 5
python3 model-pr-optimization-history/scripts/query.py --framework vllm "qwen3.5 gated delta net cache sampler" --limit 8
```

Extract only:

- upstream PR number, URL, framework, status, and model family;
- touched source files and symbols;
- optimization type: fusion, overlap, quantization, attention/cache, sampler,
  loader, graph capture, or scheduler;
- validation lane and regression risk;
- exact ARLE file or module that might be influenced.

Do not copy code snippets from upstream PR cards into ARLE. If a code-level
idea matters, inspect the real upstream source/PR under its license and record
the URL in the experience entry.

### 4. Fair Benchmark Pass

Use the BBuf benchmark material as a checklist, then run ARLE's own bench
protocol.

Required matched-control fields:

- model and checkpoint path;
- tokenizer;
- prompt/output distribution;
- concurrency or request-rate profile;
- max sequence length, batched-token budget, num slots, KV dtype, prefix cache,
  speculative decode, and graph/capture settings;
- framework versions or commit hashes;
- launch command, benchmark command, raw output table, and failed candidates.

Do not crown a framework winner until each requested framework has had its main
serving knobs tuned or explicitly fixed by the experiment design. Do not search
memory fractions by default unless capacity is the lane.

For ARLE, use `scripts/bench_guidellm.sh` and the report skeleton under
`docs/experience/wins/`. External framework tools are references, not the ARLE
truth surface.

### 5. Capacity / KV Budget Pass

Map external capacity concepts into ARLE knobs before proposing changes:

| External concept | ARLE question |
| --- | --- |
| weight load memory | model residency and quantized loader behavior |
| KV pool memory | `max_seq_len`, `num_slots`, KV dtype, paged KV block bytes |
| graph/capture memory | graph key count, captured shapes, warmup budget |
| framework overhead | scheduler/runtime buffers and allocator headroom |
| token capacity | request shape x slot budget, not just aggregate tokens |

Useful evidence:

- startup log before/after weight load and KV allocation;
- `/v1/stats` snapshots during steady state;
- `nvidia-smi` per-rank memory;
- model `config.json` for layers, heads, KV heads, hidden size, MoE/MLA/CSA
  fields;
- ARLE bench envelope logs.

Flag claims about KV replication, SWA/CSA/HCA compression, or prefix cache hit
rate as hypotheses until local logs or code paths confirm them.

### 6. Trace / Profiler Pass

Keep prefill, decode, and mixed continuous-batch windows separate. A hot decode
kernel does not explain TTFT unless wall-clock framing says it does.

Output three small tables when trace evidence exists:

| Table | Purpose |
| --- | --- |
| Kernel table | top kernels by stage and wall-clock share |
| Overlap table | CPU/GPU gaps, launch density, copies, syncs |
| Fuse table | source-backed candidates, not fuzzy matches |

For ARLE traces, prefer nsys/ncu wrappers and internal counters over
torch-profiler. Use torch-profiler guidance only for external SGLang/vLLM/
TensorRT-LLM traces or for comparing stage naming.

### 7. Output Contract

Return or write:

- upstream sources read, with repo URL and commit hash;
- kept/killed upstream candidates and why;
- ARLE modules affected or explicitly out of scope;
- hypotheses generated;
- local evidence still required to license or kill;
- exact next command or patch boundary.

If this scan produces a code/docs change, record the source scan in
`docs/experience/wins/` or `docs/experience/errors/` with kept/killed verdicts.

## References

- BBuf AI-Infra-Auto-Driven-SKILLS:
  <https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS>
- Model PR history:
  <https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/model-pr-optimization-history>
- LLM serving benchmark skill:
  <https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-serving-auto-benchmark>
- LLM capacity planner skill:
  <https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-serving-capacity-planner>
- LLM torch profiler skill:
  <https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-torch-profiler-analysis>
- Model compute simulation skill:
  <https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/model-compute-simulation>
