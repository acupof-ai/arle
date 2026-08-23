# NVFP4 serving corrupts tool-call generation

> Status: Open — mechanism unknown

## Phenomenon

`ThinkingCap-Qwen3.6-27B-NVFP4` is wrong on a subset of prompts where
`ThinkingCap-Qwen3.6-27B-FP8` is right. Matched probes, identical request
bodies, same binary, same box, `max_tokens=1500` on both:

| probe | FP8 | NVFP4 |
|---|---|---|
| tool defs + agent prompt | `tool_use Read{file_path:"textfsm/terminal.py"}` | token soup |
| prose, 51 input tokens | correct | wrong, deterministically |
| structured JSON, 80 input tokens | correct | correct |
| plain prose 100-1200 tokens, trivial output | — | coherent |

The tool-call output contains strings that are not words, and repeats
byte-identically across runs:

```
<tool_call> <endfaclettothxink> ```<toolhas_n </think>
<tool_call> <function=readme.txt> </antmltext/terminal.py</tooldi> </think>
<tool_call> <endthreadindex_to> </thinking>
```

One parse surfaced a `tool_use` block named `readme.txt` with a mangled input
object, so the damage reaches the client as a well-formed block carrying
nonsense.

The prose failure, at 51 input tokens, substitutes plausible names for the
literal ones given:

> The function `strip_ansi_escapes` in `textfsm/ansi.py` needs fixing...

The prompt named `StripAnsiText` in `textfsm/terminal.py`.

## Impact

Every agent-OPD rollout on the NVFP4 base returns `edited=false`: 68 rollouts
across two configurations, one edit total. The run looks like a capability
result and is not one.

## What it is not

- Not the prefill arm. Both corrupt, in different ways: m<512 (Marlin) gives
  total token soup, m>=512 (DeepGEMM, verified by padding the tool prompt to
  1643 input tokens) gives coherent structure with rotten identifiers
  (`textfrig/terminal.py`).
- Not prompt length. 51 tokens fails; 1200 tokens of plain prose passes.
- Not structured output. The JSON probe passes on both.
- Not tool definitions alone. The failing prose probe carries none.
- Not the sampling path. `tools_active` in `infer-server/src/coordinator.rs`
  selects prompt rendering and post-parse only; no sampling, stop, or logit
  setting.
- Not the harness or the server. FP8 answers every one of these bodies
  correctly through the same code.
- Not the repack's flush-to-zero on special-token rows. `embed_tokens` and
  `lm_head` are BF16 in the checkpoint; `repack_for_marlin_fp4` only touches
  `WeightFormat::Fp4E2M1Group`. (Falsified by the `qwen3-nvfp4-support`
  session.)
- Not the static weight chain at all. A 4-agent workflow (2026-08-23) verified
  on the actual checkpoint: repack layout bit-exact (full-chain nibbles ==
  checkpoint nibbles on gate_proj, q_proj, MTP gate_proj), scale encoding
  bit-exact roundtrip with **zero flushes on all 263 NVFP4 tensors**
  (0/5,570,560 on gate_proj), sfb algebra correct, global fold correct. The
  repack is lossless on this checkpoint.
- Not the checkpoint weights. NVFP4 vs FP8 dequantized weights: cosine 0.985,
  max abs diff 0.18 on a ±0.43 range (layer 0 gate_proj). Normal 4-bit vs 8-bit
  quantization error, not corruption.

## A trap worth naming

An intermediate reading of this bug — "NVFP4 hallucinates on structured
output" — was an artifact of `max_tokens=250`. This model's thinking preamble
consumes the budget, and the truncated result reads exactly like corruption.
Every probe here was re-run at 1500 before being believed. Budget the thinking
before calling an output damaged.

## Cause

Unknown. The static weight chain is verified clean (repack, scales, sfb, global
fold — all bit-exact on the actual checkpoint). The checkpoint weights are a
reasonable 4-bit quantization (cosine 0.985 vs FP8). The kernel's algebra is
correct on paper (2^-126 × 128 × 2^119 = 2^0, global applied once).

What remains untested: the Marlin fp4 kernel's **runtime output** on real
weights at the model's actual shapes. The existing tests
(`test_cuda_marlin_fp4_share.rs`) compare two GPU code paths against each other
at 128×128 / 256×512 with synthetic weights — never against a CPU ground truth
at 17408×5120 / 34816×5120. A correctness probe
(`crates/infer-cuda/examples/marlin_fp4_correctness.rs`) is built to close this
gap: it runs `marlin_fp4_gemm` at the model's shapes with realistic E2M1/E4M3
distributions and compares against a CPU reference.

No single variable separates the passing prompts from the failing ones. The
corruption is content-dependent (tools field, 51-token prose) but the weight
chain is content-independent — weights are loaded/repacked once and process
every token identically. This narrows the suspect to either a kernel runtime
bug that manifests on certain weight patterns, or something in the engine's
forward path that is shared with FP8 but interacts differently with the NVFP4
weight distribution.

Handed to the `qwen3-nvfp4-support` session.

## Consequence for open results

The NVFP4-vs-FP8 rubric-opd loss gap recorded on 2026-08-22 (0.3363 against
0.6414 on matched greedy rollouts) was attributed to the Marlin repack's
flush-to-zero being lossy but correct. That attribution is **dead**: the repack
is lossless on this checkpoint (zero flushes, bit-exact scale roundtrip on all
263 tensors). The gap needs a new explanation or a rerun.

## Rule

Two quantizations of one checkpoint are two models until a matched probe says
otherwise. Run the cheap one — same request body, both ports — before reading
any downstream number as a property of the workload.
