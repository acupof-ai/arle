# NVFP4 serving corrupts tool-call generation

> Status: Open — mechanism unknown

## Phenomenon

`ThinkingCap-Qwen3.6-27B-NVFP4` produces malformed output on any request that
carries tool definitions. The same request against
`ThinkingCap-Qwen3.6-27B-FP8` is clean.

Matched probe, identical JSON body, same binary, same box:

| request | NVFP4 | FP8 |
|---|---|---|
| plain text, 100–1200 tokens | coherent | — |
| one tool, one-line instruction | empty text, `end_turn` / `max_tokens` | — |
| agent prompt + 2 tools | token soup | `tool_use` `Read{file_path:"textfsm/terminal.py"}` |

The corruption is deterministic — two runs returned the identical string —
and includes tokens that are not words:

```
<tool_call> = </think>  <tool_call> ="c-plain{cat,textfsm/terminal.py" </think>
<tool_call> <endfaclettothxink> ```<toolhas_n </think> ... <dimwit_to_replace>
</toolkit_content_placeholder>
```

One parse surfaced a `tool_use` block named `readme.txt` with a mangled input
object, so the damage reaches the client as a well-formed block carrying
nonsense.

## Impact

Every agent-OPD rollout on the NVFP4 base returns `edited=false`: 68 rollouts
across two configurations, one edit total. The run looks like a capability
result and is not one.

## What it is not

- Not the DeepGEMM prefill arm. That route engages at `m >= 512`
  (`QWEN_FP4_DEEPGEMM_MIN_M`); a prompt-length sweep at 100/300/480/520/700/1200
  tokens returns coherent text at every point, with no cliff at the threshold.
- Not the harness. Claude Code completes turns cleanly against the same serve
  (`terminal_reason=completed`), and the FP8 serve answers the same prompt
  correctly.
- Not prompt length. The one-line tool prompt fails too.

- Not the schema content, and not quantization damage in general. The same tool
  schemas delivered as plain user text — no `tools` field — get a coherent,
  correct answer from NVFP4: it names `textfsm/terminal.py`, the
  `StripAnsiText` function, and the argument-order fix. Only the `tools` field
  form corrupts.

## Cause

Unknown. What is left after the exclusions above is the token sequence the chat
template produces when it renders tool definitions — and FP8 handles that same
sequence correctly, so it takes the rendered prompt AND the NVFP4 weights
together.

One hypothesis, untested: the tool template introduces special tokens
(`<tools>`, `<tool_call>`). If the repack's flush-to-zero lands hard on the
embedding or lm_head rows for those ids, the shape matches exactly — ordinary
text fine, garbage once special tokens enter. Two cheap checks would settle it:
compare the flushed fraction of those rows against an average row, or run the
probe with the repack disabled and see whether the corruption goes away. The
layout math itself is bit-exact against `marlin_fp4_gemm` on synthetic weights
that do flush (`test_cuda_marlin_fp4_share.rs`), so if the repack is implicated
it is about which rows it damages, not the index math.

Handed to the `qwen3-nvfp4-support` session.

## Consequence for open results

The NVFP4-vs-FP8 rubric-opd loss gap recorded on 2026-08-22 (0.3363 against
0.6414 on matched greedy rollouts) was attributed to the Marlin repack's
flush-to-zero being lossy but correct. That attribution is no longer safe while
the NVFP4 path is known to damage generation. It needs re-examining.

## Rule

Two quantizations of one checkpoint are two models until a matched probe says
otherwise. Run the cheap one — same request body, both ports — before reading
any downstream number as a property of the workload.
