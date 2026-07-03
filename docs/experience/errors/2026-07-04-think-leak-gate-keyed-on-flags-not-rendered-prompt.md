# The </think> leak "class-level fix" shipped green but never engaged: the gate read flags, the truth lived in the rendered prompt

## Context

9fdc4e57 rebuilt reasoning-before-tool-parsing on all four paths
(`split_reasoning` pre-split + `StreamPipeline`), with a five-think-forms ×
two-APIs unit matrix, all green — and declared the bare-`</think>` leak fixed.
Same-day pod e2e (Qwen3.6-27B-FP8, GPU 1, `/host/cc_leak_smoke/`) reproduced
the leak on 4/9 default-request probes: `text` blocks ending in
`…✅\n</think>` next to a correctly parsed `tool_use`, reasoning prose in
visible content on both APIs.

## Root Cause

The split is gated per-request on
`enable_thinking(max_thinking_tokens > 0 || template_defaults_thinking)` —
server flags plus a per-model allowlist (`defaults_thinking_on()` = DSv4
only). But thinking-mode truth lives in the checkpoint's Jinja template:
Qwen3.6 defaults `enable_thinking` ON and prefills `<think>` into the prompt,
so output arrives as `reasoning</think>answer` while the gate computes
`thinking=false` → `split_reasoning` is a byte-identical passthrough.
`/v1/messages` additionally hardcodes `chat_template_kwargs: None`, so the CC
workload could never open the gate at all. The unit matrix passed because it
*stubbed* `thinking=true` instead of deriving it the way prod does — the
tests exercised the split, never the gate. Decisive single-variable A/B on
the pod: same body + `chat_template_kwargs {"enable_thinking": true}` →
clean; default → leak.

## Fix

0f463f15 — key the parse-side gate off the rendered prompt: after every
render, `thinking |= prompt_prefills_think(&prompt)` (rendered prompt
`trim_end`-ends with an open `<think>`). Self-consistent with the template:
explicit `enable_thinking=false` renders a closed block (or none) and keeps
the gate off. Applied at all three render sites (chat, multimodal,
`anthropic_prompt`), covering both APIs, stream + non-stream. Companion
parser fix: a `<tool_call>` opener without a routable payload no longer
fabricates an empty-name `tool_use` or swallows the tail (nameless payload /
non-adjacent opener = prose, kept verbatim; adjacent-`{`-unbalanced = real
truncated call, still dropped).

Pod re-verify (same probes, `v2_*` + `CHECK_RESULTS_V2.txt`): 11/13 PASS —
both v1 leak sites clean, OpenAI no-tools now populates `reasoning_content`
(non-stream len=2992; stream 1047 deltas), CC smoke exit 0 with the leaked
narration gone (`result` = answer only), serve log 0 ERROR/WARN. The 2 FAILs
are attributed, not regressions: (c)-stream = model indecision at a 512
budget (forced `tool_choice` proves the SSE tool path); (e)-2048 = the
**accepted residue class** — output that *quotes* the protocol tags
(`</think>` example inside reasoning splits at first occurrence, mirroring
vLLM/SGLang semantics; example tool calls carrying real defined-tool names
parse as calls — no cheap discriminator exists).

## Rule

A fix is not verified until the *gate that routes into it* is proven to fire
under the target workload's default config — unit-test the derivation, not a
stubbed switch, then e2e on the real serve before declaring victory. For
template-driven behavior (thinking, tool format), the rendered prompt is the
only ground truth; request/server flags are hypotheses about it.
