# DSv4 EAGLE DeepEP Debug Lane Kill

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

The batched internal MTP/EAGLE verifier and DSv4 body graph improved the
debug-fallback all-reduce lane, but that lane is not the target. Same-binary
remote validation at commit `633c131b` kept the target confounder explicit:

- `ARLE_DSV4_MOE_BACKEND=allreduce`, c8/max_tokens=8:
  `ANSWER_PASS`; wall 4.83-6.01 s for 8 output tokens.
- `ARLE_DSV4_MOE_BACKEND=deepep`, same c8/max_tokens=8 before the guard:
  HTTP 200, 8 output tokens, but every row failed the `ZZZ406ZZZ` sentinel and
  several rows produced repeated garbage tokens.
- `ARLE_DSV4_PERFORMANCE_PROFILE=sglang` + `native-deepep` still fails closed at
  startup because the current executable route has replicated-token ownership,
  piecewise graph decode, missing DeepEP/NCCL graph replay, eager MTP draft,
  and missing FlashMLA/SWA/C4/C128 metadata replay.

Artifacts:

- all-reduce same-binary control:
  `/tmp/dsv4_spec_batch_verify_32de2c85/c8_tok8_allreduce`
- DeepEP-style bad-output run:
  `/tmp/dsv4_spec_batch_verify_14418267/c8_tok8_deepep`
- high-performance profile contract probe:
  `/tmp/dsv4_profile_probe_serial_1780444980`

## Root Cause

`ARLE_DSV4_MOE_BACKEND=deepep` is a debug dispatch/combine lane running on the
replicated-token TP/EP route. It is not the token-owned native DeepEP path that
the DSv4-Flash TP8 + EAGLE target needs.

The first MTP failure was a real missing scratch bug:

`ARLE_DSV4_EXPERT_BACKEND=deepgemm requires DeepSeek V4 MoE runtime scratch`

After `0d99f964` gave the MTP draft its own MoE runtime scratch, the next
blocker was the batched body path trying to use graph-safe local all-reduce FFN
with DeepEP enabled:

`DeepSeek V4 graph-safe FFN path is only wired for local-routed all-reduce`

After `32de2c85` routed DeepEP body FFN through the eager path, the request no
longer failed with HTTP 500, but it produced wrong tokens. The matched
all-reduce control passed on the same binary and prompt, so the failure is
isolated to the DeepEP-style dispatch lane, not the verifier script.

## Fix

- `0d99f964`: added per-request MTP MoE scratch and passed it into the MTP FFN
  route.
- `32de2c85`: avoided graph-safe local-all-reduce FFN when DeepEP is enabled in
  the batched body fallback.
- `633c131b`: killed the unsafe combination by making internal MTP frozen-KV
  draft fail closed for both native DeepEP and DeepEP-style dispatch on the
  replicated-token TP/EP route.

Remote verification for `633c131b`:

- release-fast DSv4 build passed via the prebuilt CUDA fast path:
  `/tmp/dsv4_mtp_deepep_failclosed_633c131b_build.log`.
- same DeepEP-style c8/max_tokens=8 script now returns HTTP 500 before emitting
  completion tokens.
- server log first failure:
  `DSv4 internal MTP frozen-KV draft does not support DeepEP-style dispatch on replicated-token TP/EP yet`.
- after cleanup, `nvidia-smi --query-compute-apps` reported no compute apps.

## Rule

Do not count debug-fallback all-reduce or DeepEP-style dispatch as progress
toward the DSv4-Flash TP8 + EAGLE target. The target path is token-owned
native DeepEP + full-decode graph-safe replay. Until that startup contract is
satisfied and the 256K/1500 hot-cache workload clears TTFT, TPOT, E2E, and
output throughput together, the result is not a performance win.
