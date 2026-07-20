# temp=1.0 long-gen degeneration on Qwen3.6-27B — resolved

> Status: Shipped/Resolved. The temp>0 "salad" is **temperature=1.0 + long
> generation degeneration**, uniform across ALL Qwen3.6-27B variants (base &
> ThinkingCap, FP8 & bf16) on the clean binary `fea8e1fd0`. NOT FP8, NOT a MoE
> router (there is none — hybrid linear-attn), NOT ThinkingCap weights, NOT norm,
> NOT config, NOT the sampler. Fix shipped: `--rollout-temperature 0.3`
> (`2394a2ab0`). Full forensics + the five-hypothesis false-chain in the errors
> entry.

## Verdict

Confirmed on clean binary by a controlled A/B (3 models × {greedy, temp=1.0} ×
{400, 2000 tok}): base and ThinkingCap, FP8 and bf16, are **indistinguishable** —
all coherent at greedy/top_k=1/short, all degenerate at temp=1.0 + length. The
model ships no `repetition_penalty`; long unconstrained temp=1.0 sampling loops.
temp=0.3 (already the rollout default) is the correct operating point.

Every static hypothesis was killed by measurement — see the errors entry:
- MoE router quantized → no routers (hybrid linear-attn); loose grep artifact.
- FP8 scales/values → scales bit-identical, dequant error 2.65% intrinsic floor.
- sampling/rope/template/eos config → identical to base.
- norm handling (`9851ced6b`) → a separate mis-fix, reverted `485eefe0d`.
- the driving premise "base coherent / ThinkingCap salad" → artifact of a
  pre-norm-revert binary; does not reproduce clean.

## Follow-ups

- **Sampler exonerated (closed):** host sampler truncates top_k then cuts top_p at
  first cum≥0.95 — the drawn token is always in-nucleus by construction; control
  (temp=1.0 top_k=1 coherent, top_k=20 garbage) confirms the filter is live. The
  temp=1.0 garbage token is genuinely in the model's top-20 tail — model behavior,
  not a leak. Fix stays temp=0.3.
- **Voided:** #55 router bf16 re-export (no-op — no routers); FP8 requant (FP8
  faithful); bf16 swap (bf16 salads identically).
- **OPD:** ThinkingCap-FP8 student unblocked at temp=0.3 — resume the P4 lane.

## Links

- Errors entry (full forensics + rules):
  [errors/2026-07-20-hd256-fp8-temp-sampling-corruption.md](../experience/errors/2026-07-20-hd256-fp8-temp-sampling-corruption.md).
- Shipped fix: `2394a2ab0` (`--rollout-temperature` 1.0→0.3).
- hd256 RMSNorm greedy fix (earlier, valid): `b4b293f0c`. Norm mis-fix reverted:
  `485eefe0d`. Relay/root task: #48.
