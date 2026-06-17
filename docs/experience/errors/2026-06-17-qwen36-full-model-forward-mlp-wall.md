# Qwen3.6 full-model forward wall is MoE MLP, not linear attention

## Context

After A12, the isolated real-checkpoint Qwen3.6 FP8 MoE backward gate was fast:
`MoeGroupedLinear` dropped to 0.198s and finite-diff stayed at `rel_err=3.170e-3`.
The full-model finite-diff gate, however, still stalled before printing any
backward profile. A bounded full-forward trace was added to avoid another blind
wait.

## Evidence

Remote `.62` command, GPU7, model `/data01/models/Qwen3.6-35B-A3B-FP8`:

```text
CUDA_VISIBLE_DEVICES=7 qwen36_fp8_lora_fd_gate \
  --model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --device 0 \
  --target-set all-linear \
  --target-adapter auto:routed-up \
  --mode full-model \
  --profile-forward-only
```

Trace excerpt:

```text
qwen35_full_forward_trace scope=attention layer=0 component=linear_core seconds=0.011358
qwen35_full_forward_trace scope=layer layer=0 component=mlp seconds=10.384718
qwen35_full_forward_trace scope=model component=layer_total seconds=10.491706
qwen35_full_forward_trace scope=attention layer=1 component=linear_core seconds=0.009830
qwen35_full_forward_trace scope=layer layer=1 component=mlp seconds=13.084257
qwen35_full_forward_trace scope=attention layer=2 component=linear_core seconds=0.008373
qwen35_full_forward_trace scope=layer layer=2 component=mlp seconds=13.090730
qwen35_full_forward_trace scope=attention layer=3 component=sdpa seconds=0.000222
qwen35_full_forward_trace scope=layer layer=3 component=mlp seconds=13.422737
qwen35_full_forward_trace scope=layer layer=4 component=mlp seconds=13.398635
qwen35_full_forward_trace scope=layer layer=5 component=mlp seconds=13.457988
```

The run was stopped once the repeated pattern was established. Attention and
linear-attention core are milliseconds; each sparse MLP forward is about
10-13s.

## Root Cause

The full-model wall is the train-side sparse MLP forward path. The active MoE
backward fixes did not change `moe_grouped_linear` forward: it still iterates
over every nominal expert and every `max_rows` slot, computing the public
`[experts, max_rows, dim]` tensor by host loops. With one token and top-8
routing, that means padding hundreds of inactive experts into every gate/up/down
forward.

## Fix

Apply the same active-expert principle to `moe_grouped_linear` forward: keep the
public output shape unchanged, but compute only route-active experts and leave
inactive expert rows zero.

## Rule

Do not assume the remaining full-model wall is the previous suspect. A bounded
trace can flip the target: here linear attention was milliseconds, while MoE
MLP forward was seconds per layer.
