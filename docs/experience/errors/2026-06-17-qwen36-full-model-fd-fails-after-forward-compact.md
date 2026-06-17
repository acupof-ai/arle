# Qwen3.6 full-model finite diff fails after forward compact

## Context

A14 removed the proven sparse-MLP forward padding wall. The isolated
`mlp-layer` finite-diff gate still passes, and full-model forward now completes
in about 43.5s instead of repeating 10-13s sparse-MLP layers.

This entry records the required control: the full-model finite-diff gate is
still not licensed.

## Evidence

Remote `.62` command, GPU2, model `/data01/models/Qwen3.6-35B-A3B-FP8`:

```text
CUDA_VISIBLE_DEVICES=2 qwen36_fp8_lora_fd_gate \
  --model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --device 0 \
  --target-set all-linear \
  --target-adapter auto:routed-up \
  --mode full-model \
  --layer 0 \
  --eps 1e-3 \
  --profile-backward
```

The gate reached a profiled backward and then failed the finite-diff check:

```text
qwen36_fp8_lora_fd_backward_profile total_seconds=52.195867 op_seconds=50.020469 prelude_seconds=0.005423 merge_grad_seconds=2.153578 op_kinds=18 site_kinds=1167
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MatmulBT count=1167 seconds=25.700711 pct_total=49.239
qwen36_fp8_lora_fd_backward_profile_op rank=2 op=MoeGroupedLinear count=120 seconds=23.624968 pct_total=45.262
qwen36_fp8_lora_fd_backward_profile_op rank=3 op=LinearAttention count=30 seconds=0.263393 pct_total=0.505
qwen36_fp8_lora_fd_backward_profile_site rank=1 op=MatmulBT site=model.language_model.layers.8.linear_attn.in_proj_qkv.weight count=1 seconds=0.322067 pct_total=0.617
qwen36_fp8_lora_fd_backward_profile_site rank=2 op=MatmulBT site=model.language_model.layers.23.self_attn.q_proj.weight count=1 seconds=0.321708 pct_total=0.616
qwen36_fp8_lora_fd_backward_profile_site rank=3 op=MatmulBT site=model.language_model.layers.39.self_attn.q_proj.weight count=1 seconds=0.318333 pct_total=0.610
qwen36_fp8_lora_fd_gate_result load_seconds=13.816454 analytic_seconds=56.723278 plus_seconds=4.646927 minus_seconds=5.142230 live_host_mib=17271.8 mode=full-model layer=0 target=model.language_model.layers.0.mlp.experts.179.up_proj.weight.lora_b index=508 eps=1.0e-3 loss_base=-1.285077477e1 loss_minus=-1.245186043e1 loss_plus=-1.292645359e1 analytic=3.636742830e-1 numeric=-2.372965698e2 rel_err=1.002e0
Error: Qwen3.6 FP8 LoRA finite diff failed
```

## Root Cause

Not fully rooted yet. The measured facts are:

- The previous pre-backward full-forward wall is gone.
- Full-model backward now spends 49.2% in ordinary `MatmulBT` sites and 45.3%
  in `MoeGroupedLinear`.
- The finite-diff failure is not a tolerance miss: numeric `-2.37e2` versus
  analytic `3.64e-1`.

Two hypotheses are plausible and must be licensed or killed before the next
fix:

1. Full-model finite diff is crossing discrete MoE route boundaries in later
   layers, so the scalar full-logit loss is not a smooth local function for
   central diff at `eps=1e-3`.
2. The full-model backward is still doing unnecessary projection-weight
   `MatmulBT` work for frozen base tensors because input gradients or
   parameter ownership are too broad under all-linear QLoRA.

The profile supports both as investigation targets, but proves neither as root
cause.

## Fix

Do not claim full-model gradient license from the A14 MLP gate. Next steps:

1. Add a route-stability probe around the full-model plus/minus passes to count
   whether layer/expert top-k routes change for the target perturbation.
2. Audit the `MatmulBT` ownership path for base `.weight` sites in full-model
   all-linear QLoRA and skip frozen-base input-gradient work when the upstream
   path does not require it.
3. Re-run a full-model finite-diff gate only after the root cause is measured.

## Rule

A faster micro-gate is not a full-model gradient license. If the full-model
central diff fails by orders of magnitude, record it as unlicensed and root the
smoothness/ownership issue before moving to a broader OPD training claim.
