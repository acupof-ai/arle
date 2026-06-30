# DSv4 batched O-LoRA route-GEMV swap killed

## Context

After extending the compact FP8 MoE decode lane, the TP4 phase profile still showed
`sw_attn.finish` around 6ms and the linear profile showed `dsv4/linear/wo_a` as the
largest O-LoRA projection bucket. The hypothesis was that TP4 multi-group `wo_a`
would be faster through the existing one-launch grouped `dsv4_fp8_route_gemv_batch`
path than through per-group gather -> DeepGEMM(m=n) -> scatter.

## Root Cause

The route-GEMV path is slower for this batched O-LoRA shape. It avoids per-group
DeepGEMM launches but pays scalar FP8 dequant/GEMV bandwidth and loses the tensor-core
path. Pod profile with the test commit (`f2b12b09`) showed `finish` regressing from
about 6.3ms to about 8.3ms and `proj` rising to about 5ms.

Representative bad lines:

```text
[decode-phase] n=2 sw_attn=29.9ms (prep=18.1 [proj=5.0 compidx=5.1 ...] fwd=2.2 finish=8.3) moe=21.0ms
[dsv4-linear-profile] dsv4/linear/wo_a calls=43 total_ms=4.67 avg_us=108.6
```

The test was reverted immediately in `64e7cc2f`; do not re-land this shape without a
new kernel or a direct measured counterexample.

## Fix

Keep the existing grouped DeepGEMM decode path for `wo_a` at B>1. The next finish
lever is not the existing route-GEMV fallback; it needs either fewer gather/scatter
launches around DeepGEMM or a purpose-built grouped tensor-core path.

## Rule

Do not replace tensor-core DeepGEMM with scalar route-GEMV just to reduce launch
count. For DSv4 O-LoRA, the bandwidth cost dominates and the route-GEMV shortcut is
slower despite being structurally simpler.
