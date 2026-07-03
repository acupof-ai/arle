# DSv4 mhc_params: first-warp-parallel tail + fused params|pre_rms_norm (#143)

> Status: pending-remote — H20 build + needle gate + matched A/B ride the next pod run.

## Goal
Remove the `threadIdx.x==0` serial tail of `dsv4_mhc_params_kernel` (sigmoids +
hc_mult² sinkhorn on 1 of 1024 threads, replayed 122×/token = 9.5% of decode
GPU-busy) and one launch boundary per hc site. 6ms/token plan follow-on to G1.

## Hypothesis
(a) Lane-per-element/row/col tail on the first warp (`__syncwarp` phases,
per-element math order preserved) removes the serialization; (b) the three
decode-graph sites launch `params` and `pre_rms_norm` strictly back-to-back
with `pre` as the only coupling — fusing them drops 1 of 2 launches and the
`pre` global round-trip (staged in shared instead).

## Params
- `dsv4_mhc_params_tail` device fn (warp-0-only, syncwarp discipline: standalone
  kernel retires warps>0 after the last block sync; fused kernel keeps all
  threads alive and parks warps>0 at a `__syncthreads`).
- New `dsv4_mhc_params_pre_rms_norm_cuda` fused entry; the three decode-graph
  pair sites in dsv4.rs switched; eager prefill stays unfused (not the 9.5%
  target; still gets the tail speedup via the shared kernel).
- `pre`/`post`/`comb` still written to global for `hc_post` and A/B/debug.
- 1024-thread launch kept (measured −48% for the rms read at that width).

## Env
pending-remote (8×H20, TP=4/EP=4, DSv4-Flash-FP8).

## Results
pending-remote: needle gate ×3 + same-shell binary-pair A/B + nsys
`dsv4_mhc_params*` share re-measure.

## Problems
none yet.

## Learnings
pending-remote.
