# P0 line-level spec — chunked GDR backward (adopt FLA, kill the reverse-division hand-roll)

**Date**: 2026-06-18  **Author**: Claude (integration/spec).
**Source-grounded in**: FLA `/tmp/fla/fla/ops/gated_delta_rule/chunk.py`
(`chunk_gated_delta_rule_bwd`, lines 120-238) + our forward AOT
`crates/cuda-kernels/tools/tilelang/gated_delta_rule.py` (7 stages, read in full).
Supersedes [`2026-06-18-gated-delta-kernel-and-org-gap.md`](../research/2026-06-18-gated-delta-kernel-and-org-gap.md).

**Execution: a non-tmux2 pane after approval + H20-on. No code yet.**

## ⚠️ DEMOTED (2026-06-18, adversarial review wf_6a493801)

**P0 is no longer a correctness fix — re-framed to memory+perf, deprioritized.**
The committed kernel (HEAD `5b26db30`) is numerically **correct** (`state_history` +
multiply-by-`exp_g`, no division). The reverse-division "bug" was an uncommitted WIP
(now deleted). So the committed gated-delta backward works today; the FLA chunked
backward is a **memory** (drop `state_history` `[B,T,H,Dk,Dv]` for chunk checkpoints)
**+ perf** (tensor cores) improvement that needs a **license-or-kill A/B on H20** —
and the measured OPD backward wall is MoE grouped-linear, not this kernel, so benefit
is unmeasured. **Higher-value, locally-progressable work goes first** (sparse top-k
teacher targets `2026-06-18-sparse-topk-teacher-targets.md`, P1 beta-JSD, P2 hardening).
The invariant below is still the right design for the eventual chunked backward; just
not urgent. P0 working-tree WIP left as a correct-but-slow CPU-fallback interim.

## Correctness invariant (the whole point)

The hand-roll reconstructs prior state by **reverse-dividing the gate**
(`state[t-1]=(state[t]−k⊗δ)/exp_g`, `exp_g<1e-8→0` → wrong gradient in the
hard-forget regime). FLA's backward NEVER divides by the gate: it **recomputes
the forward chunk states by replaying `chunk_gated_delta_rule_fwd_h` from
`initial_state`** (chunk.py:153-163), then runs a backward scan over those
recomputed states. **Every backward stage must consume recomputed/saved forward
intermediates — no reverse-recurrence.**

## Forward AOT recap (already in-repo, `gated_delta_rule.py`)

7 stages already produce the backward's inputs. Saved/derivable tensors:
`q,k,v` (post-l2norm q/k from `gdr_chunk_prepare`), `g`=`g_cumsum`
(`gdr_chunk_cumsum`), `beta`, `A`=`a_inv` (the WY inverse from `gdr_chunk_a`+
`gdr_chunk_solve`), `w,u` (`gdr_chunk_recompute`), `chunk_state` (per-chunk
boundary h `[num_chunks,hv,K,V]`) + `v_new` (`gdr_chunk_state`).
FLA saves only `{q,k,v,g,beta,A,initial_state}` and **recomputes** w/u/h/v_new in
bwd — we follow FLA (recompute) so the forward stops saving `state_history`.

## Backward stage DAG (port target = FLA `chunk_gated_delta_rule_bwd`)

| # | New TileLang AOT stage | FLA source (`/tmp/fla/...`) | consumes | produces |
|---|---|---|---|---|
| B0 | (reuse fwd) recompute w,u | `wy_fast.py::recompute_w_u_fwd` (= our `gdr_chunk_recompute`) | k,v,beta,A,g | w,u |
| B1 | (reuse fwd) recompute h,v_new | `common/chunk_delta_h.py::chunk_gated_delta_rule_fwd_h` (= our `gdr_chunk_state`) | k,w,u,g,initial_state | h,v_new |
| B2 | `gdr_bwd_dv_local` | `common/chunk_o.py::chunk_bwd_dv_local` | q,k,g,do,scale | dv (local) |
| B3 | `gdr_bwd_dhu` **(stability crux)** | `common/chunk_delta_h.py::chunk_gated_delta_rule_bwd_dhu` | q,k,w,g,h0,dht,do,dv | dh,dh0,dv |
| B4 | `gdr_bwd_dqkwg` | `common/chunk_o.py::chunk_bwd_dqkwg` | q,k,v_new,w,g,h,dv,do,dh | dq,dk,dw,dg |
| B5 | `gdr_bwd_wy` | `wy_fast.py::prepare_wy_repr_bwd` | k,v,beta,g,A,dw,du=dv | dk2,dv,db,dg2 |
| B6 | accumulate | chunk.py:232-233 | dk+=dk2, dg+=dg2 | dk,dg |
| B7 | `gdr_bwd_dg_revcumsum` | `chunk_local_cumsum(reverse=True)` chunk.py:234 | dg | dg |
| B8 | `gdr_gate_bwd` | `gated_delta_rule/gate.py::gdn_gate_bwd` | g_input,A_log,dt_bias,dg | dg,dA_log,ddt_bias |
| B9 | elementwise bwd | `modules/l2norm.py::l2norm_bwd` (dq,dk), `gate.py::fused_beta_sigmoid_bwd` (db) | q/k_rstd, db | dq,dk,db |

Outputs back to autograd: `dq,dk,dv,db(=dbeta),dg,dh0,dA_log,ddt_bias`.
(Map to the hand-roll's FFI outputs: dqkv = scatter of dq/dk/dv into the packed
`[q|k|v]` layout; da_log=dA_log; ddt=ddt_bias; db=dbeta; dnorm via B9.)

## Autograd wiring

- `crates/autograd/src/backend_cuda.rs::cuda_linear_attention_scan_backward`:
  replace the single hand-rolled `linear_attention_scan_backward_f32` launch
  with the B0-B9 AOT-stage sequence (mirrors the forward dispatch already in
  `infer/src/ops/recurrent.rs` for the 7 fwd stages). Drop the
  `state_current`/`grad_state_scratch`/reverse-division args.
- `crates/autograd/src/ops/linear_attention.rs`: forward stops saving
  `state_history` (already WIP-removed); the `LinearAttentionForward` struct
  keeps only `{q,k,v,g,beta,A,initial_state}` + final_state. The CPU forward path
  keeps an **exact** `state_history` replay ONLY under a test/oracle cfg (the
  gradient-check reference).
- **Open Q (decide before code)**: B0-B9 live in `cuda-kernels` (the canonical
  GDR family); autograd's CUDA backend FFIs into them → adds an
  `autograd → cuda-kernels` build edge. Confirm that direction vs. duplicating
  the AOT entry points.

## Delete

`crates/autograd/src/backend_cuda/kernels/linear_attention.cu` (the 424-line
reverse-division scan) + the WIP `recover_previous_state` in
`ops/linear_attention.rs`. Resolves the 4-file WIP.

## Verification gate (needs H20 sm_90a — defer to approval)

1. **Gradient-check** B0-B9 vs the exact CPU `state_history` replay reference,
   per-output relative error (dq,dk,dv,dbeta,dg,dA_log,ddt). **MUST include the
   hard-forget regime** (drive `a_log`/`dt_bias` so `exp_g→1e-9`) — that is
   exactly where the hand-roll silently zeroed.
2. TileLang AOT builds the B-stages for sm_90a; reuse the forward AOT's
   `gen_tilelang_aot.py --kernel-family gdr` path (add bwd kernel keys).
3. Then (separate, perf license): OPD step wall-clock A/B vs the hand-roll.

## Why this is not 闭门造车

B0/B1 reuse our already-licensed forward stages; B2-B9 are 1:1 ports of named
FLA functions (cited above), the same library vLLM+SGLang ship for Qwen3-Next.
The TileLang substrate is the one we already use for the forward.
