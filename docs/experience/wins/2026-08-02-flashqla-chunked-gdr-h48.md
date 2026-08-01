# FlashQLA chunked GDR at H=48: 33K prefill −27% — 2026-08-02

> Status: Shipped (`778fef873` parameterization, `5b851d193` compile fixes).
> Flag stays **opt-in** (`--qwen35-gdr-chunked true`); default flip pending the
> needle ladder ×3.

## Context

The whole-step profile (2026-08-01) ranked the serial
`gated_delta_rule_prefill_recurrent` scan #1: 9.37 s of a 28.6 s 33K prefill
(33%), a 5.9 µs/token latency chain. The FlashQLA chunked path existed but was
dead twice over: shape-guarded to `v_heads == 32` (canonical 27B has 48), and
— decisive — **it had never compiled anywhere**: no `generated/` artifacts,
`fq_fwd` referenced an undefined `bhg`, and tilelang 0.1.11's TMA lowering
emits host-built `*_desc` wrapper params the AOT surface cannot construct.

## What Worked

Parameterize, don't re-bake: H/Hg became AOT instantiation parameters
(`kernels.toml` row triple per geometry, `q_heads`/`kv_heads` fields the
attention family already used; build.rs needed zero changes). Runtime
dispatches on `(local_linear_k_heads, local_linear_v_heads)` with recurrent
fallback — a new GDN geometry is 3 toml rows + 1 match arm. Compile fixes:
`bhg = bh // (H // Hg)` and `TL_DISABLE_TMA_LOWER` on kkt/fwd.

Measured (1× H20 GPU 6, ThinkingCap-Qwen3.6-27B-FP8, same binary, two
distinct 33K prompts, cold):

| arm | 33K cold prefill | Δ |
|---|---|---|
| recurrent (flag off) | 28.95 s / 28.64 s | baseline |
| chunked (flag on) | **21.63 s / 20.65 s** | **−26% / −28%** |

- Correctness: greedy-64 outputs byte-identical across arms.
- Path probe (nsys, flag-noop lesson): `fq_fwd` 1152 launches / 0.81 s,
  `fq_kkt` 1152 / 0.14 s, `gdr_fq_prep` 1152 / 0.10 s — **1.06 s total where
  the recurrent scan spent 9.37 s (−8.8×)**.
- New prefill #1: TileLang full attention (`kernel_kernel`), 3.99 s / 25% —
  backlog #3 (promote to FA3) is now the top prefill lever.

## Rule

**"Vendored + flagged" is not "compiled".** The path shipped with an FFI
surface, a Rust guard, and a CLI flag, and still had never survived codegen —
the only proof a kernel exists is its artifact in the build output. Check
`generated/` (or OUT_DIR) before costing any flagged path into a plan.
