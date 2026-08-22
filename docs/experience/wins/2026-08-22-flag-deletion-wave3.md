# Flag deletion wave 3 — spec gates, FlashMLA decode lever, reuse shim

2026-08-22 · CUDA · refactor

## Context

Wave 2 (see `2026-08-22-flag-deletion-wave2.md`) left four proven-dead levers
plus one hardcoded shim on the CUDA path. Each still had a CLI flag, a seam
field, a runtime-flag static, and read sites — the full A/B surface for
experiments that had already concluded.

## What worked

Deleted with the full chain (flag def → seam field → static/accessor → read
sites → doc rows), hardcoding the winner:

| Flag | Verdict | Replacement |
|------|---------|-------------|
| `--dspark-confidence-threshold` | Never shipped enabled; unset was the only shipped config. The `<= 0` short-circuit existed for DeepSpec-paper-style drafter evaluation, not serving. | `DsparkSps` drops the field; `dspark_verify_lens` always runs the goodput budget. |
| `--mtp-adaptive` + `--mtp-min-accept` | Opt-in B=1 adaptive gate; never defaulted on. The EMA machinery (`mtp_should_speculate`, `mtp_note_accept`, `mtp_adaptive_skip`, probe interval) was a half-built product: executor-global EMA across slots, with a "make per-slot before defaulting on" comment that never happened. | Machinery deleted. `forward_mtp_warm_step` stays (desync re-seed path at `executor/dsv4.rs:386`). `mtp_accepts`/`mtp_rejects`/`mtp_chains` stats stay. |
| `--dsv4-flashmla-decode` | Scalar fallback was the A/B arm; FlashMLA won. Flag also fed a public override API (`set_dsv4_flashmla_decode_override`) consumed only by the resident A/B example. | `dsv4_flashmla_decode_enabled()` is now unconditional on `cuda_kernels::HAS_FLASHMLA`. Override static, setter, public re-export, and the example's FlashMLA axis deleted with it — a public A/B knob whose scalar arm silently ran FlashMLA would have been a half-state. |
| `dsv4_decode_reuse_enabled()` shim | Hardcoded `true` since wave 1; its only call site (`executor/dsv4.rs:441`) was a `!true` dead block. | Block and shim deleted. The finish-path publisher (`executor/dsv4.rs:742`) is the live path. |

Also cleaned: two stale cuda-kernels comments naming the deleted flag, and the
`lever_gate.sh` usage example that passed `--dsv4-flashmla-decode false`.

Serve flags: 53 → 49.

## Follow-up: the resident A/B example and the fused-WQKV override chain

The wave-3 edit agent flagged `examples/dsv4_resident_ab.rs` as a deletion
candidate: its reason-for-existing was scalar-vs-FlashMLA, and after wave 3 it
only A/Bed the fused-WQKV axis — itself a proven winner (default ON, +18.4 %
token-exact, TP=8/EP=8 pod, runtime `has_deepgemm_native()` preflight with
scalar fallback). Deleted with the full chain:

- `examples/dsv4_resident_ab.rs` (601 lines; no script or CI reference).
- `set_dsv4_fused_wqkv_decode_override` public re-export, the attention setter
  + `DSV4_FUSED_WQKV_DECODE_OVERRIDE` static, and the now-orphan
  `DSV4_FLASHMLA_OVERRIDE_*` constants (wave 3 had left them with one user).
- `dsv4_fused_wqkv_decode_enabled()` is now unconditional on the preflight,
  same shape as the FlashMLA deletion.
- A dangling doc comment + duplicate `#[cfg]` in `lib.rs` for the long-deleted
  `set_dsv4_moe_contig_decode` — it had re-attached to
  `set_qwen35_moe_experts_bf16_resident`, polluting its rustdoc.

The stage-profile APIs (`print_dsv4_stage_profile` et al.) stay —
`examples/dsv4_parity.rs` still uses them.

## Verification

- `cargo check -p cli --features cpu,no-cuda` clean
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --features cuda,no-cuda,nccl,deepep` clean
- `cargo check -p infer-cuda --example dsv4_resident_ab --features cuda,no-cuda,nccl` clean (example not covered by `--lib`)
- CUDA-lane clippy (cli + infer-api, `-D warnings`) clean
- Metal-lane `cargo check -p cli --features metal,no-cuda` clean
- `cargo test -p arle --profile release-fast --features cpu,no-cuda,cli`: 5 passed, 0 failed
- Repo-wide grep for all deleted symbols: 0 hits outside historical docs
- Follow-up: CUDA-lane clippy clean after the override-chain deletion;
  `cargo check -p infer-cuda --example dsv4_parity` clean (the surviving
  example's profile-API users intact).

## Rule

Deleting a flag that fed a public override API means deleting the override
chain with it — setter, static, re-export, and every consumer. Leaving the
setter orphaned is a half-state: a knob that looks live but whose arms
converged.
