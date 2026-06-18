# DSv4 config mechanism — classify every lever, then deletion-refactor

Status: **design** (2026-06-13). Systematic classification of every DSv4 runtime
"lever" into the RIGHT mechanism, then an incisive deletion-refactor. Grounded in
code-level research of our build system (`crates/cuda-kernels/build.rs`) + SGLang's
taxonomy (`/sgl-workspace/sglang`, verified, not inferred).

## Principle (4 mechanisms — each lever belongs to exactly one)

Per ckl: "环境变量顾名思义就是环境的 — log 等级、实验控制变量用 env；依赖编译的走编译"
— matched 1:1 by SGLang's own pattern (verified):

| Mechanism | What belongs here | SGLang precedent | Our mechanism |
|---|---|---|---|
| **Compile capability** | "Is this kernel actually built for this SM?" | `is_flashinfer_available()` guards (`flashinfer_backend.py:46`) | `build.rs` emits `cargo:rustc-cfg=…` from the enable-flags it already computes; runtime gates with `cfg!(…)`, falls back to scalar if absent |
| **Locked default** | Licensed + always-compiled + not an A/B knob | hard-coded best path | hard-wire `true`, no branch |
| **Experiment / A-B / debug / log / tuning** | Runtime control flipped during investigation | `SGLANG_*` env, `get_bool_env_var` | keep the env var |
| **User-facing selection** | First-class user knob | `server_args.attention_backend` (`attention_registry.py:203`) | CLI `--flag` (e.g. `--spec-type`) |

Why **cfg, not a runtime probe**, for our compile-dependent levers: unlike
SGLang's flashinfer (a separate pip package probed at runtime), our kernels are
compiled INTO the binary — `build.rs` *already* decides `enable_flashmla` /
`enable_deepgemm_native` / Triton-AOT. So the availability is known at compile
time; `cfg!` is exact, zero-cost, and deterministic. The runtime fallback (scalar
path) stays for the cfg-off build.

## Classification (every DSv4 lever)

### A. COMPILE-DEPENDENT → move to `cfg!` (delete the env read)
The env opt-out doubled as a build-fallback (codex P1). Replace with a cfg
emitted by build.rs; on a build that didn't compile the kernel, cfg=false →
scalar fallback (no `NOT_SUPPORTED` at runtime).

| Lever | Kernel | Build gate (build.rs) | New gate |
|---|---|---|---|
| `ARLE_DSV4_FLASHMLA_DECODE` | `arle_flashmla_sm90_sparse_decode_*` | `ARLE_CUDA_DISABLE_FLASHMLA` + vendor + sm90 (b.rs:2388) | `cfg!(arle_flashmla)` (keep AtomicI8 override) |
| `ARLE_DSV4_FLASHMLA_PREFILL` | `arle_flashmla_sm90_sparse_prefill_*` | same | `cfg!(arle_flashmla)` |
| `ARLE_DSV4_FP8_LINEAR_DEEPGEMM` | `dsv4_deepgemm_m_grouped_fp8_*` | `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE` (b.rs:2501) | `cfg!(arle_deepgemm_native)` |
| `ARLE_DSV4_DECODE_PROJ_DEEPGEMM` | same | same | `cfg!(arle_deepgemm_native)` |
| `ARLE_DSV4_PREFILL_PROJ_DEEPGEMM` | same | same | `cfg!(arle_deepgemm_native)` |
| `ARLE_DSV4_PREFILL_INDEXER_DEEPGEMM` | same | same | `cfg!(arle_deepgemm_native)` |
| `ARLE_DSV4_FUSED_WQKV_DECODE` | DeepGEMM fused | same | `cuda_kernels::HAS_DEEPGEMM_NATIVE` (keep override) |

**Cross-crate mechanism** (`cfg!` is per-crate; `infer-cuda` has no build.rs):
`cuda-kernels/build.rs` emits `cargo:rustc-cfg=arle_flashmla` (after
`enable_flashmla`, b.rs:1894) and `…=arle_deepgemm_native` (after
`enable_deepgemm_native`, b.rs:2007), plus `cargo:rustc-check-cfg` at `main()`
start (before the no-cuda early-return). `cuda-kernels/src/lib.rs` exposes
`pub const HAS_FLASHMLA / HAS_DEEPGEMM_NATIVE: bool` (cfg-gated — the single
source of truth). `infer-cuda` gates read `cuda_kernels::HAS_*`. The two
FlashMLA gates above use `cuda_kernels::HAS_FLASHMLA`; the five DeepGEMM gates
use `cuda_kernels::HAS_DEEPGEMM_NATIVE`.

### B. LOCKED DEFAULT → hard-wire `true` (delete the env read)
Licensed, always-compiled, not an A/B knob:

- `ARLE_DSV4_GPU_ROUTER` → removed by `087df440` (2026-06-18). DSv4 routing is
  device-only; the host D2H route oracle and host `tid2eid` table were deleted.
- `ARLE_DSV4_COMM_OVERLAP` → removed by `087df440` (2026-06-18). B=1 allreduce
  decode always overlaps the shared expert on the comm stream; DeepEP keeps the
  non-overlap ordering.
- `ARLE_DSV4_MOE_DECODE_FP8` (moe.rs:2922) → `true`. **Verified native**: the FP8
  decode lane calls `dsv4_fp8_grouped_swiglu_decode_cuda` (native CUDA in the
  unconditional csrc glob — NOT Triton/DeepGEMM). It is the bandwidth-fixed
  *successor* lane; locking it on means the MoE decode never depends on DeepGEMM.
- `dsv4_mtp_batched_verify_enabled`, `dsv4_mtp_tree_attn_enabled`,
  `dsv4_mtp_commit_fold_enabled` → `true`. **Verified scheduling-only** (dsv4.rs
  1523/2414/2468 gate logic over already-compiled kernels, no conditional kernel).
  Only active when spec is on; harmless no-ops otherwise.

### C. KEEP as env — experiment / validation / debug / log / tuning
These ARE environment/experiment knobs (ckl's "实验时可以用环境") — leave them:

- **Experiment / A-B**: `ARLE_DSV4_DSA_INDEXER` (used by
  `scripts/dsv4_variable_shape_dsa_gate.py` for legacy-vs-official A/B — codex P2),
  `ARLE_DSV4_MOE_TRANSPORT`/`MOE_BACKEND`,
  `ARLE_DSV4_MOE_CONTIG_DECODE`, `ARLE_DSV4_WHOLE_STEP_GRAPH`,
  `ARLE_DSV4_DECODE_GRAPH`, `ARLE_DSV4_MTP_FROZEN_LAYER`,
  `*_ALLOC` force-alloc probes.
- **Debug / log**: `ARLE_DSV4_ATTN_DUMP`, `KNEW_DUMP`, `CSA_DUMP`, `STEP_PROFILE`,
  `STAGE_PROFILE`, `TAIL_DUMP`, `MTP_ROLLBACK_DUMP`, `DSA_LOGITS_PROBE`, `RUST_LOG`.
- **Tuning**: `ARLE_DSV4_DSA_INDEXER_SMS`, `DSA_LOGITS_PROBE_SMS`.

### D. USER-FACING → CLI (no change)
- `--spec-type {none|mtp|auto}` stays the user knob. (`ARLE_DSV4_SPEC_DECODE` env is
  a redundant legacy alt of the CLI — fold into the CLI or mark experiment;
  low priority.) The *default* spec policy (zero-config d2) is a separate
  model-aware-`Auto` task, NOT this refactor.

## Why correct + safe
- **Production (pod: full build — FlashMLA + native DeepGEMM + Triton) is
  byte-identical**: all cfgs on → same paths as today's default-on.
- **Reduced builds fixed** (codex P1): a no-DeepGEMM build → `cfg!(arle_deepgemm_native)=false`
  → scalar fallback, no `NOT_SUPPORTED`. The env was the wrong layer; cfg is right.
- **Validation scripts fixed** (codex P2): `ARLE_DSV4_DSA_INDEXER` stays env.
- **No A/B loss that matters**: compile-dependent levers were build-fallbacks, not
  perf A/B; the perf A/B knobs (transport, router, graph) stay env.

## Deletion-refactor steps (incisive)
1. `build.rs`: emit `cargo:rustc-cfg` + `cargo:rustc-check-cfg` for `arle_flashmla`,
   `arle_deepgemm_native`, `arle_fused_moe_triton` in the existing enable-flag blocks.
2. `attention.rs`: 8 compile-dependent gates → `cfg!(…)` (keep the 2 AtomicI8
   overrides); `moe.rs`: the FP8-lane gate → `cfg!(arle_fused_moe_triton)`.
3. `dsv4.rs`: 3 MTP levers → `true`.
4. Leave every C/D lever untouched.

## Verify
- Mac typecheck (full build, cfgs on): byte-identical to default → compiles.
- A `--no-default-features … no-cuda` / no-DeepGEMM check build: cfgs off →
  scalar fallback path compiles (proves the build-fallback works).
- Pod rebuild (full): needle 512/6000 ×3 + B=1 ≈ 53.3 (unchanged, byte-identical).

## Out of scope (separate, honest)
- **Zero-config d2-spec default** → model-aware `--spec-type auto`.
- **Batched MLA decode (#60)** + whole-step graph (#70) → the concurrency PERF
  keystone (multi-session; `dsv4-concurrency-throughput.md`). "DP-attn" is lever
  #4 there, not #1 — the #1 is batched MLA decode.
