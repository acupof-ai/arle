# DSv4 config-mechanism classification — compile-dependent → cfg, proven → locked, experiment → env

## Context

The DSv4 runtime had ~30 `ARLE_DSV4_*` env gates of mixed nature. Per ckl's
principle ("环境变量顾名思义就是环境的 — log/实验 用 env;依赖编译的走编译") —
verified 1:1 against SGLang's own taxonomy at the code level
(`is_flashinfer_available()` capability guards `flashinfer_backend.py:46`;
`server_args.attention_backend` for user knobs; `SGLANG_*` env for debug) — each
lever is moved to the RIGHT mechanism. Design:
[`dsv4-config-mechanism-classification.md`](../../plans/dsv4-config-mechanism-classification.md).

## What changed

### A. Compile-dependent → capability (no env). TWO sub-cases, by symbol nature:
The env opt-out doubled as a build-fallback (codex: a reduced build →
`NOT_SUPPORTED`). The right layer depends on whether the stub exports the symbol:

**FlashMLA → build `cfg` const** (its real-kernel marker is **real-only** — the
stub omits it, so presence is build-determinable):
- `cuda-kernels/build.rs` emits `cargo:rustc-cfg=arle_flashmla` from
  `enable_flashmla` (+ `rustc-check-cfg` at `main()` start), **and** in the
  prebuilt fast-build path (`link_prebuilt_cuda_artifacts`, after
  `validate_prebuilt_cuda_archive_symbols` mandates the real marker) — that path
  returns before the normal emission, so without it a valid prebuilt would report
  `HAS_FLASHMLA=false` and lose FlashMLA (**codex P1, fixed**).
- `cuda-kernels/src/lib.rs`: `pub const HAS_FLASHMLA: bool` (cfg-gated; `cfg!` is
  per-crate, so the const is how `infer-cuda` sees it). FlashMLA decode/prefill
  gates → `cuda_kernels::HAS_FLASHMLA`. The `AtomicI8` overrides stay (test API).

**DeepGEMM → cached RUNTIME preflight probe** (the non-native **stub exports the
same bridge symbols**, incl. `dsv4_deepgemm_native_preflight_cuda` returning
`NOT_SUPPORTED` — so it is NOT build-determinable; **codex P2** showed a cfg would
misreport a stub prebuilt as native):
- `cuda-kernels/src/lib.rs`: `pub fn has_deepgemm_native() -> bool` — cached
  `OnceLock` over `moe::dsv4_deepgemm_native_preflight().is_ok()` (the device
  truth; `#[cfg(not(cuda))] → false`). Mirrors SGLang's `is_*_available()`.
- The 5 DeepGEMM gates (fp8-linear, decode-proj, prefill-proj, prefill-indexer,
  fused-wqkv) → `cuda_kernels::has_deepgemm_native()`. No build cfg, so all build
  paths (normal / prebuilt / stub) report the truth — **codex P2, resolved**.

### B. Proven + always-compiled → locked default (`true`, no branch)
Verified native/scheduling (not Triton/DeepGEMM-dependent):
- `ARLE_DSV4_MOE_DECODE_FP8` → `true`: the FP8 decode lane calls
  `dsv4_fp8_grouped_swiglu_decode_cuda` (native CUDA, unconditional csrc) — the
  successor lane, so MoE decode no longer depends on DeepGEMM at all.
- `dsv4_mtp_{batched_verify,tree_attn,commit_fold}_enabled` → `true`: pure
  scheduling over already-compiled kernels (dsv4.rs 1523/2414/2468).

### C. Kept as env (correctly classified)
Experiment/A-B (`ARLE_DSV4_DSA_INDEXER` — used by the validation script;
`GPU_ROUTER`, `MOE_TRANSPORT`, `MOE_CONTIG_DECODE`, graph levers, `*_ALLOC`),
debug (`*_DUMP`, `*_PROBE`, `STEP/STAGE_PROFILE`), tuning (`*_SMS`), log
(`RUST_LOG`). These ARE environment/experiment knobs — untouched.

### D. User-facing → CLI (`--spec-type`) — untouched.

## Why correct + byte-identical for production
- **Pod (full build: FlashMLA + native DeepGEMM)**: both cfgs emit → `HAS_*=true`
  → identical paths to today's default-on. Byte-identical.
- **Reduced build** (codex P1 fixed): no-DeepGEMM → `HAS_DEEPGEMM_NATIVE=false`
  → scalar FP8-GEMV fallback, no `NOT_SUPPORTED`. The env was the wrong layer.
- **Validation scripts** (codex P2 fixed): `ARLE_DSV4_DSA_INDEXER` stays env.

## Verify
- Mac CUDARC typecheck (`infer-api`, `cuda,no-cuda`): clean. The `no-cuda` build
  takes `HAS_*=false` → exercises + compiles the **scalar fallback path** (the
  P1-fix path). No unexpected-cfg lint (check-cfg declared).
- `codex review`: clean (0 findings) after the P1 (prebuilt-path cfg) + P2
  (DeepGEMM stub-vs-native → runtime probe) fixes.
- Pod (full build) needle 512/6000 + B=1: **DONE 2026-06-14** — synced to clean
  `7d660f66` via git bundle (ckl's #88 Triton WIP backed up + reset), rebuilt
  (cuda-kernels recompiled, FlashMLA/DeepGEMM markers present). **Correctness
  CLEAN**: needle exact retrieval at 512/2000/6000 (one 2000 partial = MoE
  non-determinism, within the locked envelope). **Perf CLEAN, no regression**:
  B=1 decode **forward floor 42.0 ms/step** (acceptance-independent; rock-solid
  41.7–42.3 across factual/code/structured/creative prompts, <1.5% spread —
  slightly better than the pre-refactor 42.7 ms/step). tok/s 43.2→55.6 swings
  **entirely by MTP acceptance** (tok/step 1.82→2.34); code-prompt hits 55.6,
  exceeding the 53.3 d2 chain-fold doc baseline. The apparent "42-45 vs 53.3 gap"
  was a creative-writing-prompt low-acceptance artifact + cross-tree confound
  (53.3 was measured on `/data01/build/arle-dsv4` with DeepEP), NOT a forward
  regression. Confirms the refactor is byte-identical-behavior on the full build.
- Two serve-interface facts surfaced (rewrite stack, not refactor-caused): DSv4
  rejects `--kv-cache-dtype fp8` (it owns FP8 KV internally, #68 T3 flag is
  dense-Qwen3-only); DSv4 slot cap is `INFER_DSV4_MAX_SEQ_LEN` env, no CLI flag.

## Coordination note
Landed alongside ckl's concurrent `#88` commit (`23d6a0b8` — delete
validated-losing SGLang kernel-align/Triton lanes). The two are orthogonal +
compatible: ckl's deletion *confirms* MOE_DECODE_FP8 is native (B above), and the
FlashMLA/DeepGEMM `enable_*` flags this refactor's cfgs read are retained.

## Rule

- **Classify config by NATURE, not by "make it default."** Compile-dependent
  kernel availability → compile layer (`cfg`/capability const), never a runtime
  env (the env can't know what was compiled, and removing it breaks the
  build-fallback). Proven+always-compiled → locked default. Experiment/debug/log
  → env. User selection → CLI. (My first pass hard-wired all to `true` — wrong:
  it deleted build-fallbacks; codex caught it.)
- **`cfg!` is per-crate.** To expose a build decision across crates, the
  deciding crate exports a cfg-gated `pub const`; downstream reads the const.
