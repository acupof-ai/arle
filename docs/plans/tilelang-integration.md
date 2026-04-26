# TileLang integration — Phase 0: prefill HD128 attention

**Status:** Phase 0 in flight (2026-04-26). Owner: ckl. Verification: remote
H100 (user-driven). Prior art: `docs/plans/cuda-kernel-crate-extraction.md`
(blueprint style), `crates/cuda-kernels/build.rs` lines 195–263 + 265–510
(Triton AOT pattern this plan mirrors).

---

## 1 · Why this plan exists

We want to evaluate whether a [TileLang][tilelang] kernel can beat the
current FlashInfer batch prefill HD128 path end-to-end. The decision rule
the user laid down is: **don't migrate small kernels — only kernels whose
end-to-end delta is large enough to justify the swap, and only by replacing
a complete sub-graph, not a fragment**.

Phase 0 picks the smallest sub-graph that is (a) on the hot path of
`bench_guidellm.sh` TTFT/throughput and (b) cleanly bounded by an existing
FFI seam. That is the **batch prefill HD128 attention compute** — the
`flashinfer_batch_prefill_paged_hd128_plan` + `_run` pair. Everything around
it (prep kernel, projections, decode path, HD256) stays untouched.

**Verification path:** user runs `scripts/bench_guidellm.sh
tilelang-prefill-{on,off}` on remote H100 + `cargo test --features
cuda,tilelang-attn --test e2e` for numerical parity. Local Mac workspace
verifies type-check (`cargo check -p infer --no-default-features --features
cuda,no-cuda`) and Triton-style build wiring; CUDA runtime + bench are
remote-only.

[tilelang]: https://github.com/tile-ai/tilelang

---

## 2 · Sub-graph contract (precise boundary)

Replace **only** these two FFI calls:

- `flashinfer_batch_prefill_paged_hd128_plan`
  (`crates/cuda-kernels/src/ffi/attention.rs:410`)
- `flashinfer_batch_prefill_paged_hd128_run`
  (`crates/cuda-kernels/src/ffi/attention.rs:427`)

Keep untouched:

- `prefill_attention_paged_prep_cuda` (RMS norm + RoPE + KV write to paged
  pool) — this is data-path, orthogonal to attention compute.
- HD256 prefill (`flashinfer_batch_prefill_paged_hd256_*`) — Qwen3.5
  full-attn parity, out of scope.
- Decode HD128/HD256 — out of scope (Phase 1 if Phase 0 wins).
- All projections, residual, MLP, norms outside the prep kernel.

**Contract for the TileLang kernel** (BF16 throughout, causal mask, no soft-cap):

```text
Inputs (device pointers):
  q          : [packed_tokens, num_q_heads * 128]   already RMS-normed + RoPE'd
  k_pool     : paged K storage, HND layout, page_size = 16
  v_pool     : paged V storage, HND layout, page_size = 16
  q_indptr   : [batch_size + 1]   packed-token offsets per request
  kv_indptr  : [batch_size + 1]   paged-KV page-index offsets per request
  kv_indices : flattened page indices for all sequences in the batch
  kv_last_page_len : [batch_size]
Outputs:
  o          : [packed_tokens, num_q_heads * 128]
Scalars:
  num_q_heads = 32, num_kv_heads = 8 (Qwen3-4B/8B; both share these)
  head_dim    = 128
  sm_scale    = 1.0 / sqrt(128)
```

This signature is identical to the FlashInfer `_run` call, minus the
plan-info opaque buffer and minus LSE (we never read it).

---

## 3 · Build-time AOT pattern (mirror Triton)

The project already runs Python-driven AOT compilation for Triton kernels in
`crates/cuda-kernels/build.rs`. Phase 0 adds a parallel TileLang track that
follows the same shape:

| Step | Triton (existing) | TileLang (new) |
|------|-------------------|----------------|
| Find Python | `find_triton_python()` (env `INFER_TRITON_PYTHON`, then `.venv`s, then `python3`/`python`) | `find_tilelang_python()` (env `INFER_TILELANG_PYTHON`, same fallback chain, probes `import tilelang`) |
| AOT script | `tools/triton/gen_triton_aot.py` | `tools/tilelang/gen_tilelang_aot.py` |
| Per-kernel spec | `TritonKernelSpec` (kernel_path, name, signature, grid, num_warps, num_stages) | `TileLangKernelSpec` (kernel_path, name, target SM, output dir) |
| Output | CUBIN + generated C wrapper → linked into `libcuda-kernels.a` | Same: CUBIN + generated C wrapper → linked in |
| Runtime | C function pointer; no Python | Same |
| Feature gate | always on under `feature = "cuda"` | only under `feature = "tilelang-attn"` (which itself implies `cuda`) |

Why this pattern:

- Production runtime contains no Python — matches ARLE's existing posture.
- Reuses existing `cargo:rerun-if-changed` plumbing for `tools/`.
- Keeps `cargo build --features cuda` (default) byte-identical to today —
  TileLang work is purely additive behind its feature flag.

`@tilelang.jit` JIT-and-cache was considered and rejected (Option B in the
2026-04-26 design discussion) because it pulls Python into the prod path
and breaks parity with how Triton is wired today.

---

## 4 · File-level changes

### 4.1 New files

| Path | Purpose |
|------|---------|
| `crates/cuda-kernels/tools/tilelang/__init__.py` | Package marker. |
| `crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py` | AOT compile script. Inputs: kernel module path, kernel function name, target SM, output dir. Outputs: CUBIN + generated C wrapper. Prints `FUNC_NAME=…` and `C_PATH=…` on stdout (mirrors Triton). |
| `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py` | TileLang kernel definition for the contract in §2. |
| `crates/cuda-kernels/tools/tilelang/README.md` | Bootstrap notes: `pip install tilelang`, `INFER_TILELANG_PYTHON`, expected versions. |
| `docs/plans/tilelang-integration.md` | This document. |
| `docs/experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md` | Bench stub (pending-remote per §Benchmarks rules in CLAUDE.md). |

The AOT-generated C wrapper lands under `OUT_DIR/tilelang_aot/<artifact>/`,
not in the source tree. No hand-written `.cu` wrapper is added.

### 4.2 Modified files

| Path | Change |
|------|--------|
| `infer/Cargo.toml` | Add `tilelang-attn` feature → forwards to `cuda-kernels/tilelang-attn`. Implies `cuda`. |
| `crates/cuda-kernels/Cargo.toml` | Add `tilelang-attn = ["cuda"]` feature. |
| `crates/cuda-kernels/build.rs` | Behind `cfg(feature = "tilelang-attn")`: probe Python with `import tilelang`, run AOT generator for the prefill kernel, compile generated C wrapper into the kernel `.a`. Mirrors the existing Triton track lines 265–510. |
| `crates/cuda-kernels/src/ffi/attention.rs` | Add `tilelang_batch_prefill_paged_hd128_run` extern declaration behind `cfg(feature = "tilelang-attn")`. Same arg list as the FlashInfer `_run`, minus the opaque plan info. |
| `crates/cuda-kernels/src/lib.rs` | Module/feature gating only. |
| `infer/src/ops/attention.rs` | At the FlashInfer call site (~line 540 in `prefill_attention_paged_batch`): when `cfg(feature = "tilelang-attn")` is set, dispatch to TileLang; otherwise dispatch to FlashInfer. **No runtime branch** (compile-time only) for Phase 0 — A/B is feature-build vs default-build, not a runtime toggle. This matches `feedback_no_half_states.md`: one canonical path per build. |
| `pyproject.toml` | Add `[project.optional-dependencies] tilelang = ["tilelang>=…"]`. Pin in §6 once spike picks a version. |

**Total:** 6 new files, 7 modified files = 13 files. Within the >5-file
threshold; this is why approach-first + plan doc lands before code.

### 4.3 Deliberately NOT changed

- HD256 prefill, decode HD128/HD256, prep kernel, scheduler, model code
  outside the single dispatch site, any `.cu` source, any Triton kernel,
  any test data baseline.
- No changes to `crates/cuda-kernels/src/prelude.rs` — the new FFI is not a
  proto-public symbol, it's an internal alternate path.

---

## 5 · Risk gates and stop conditions

This plan is fail-fast. Stop conditions, in order:

1. **TileLang cannot AOT-export the kernel for `sm_90`.** The AOT generator
   panics in `build.rs`, no FFI is wired yet. Outcome: write
   `docs/experience/errors/2026-04-…-tilelang-aot-sm90-blocker.md`,
   delete the new files, revert. Phase 0 closed; user decides whether to
   reopen as Option B (runtime JIT).
2. **TileLang lacks paged-KV BatchPrefill primitives at the API level.** If
   the `tools/tilelang/batch_prefill_paged_hd128.py` cannot express the
   §2 contract without forking TileLang, stop and write the same
   blocker entry.
3. **Numerical parity failure on H100.** `cargo test --features
   cuda,tilelang-attn --test e2e` fails the `infer/test_data/Qwen3-4B.json`
   substring match. Outcome: errors entry, revert.
4. **Performance regression on H100 bench.** TileLang TTFT or out-tok/s is
   ≥5% worse than FlashInfer at any sweep step. Outcome: errors entry,
   feature flag stays in tree off-by-default but Phase 1 does not start.
5. **Performance flat (within ±5%).** Outcome: wins entry documenting flat,
   feature flag stays in tree off-by-default, Phase 1 does not start.
6. **Performance win ≥10% on TTFT or saturation throughput.** Outcome:
   wins entry, Phase 1 (decode HD128/HD256 migration) begins; feature
   flag eventually retired in favor of TileLang as the only path.

The 5–10% band is intentionally a no-go zone — too small to justify the
runtime complexity of carrying two attention paths long-term per the
"clean and uniform" principle in CLAUDE.md.

---

## 6 · Open questions (resolved during spike, not before)

- **TileLang version pin.** Picked by the AOT generator's first successful
  run on H100. Documented in the wins entry.
- **CUTLASS / CuTeDSL backend selection.** TileLang's CuTeDSL backend
  (per the upstream README) gives the best Hopper utilization. The Python
  kernel will request it explicitly; if unavailable in the pinned version,
  fall back to the default backend and note it.
- **`num_warps` / pipeline stages.** Tuned during spike; logged in the
  bench entry. Not a contract change.

---

## 7 · Acceptance checklist

Phase 0 is "done" only when **all** of these are true:

- [ ] `cargo build --release --features cuda` produces a binary identical
  in behavior to pre-Phase-0 (no functional change to default build).
- [ ] `cargo build --release --features cuda,tilelang-attn` builds on the
  H100 host.
- [ ] `cargo check -p infer --no-default-features --features cuda,no-cuda`
  passes on macOS (type-check guard).
- [ ] `cargo test --release --features cuda,tilelang-attn --test e2e`
  passes on H100 (numerical parity vs `infer/test_data/Qwen3-4B.json`).
- [ ] `scripts/bench_guidellm.sh tilelang-prefill-on` and
  `…-prefill-off` both ran on H100; deltas captured in the wins entry.
- [ ] Decision recorded per §5: ship to Phase 1, hold flat, or revert.
- [ ] No half-states left in tree (`feedback_no_half_states.md`): either
  the feature is in and gated, or the new files are gone. Never both.

---

## 8 · Phase 1 preview (informational only)

If Phase 0 wins ≥10%: Phase 1 migrates decode HD128 + HD256 to TileLang
under the same feature flag, then retires the flag and removes the
FlashInfer attention dependency from the prefill+decode hot path. HD256
prefill (Qwen3.5 full-attn) follows after decode is stable. MLA (DeepSeek
path, currently roadmapped) becomes a Phase 2 target with separate plan
doc — TileLang's MLA primitives are the strongest case in the upstream
project and warrant their own evaluation.
