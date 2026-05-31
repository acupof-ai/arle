# DSv4 DeepGEMM expert-GEMM — integration, root-causes, fix state, alternatives, and cleanup plan

**Audience:** Codex (execution). This doc is self-contained: every change has an exact
file path, a verification step, and a sanitized remote-run recipe. No IPs / no absolute pod
paths anywhere (directory tails only).

**One-paragraph summary.** DSv4-Flash routes its MoE expert FFN through one of two expert
backends: the scalar grouped GEMV loop (`native`, the working default) or DeepGEMM's FP8
grouped masked GEMM (`deepgemm`). DeepGEMM is a from-scratch SM90 port that JIT-compiles its
kernel at runtime. It was "crashing" for weeks; the real cause was a **runtime JIT nvcc
compile-flag bug** (now fixed — coherent output, 0 GEMM failures). But the now-running path
has a **second, independent problem**: it's fast at small prefills (−9.5% vs scalar at 545
tok) yet **times out at ≥1024 tok** because the grouped-GEMM tile grid is sized to the
per-rank route *total* instead of the per-expert *max* (~32× oversizing). Net: the crash is
solved, deepgemm is *usable only ≤~512 tok*, and it's a ~10% lever even when fixed — so the
real prefill bottleneck is elsewhere.

---

## 1. How the GEMM is wired (systematic, top→bottom)

### 1.1 Backend selection
- Env `ARLE_DSV4_EXPERT_BACKEND` ∈ {`native` (scalar, default), `deepgemm`}. The native-deepep
  MoE path reads it and sets `use_deepgemm_experts`.
- Build gate: DeepGEMM native is `#ifdef ARLE_ENABLE_DEEPGEMM_NATIVE` in
  `crates/cuda-kernels/csrc/gemm/deepgemm_native.cu`; the macro is only defined when the
  **build-time** env `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` is set (`crates/cuda-kernels/build.rs`
  ~1355). Without it, `deepgemm_bridge_stub.cu` links and returns `CUDA_ERROR_NOT_SUPPORTED`
  unconditionally. (This stub-masking burned a full session — see §2.0.)

### 1.2 The call chain (prefill, native-deepep path)
```
infer/src/model/deepseek/mlp.rs
  forward_native_deepep_routed_gpu            (~4877)  # DeepEP IPC dispatch → recv → pack
    └─ dsv4_pack_local_experts_cuda                    # recv_x → packed_x (slot-major), packed_token, packed_weight
    └─ if use_deepgemm_experts:
         forward_deepgemm_all_dsv4_experts_gpu (~2664) # "all experts" wrapper
           ├─ dsv4_prepare_deepgemm_all_expert_metadata_cuda  # build active.{indices,offsets,counts} from local_{offsets,counts}
           └─ forward_deepgemm_grouped_dsv4_experts_gpu (~2356)
                ├─ ensure_deepgemm_scratch (state.rs ~827)     # capacity_m, scale_stride_m, fp8/scale/out buffers
                ├─ dsv4_deepgemm_pack_quantize_bf16_to_fp8_cuda    # compact bf16 → padded FP8 + MN-major scales (w13 input)
                ├─ dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda # ── the GEMM (w13) ──
                ├─ dsv4_deepgemm_swiglu_quantize_w13_cuda          # SwiGLU + requantize for w2
                ├─ dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda # ── the GEMM (w2) ──
                ├─ dsv4_deepgemm_unpad_grouped_bf16_cuda           # padded out → compact out
                └─ dsv4_scatter_all_route_slots_cuda               # compact out → expert_out[route_slot]*weight
```
Decode uses the analogous padded path `forward_deepep_routed_gpu` (~3180) with the same
grouped GEMM but a fixed ≤256 capacity.

### 1.3 The GEMM kernel itself (the SM90 port)
`deepgemm_native.cu` is a hand-port of DeepSeek's DeepGEMM, *not* a vendored binary:
- `get_best_config` / `get_layout_candidates` pick block_m/block_n/cluster from the GEMM
  shape (uses `expected_m` = `max_m`).
- `generate_kernel_code` emits CUDA source; `compile_with_nvcc` shells `nvcc -cubin` at
  **runtime** (JIT), cached under `$HOME/.deep_gemm/cache/kernel.<name>.<digest>/`.
- `make_tma_{a,b,d,sfa}_desc` build the TMA tensor maps; `launch_sm90_grouped_masked` does
  the cluster launch. Gate: `prop.major != 9 → NOT_SUPPORTED` (Hopper-only).
- Scale layout (verified consistent): pack writes `expert*stride*k_blocks + k_block*stride +
  row` (`dsv4_deepgemm_ops.cu` `dg_scale_offset`), and the SFA TMA reads the identical
  MN-major layout — so the scale path is **not** a bug.

---

## 2. Why it had problems

### 2.0 Confounder layer (cost us most of the time — record so we don't repeat)
1. **Stub masking.** Every build that didn't set `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` linked
   the stub → `CUDA_ERROR_NOT_SUPPORTED` for *every* GEMM, which read as a kernel/shape bug.
   Rule: a `NOT_SUPPORTED` from a build-flag-gated bridge = "stub linked" until the binary is
   grepped for the real symbol. (`memory/feedback_deepgemm_build_flag_stub.md`.)
2. **A premature root cause that got refuted (record the discipline).** The `mem-fraction-static=0.10`
   smoke default + a `core.rs:1733` "host tier full, dropped GPU blocks" WARN *looked* like KV
   starvation causing the ≥1024–2048 timeouts. **A controlled 0.10-vs-0.85 re-run REFUTED it:**
   the 512/1024 numbers are byte-identical and 2048 still times out at both budgets — so
   mem-fraction is NOT the cause. The 8 WARNs are a one-time early event (one per rank, same
   instant), not a thrash loop. (Use 0.85 for perf benches anyway — it's the default — but it
   is not the lever here.) Lesson: a suggestive WARN + a low config value is a *hypothesis*;
   the controlled flip is the evidence (§0).

### 2.1 Problem 1 — the crash (SOLVED, see §3.1)
`CUDA_ERROR_LAUNCH_FAILED "unspecified launch failure"` / `CUDA_ERROR_UNKNOWN` on every
expert GEMM. **Root cause: the runtime JIT nvcc compile *failed*.** The JIT passed
`-std=c++17 --compiler-options=...,-fconcepts`; `-fconcepts` makes gcc-13's libstdc++
`<type_traits>` expose its C++20 `requires` detection idiom (type_traits:2651), which nvcc's
*device* C++17 frontend (cicc/EDG) cannot parse. The compile died and the failure surfaced
downstream as a driver error on the GEMM call. The flag was correct for the old CUDA-12.2 pod
(commit 38bf157b); this pod is CUDA 12.9 + gcc 13.3. It was **not** an IMA, **not** the Rust
calling contract ("candidate-2"), **not** the kernel port. (Lesson: a device error code names
the surface; grep the whole error chain incl. subprocess stderr first —
`memory/feedback_jit_error_code_names_surface_not_cause.md`.)

### 2.2 Problem 2 — the perf/scaling cliff (NOT yet fixed)
`forward_deepgemm_all_dsv4_experts_gpu` (mlp.rs ~2725-2726) passes `route_capacity =
total_local_routes` as **both** `total_local_routes` **and** `max_local_routes` to the grouped
fn. So the masked GEMM's `expected_m = max_m = the full per-rank route total`, applied **per
group across all 32 experts**. The persistent-kernel tile grid is therefore sized for
`total_routes × 32` row-slots when only `total_routes` rows exist (avg ~total/32 per expert)
— ~**32× tile oversizing** that grows with token count. Most tiles early-exit via `masked_m`,
but the tile scheduling over 32× too many tiles is tolerable at 545 tok (max_m≈384) and
pathological at ≥1024 tok (max_m≈768) → >300 s. The non-`_all_`
`forward_grouped_dsv4_experts_gpu` already computes `max_local_routes = local_counts.max()`
correctly; the `_all_` wrapper doesn't.

---

## 3. How it was solved & current state

### 3.1 Fix (LANDED)
`crates/cuda-kernels/csrc/gemm/deepgemm_native.cu` `compile_with_nvcc`: `-std=c++20`, drop
`-fconcepts`. CUDA 12.9 nvcc supports c++20 device-side, so concepts parse natively.
Ground-truthed *without* an infer rebuild by recompiling the exact failing leftover
`$HOME/.deep_gemm/tmp/arle-*/kernel.cu`: c++17+fconcepts → EXIT=1 (type_traits:2651),
c++20-no-fconcepts → EXIT=0 (clean 83 KB cubin). Then e2e on the pod: native-deepep +
`EXPERT=deepgemm` returns coherent "The capital of France is Paris.", `GEMM_FAILURES=0`, 6 JIT
kernels compiled clean. (Also landed: `scripts/dsv4_toolchain.sh` `ARLE_SERVER_WRAP` hook that
surfaced the buried compile error via compute-sanitizer.)

### 3.2 Current measured state (mem-fraction-static=0.85, the valid budget)
| prefill | deepgemm | scalar (native) |
|---|---|---|
| 545 tok  | **6.91 s** ✓ (−9.5% vs scalar) | 7.63 s ✓ |
| 1089 tok | **TIMEOUT >300 s** ✗ (compute cliff §2.2) | 16.70 s ✓ |
| 2048 tok | timeout ✗ | **TIMEOUT >300 s ✗** (real cliff, root cause OPEN — not mem-fraction) |

deepgemm is **partially solved**: correct + fast for small prefills, **unusable at ≥1024 tok**
until §2.2 is fixed. Even fully fixed it's a ~10% prefill lever (the expert GEMM is ~1/3 of
prefill; flipping it moved e2e only 9.5%).

---

## 4. Other solutions / alternatives (还有什么其他方案)

### 4.1 To make deepgemm viable (fix the §2.2 cliff) — ranked
- **A (recommended, cheap): pass the real per-expert max.** In
  `forward_deepgemm_all_dsv4_experts_gpu`, compute `max_local_routes = max_e(count_e)` (the
  native path already has `counts_host`) and thread it as the `max_local_routes` arg, keeping
  `route_capacity`/`total_local_routes` for scratch+output sizing. Mirrors the working
  non-`_all_` path. ~1 function change. Expected: removes the 32× oversizing → deepgemm should
  serve ≥1024 and the win may grow (the GEMM does ~1/32 the wasted tiles).
- **B: fixed max_m buckets.** Pad `max_m` to a small bucket set (e.g. {64,128,256,512}) so the
  JIT compiles ≤4 kernels and the grid is bounded. More code; also caps JIT-kernel count.
- **C: route through the non-`_all_` grouped path** for the native-deepep prefill (it already
  sizes `max_local_routes` correctly) instead of `_all_`. Smallest behavioral delta if the
  metadata is available host-side.

### 4.2 Expert-GEMM backend alternatives (the broader design space)
- **Scalar grouped GEMV (`native`, current default)** — works to 2048, no JIT, but M-blind;
  fine as the safe default.
- **DeepGEMM FP8 grouped (this work)** — fast small, needs §4.1; best ceiling for FP8.
- **cuBLASLt / CUTLASS grouped GEMM** — mature, no hand-port risk, but not FP8-block-scaled
  the way DeepGEMM is; would lose the FP8 perf.
- **The existing tiled `dsv4_fp8_gemv_batch` (`_tiled_kernel`, DSV4_BATCH_TILE=32)** — already
  used for prefill per-expert with weight reuse; a known-good middle ground.

### 4.3 The real lever (deepgemm is ~10% — don't over-invest)
The A/B proves the expert GEMM is a minority of prefill; ~2/3 is the non-GEMM MoE path. The
high-ROI work for "beat SGLang 30%" is, in priority order, each **gated behind a fresh
nsys/phase profile of a 1024-tok prefill at 0.85** (do NOT act on stale profiling — §0):
1. **native-deepep dispatch/combine/host-poll.** Prior profiling put combine ~52% of FFN and
   flagged the per-layer `num_recv` host-poll. Cheapest experiment: A/B native-deepep vs
   `allreduce` MoE backend for **prefill only** (native-deepep was +46% at *decode* but
   *slower* at prefill) → consider allreduce-for-prefill / native-deepep-for-decode.
2. **The 2048 cliff (CONFIRMED real, root cause OPEN).** native is 16.7 s @1024 but >300 s
   @2048 at BOTH 0.10 and 0.85 budgets (mem-fraction refuted, §2.0.2). >18× for 2× tokens is
   far past quadratic → a threshold, not smooth scaling. Cheapest next experiment: ONE 2048
   prefill at `RUST_LOG=info` (raise/stream past the 300 s handler cap) to see whether it's
   compute-superlinear (attention/chunked-prefill), the host-tier demotion path, or a buffer
   realloc — do NOT guess again.
3. **§4.1 deepgemm fix** — cheap, makes the FP8 path correct, ~10% where it applies.

---

## 5. Code cleanup (整理干净代码) — exact items for Codex

1. **Remove the inert debug probe.** `deepgemm_native.cu` `get_layout_candidates` (~490-503):
   the `ARLE_DEEPGEMM_CONSERVATIVE_LAYOUT` env block was a b335 scaffold added when the stub
   was linked (commit e2b3f40e) — it never gated anything real. Delete the `conservative`
   branch + the comment unless §4.1-B chooses fixed buckets (then repurpose it). Verify the
   default candidate set is unchanged.
2. **Implement §4.1-A** (the `max_local_routes` fix) — the one real perf/correctness gap.
3. **Promote the A/B harness to the repo.** The bench ran from pod-local `/tmp/dgbench_run.sh`
   + `dgbench_client.py`. Commit a cleaned version as `scripts/dsv4_expert_backend_ab.sh`
   (same-binary, flips `ARLE_DSV4_EXPERT_BACKEND`, warm-up + median, `mem-fraction-static=0.85`)
   so the A/B is reproducible. Reference it from the wins entry.
4. **Delete untracked debug scaffolds.** 12 unreferenced `scripts/_*.sh` (`_dg_cons.sh`,
   `_dg_short.sh`, `_flashmla_24k_test.sh`, `_mem_probe.sh`, `_nd_*.sh`, `_trace_profile.sh`)
   — 0 refs in the tree. **Verify with the owner of the parallel qwen35/train session first**
   (some may be their scratch), then `rm`.
5. **Update the wins entry** `docs/experience/wins/2026-05-31-bench-dsv4-deepgemm-vs-scalar-prefill.md`
   with the 0.85 numbers + the corrected verdict (the 0.10 conclusions were confounded).
6. **Pod build-tree hygiene.** The pod checkout had `dsv4_toolchain.sh` + `deepgemm_native.cu`
   sed-patched in place to match origin; they're functionally aligned but comments drift —
   `git -C <build-root> checkout -- scripts/dsv4_toolchain.sh crates/cuda-kernels/csrc/gemm/deepgemm_native.cu`
   then re-pull origin so the pod tree == origin before the next build.

---

## 6. Verification (Codex must run each gate)

- **Local CI parity** (Mac, no nvcc):
  - `cargo fmt --all -- --check`
  - `CUDARC_CUDA_VERSION=12060 cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo clippy --workspace --no-default-features --features no-cuda -- -D warnings`
  - `cargo test --release --no-default-features --features no-cuda`
- **For any `deepgemm_native.cu` / mlp.rs deepgemm change** (CUDA — pod only): rebuild with the
  full env (§7), then smoke (coherent output + `GEMM_FAILURES=0`), then the A/B at
  `mem-fraction-static=0.85` across 512/1024/2048. A diff isn't done until a dated
  `docs/experience/wins/` entry lands (CLAUDE.md Verify gate).
- **§4.1-A acceptance:** deepgemm serves 1024 *and* 2048 without timeout AND
  median_prefill(deepgemm) < median_prefill(scalar) at ≥2 shapes → only then consider enabling
  it (gated first, default after multi-shape per the distilled "default flips need multi-shape"
  rule).
- Expert-GEMM changes don't touch KV → `kv_precision_parity` not required.

---

## 7. How to use the remote (sanitized — no IP, directory tails only)

- **Access:** `~/bin/pod '<one command>'` (tn tunnel → kubectl; see
  `memory/project_h20_pod_access`). Never ssh a jumpbox. Never embed IPs or absolute pod
  paths in committed artifacts — cite by directory tail (`<build-root>`, `<models>/...`).
- **Build tree:** `CARGO_TARGET_DIR=<build-root>/target-pod`. Model at `<models>/DeepSeek-V4-Flash`.
  DeepEP source at `<deepep-src>`; DeepGEMM vendored under the cuda-kernels crate.
- **Build a deepgemm+deepep binary (BOTH envs required):**
  ```
  CARGO_TARGET_DIR=<build-root>/target-pod \
  ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 \
  ARLE_DEEPGEMM_ROOT=<cuda-kernels>/vendor/deepgemm \
  ARLE_DEEPGEMM_LIBRARY_ROOT=<cuda-kernels>/vendor/deepgemm/deep_gemm \
  ARLE_DEEPEP_DIR=<deepep-src> \
  ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1 CUDA_HOME=/usr/local/cuda \
  cargo build --release -p infer --features cuda,nccl --bin infer
  ```
  Omitting `ARLE_DEEPEP_DIR` ships a deepep-sys **stub** → native-deepep panics at boot
  ("built in stub mode"). Omitting `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE` ships the GEMM stub.
- **Long ops survive the tunnel via detached tmux:**
  `tmux new-session -d -s <name> "bash /tmp/<script>.sh"`. **Do NOT prefix the launch with
  `pkill`** — a `pkill` in the same exec call trips the tunnel's SIGTERM (137/143) and the
  session never starts. Kill stale `infer` in a *separate* call
  (`~/bin/pod 'pkill -9 -f target-pod/release/infer'`), then launch. Poll with short separate
  calls reading the script's output file.
- **Serve / bench:** `scripts/dsv4_toolchain.sh smoke --moe-backend native-deepep
  --expert-backend {deepgemm|native} --deepep-dir <deepep-src> --model-path <models>/DeepSeek-V4-Flash
  --server-bin target-pod/release/infer ...`. For OOB tracing set
  `ARLE_SERVER_WRAP="/usr/local/cuda/bin/compute-sanitizer --tool memcheck --target-processes all"`.
- **mem-fraction-static:** use **0.85** (default) for perf; the toolchain smoke default 0.10
  starves KV and produces false timeouts (§2.0.2).
- **Ship a script to the pod without quoting hell:** base64-encode locally,
  `~/bin/pod "echo <b64> | base64 -d > /tmp/x.sh"`.

---

## 8. CI status (this session)

- `cargo fmt --check`, `cargo check cuda,no-cuda`, `cargo test --release` — **green**.
- `cargo clippy --workspace -- -D warnings` was **red on main (pre-existing)**: fixed 5 trivial
  lints (none in the deepgemm work) — `crates/train/src/opd.rs` ×2, `crates/train/src/qwen35.rs`
  ×1, `infer/src/bin/multiproc_relay_smoke.rs` ×2. Re-verify clippy fully green before handoff
  (whack-a-mole: each fix can expose the next masked crate).

---

## Refs
- `errors/2026-05-27-b335-deepgemm-runtime-crash-h20.md` (full root-cause saga)
- `wins/2026-05-31-bench-dsv4-deepgemm-vs-scalar-prefill.md` (the A/B)
- memories: `feedback_deepgemm_build_flag_stub`, `feedback_jit_error_code_names_surface_not_cause`,
  `project_h20_pod_access`
