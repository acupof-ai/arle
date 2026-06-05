# Build / compile speed optimization — research + plan

**Date:** 2026-06-05. Hypothesis-grade (source inspection; wall-clock deltas need
a GPU measurement per §0). Builds on `2026-06-04-load-and-compile-optimization.md`
§B-build with verified current line numbers (~26-crate split since that doc).

## The two cost regimes — the pain is iteration regime (b)

| Regime | What dominates | Trigger |
|---|---|---|
| **Clean build** | 62 `.cu` serial nvcc (multi-arch) + 31 TileLang configs × 4 SMs = **124 `nvcc -cubin`** + MLX cmake | first build, `cargo clean`, `$OUT_DIR` wipe |
| **Incremental (the pain)** | (a) Rust-only edit → only rustc + relink; (b) **any `csrc/**` or `.cuh` touch → build.rs reruns and rebuilds ALL 62 `.cu` + ALL 124 TileLang cubins from scratch** | every `csrc` touch |

`cargo:rerun-if-changed` is correctly recursive now (`emit_rerun_recursive`,
build.rs:1423 — the old non-recursive-dir bug is fixed) — but it only gates
*whether* build.rs runs; once it runs, the nvcc loop (build.rs:1821) and TileLang
AOT (build.rs:1000) redo everything. **There is no source-keyed cache inside
build.rs.** Touching one widely-included `.cuh` = full 124-cubin rebuild.

## Verified facts (file:line)

- Serial nvcc loop, no parallelism, no cache: `build.rs:1821`.
- TileLang AOT, no cache, once per SM: `build.rs:1000` → `:638` `for sm in sm_targets`; `gen_tilelang_aot.py:535 nvcc_compile_cubin` re-`nvcc -cubin` unconditionally.
- **`ARLE_NVCC_WRAPPER` reaches the 62 native `.cu` (build.rs:1604) but NOT the 124 TileLang cubins** — `gen_tilelang_aot.py:543` uses bare `shutil.which("nvcc")`. So sccache currently misses the bigger half.
- 4 SM targets by default (T1 = sm_80/86/89/90, build.rs:6). Each TileLang config compiles ×4.
- Iteration profile = `--release` (codegen-units=1 + thin-LTO + strip; root Cargo.toml). `release-fast` (lto=false, cu=16, incremental) exists but is opt-in (only `scripts/dsv4_fast_build.sh`).
- No linker override (`.cargo/config.toml` has only the CUDA stub `-L`). No sccache in config/env (only inside `dsv4_fast_build.sh`).
- Prebuilt escape hatch (`ARLE_CUDA_KERNELS_PREBUILT_DIR`, build.rs:1648 → 4.92s link-only) exists but is all-or-nothing + RUSTFLAGS-blind + manual harvest.
- MLX cmake is incremental (cheap no-op recheck); 4 bridge `.cpp` recompile on edit — low priority. `Engine<E,K>` monomorphization is a small fixed set — deprioritize.

## Prioritized levers

| # | Lever | Speedup | Effort | Risk | Pure-config? |
|---|---|---|---|---|---|
| **1** | `RUSTC_WRAPPER=sccache` + `ARLE_NVCC_WRAPPER=sccache` (env or `~/.cargo/config.toml`) — caches rustc + the 62 native `.cu`; survives `cargo clean` / RUSTFLAGS flip / branch switch | large | trivial | very low | yes |
| **2** | **Wire `ARLE_NVCC_WRAPPER` into TileLang's nvcc** (`gen_tilelang_aot.py:543`) — caches the **124 TileLang cubins**, the half sccache misses | very large (clean + post-`.cuh`) | ~10 LOC | low | code, no-op when unset |
| **3** | Narrow `TORCH_CUDA_ARCH_LIST` to the dev GPU (`="9.0"` H20 / `="8.9"` 4090) — 4 SMs → 1 (124→31 cubins) | ~3–4× CUDA half | trivial (env) | ⚠ **correctness — single-SM binary errors on other GPUs; never ship; pod release keeps full T1** | yes |
| **4** | Default `release-fast` for iteration (not `--release`) — kills thin-LTO + cu=1 relink tax | large (Rust relink) | trivial | low (don't bench on it) | yes |
| **5** | Per-(source+SM) content-addressed cubin cache in build.rs, keyed `sha256(src+includes+full nvcc argv+SM+versions)`, `nm`-symbol gate | largest post-`csrc` win | med-high (~200 LOC) | medium (stale-cubin footgun) | extends prebuilt machinery — **but #2 delivers ~80% at ~10% effort; do #2 first** |
| **6** | Parallelize the serial nvcc loop (build.rs:1821, bounded pool ~8) | Ncore× clean half | low-med | low-med (nvcc RAM) | yes |
| **7** | mold/lld linker (`.cargo/config.toml` `-C link-arg=-fuse-ld=mold` Linux / `lld` Mac) | moderate relink | trivial | low | yes |

## Recommended sequence

1. **#1 + #4 + #7 now** (sccache wrapper, `release-fast` iteration default, mold) — trivial config, no correctness risk, attack the Rust-iteration half. Do before measuring.
2. **#3 for local dev** — single dev-SM env. ⚠ document loudly: release/pod must restore full T1.
3. **#2 (TileLang nvcc → sccache)** — the one code change worth doing first; closes the 124-cubin gap. **DONE 2026-06-05** (see commit).
4. **Measure** on the pod: `touch csrc/common.cuh && time cargo build` cold → warm-sccache → RUSTFLAGS flip. Only build **#5** if the per-config Python re-lower (the part sccache can't cache) proves to dominate.

## Correctness flags (do not silently ship)

- **#3** changes which GPUs the binary runs on (single-SM → `CUDA_ERROR_NOT_SUPPORTED` elsewhere). Pod release **must** restore full T1; the `cargo:warning=Compiling CUDA kernels for targets: …` (build.rs:1664) is the guardrail.
- **#5** stale-cubin is the documented footgun (`errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail`); key on full nvcc argv + tilelang/nvcc versions, keep the `nm` gate (build.rs:1537).
- **#4** never bench/SLO on `release-fast` (lto=false changes perf) — final numbers on `--release` per CLAUDE.md.

## The dev-fast iteration recipe (recommended; opt-in, not committed defaults)

`.cargo/config.toml` `rustc-wrapper`/linker are **not** hardcoded in-repo (would
break anyone without sccache/mold, incl. the pod) — set per-box:

```bash
# ~/.cargo/config.toml  (per developer)  OR export in shell:
export RUSTC_WRAPPER=sccache            # #1  (cargo install sccache)
export ARLE_NVCC_WRAPPER=sccache        # #1+#2  (now also caches TileLang cubins)
export TORCH_CUDA_ARCH_LIST=9.0         # #3  dev-only single SM (H20); RESTORE for release
# build/iterate with:
cargo build --profile release-fast      # #4  (NOT for benches)
# linker (#7): add to a personal .cargo/config.toml [target.<triple>]:
#   rustflags = ["-C", "link-arg=-fuse-ld=mold"]   # Linux;  lld on Mac
```
