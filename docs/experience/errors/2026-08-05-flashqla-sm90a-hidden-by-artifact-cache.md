# FlashQLA never compiled on this box — an artifact cache hid a broken build for three days

**Date:** 2026-08-05 · **Fix:** `build.rs:1289` target `sm_90` → `sm_90a` for the flashqla family

## Context

The FlashQLA GDN backward port (`4846f8046`) went to the pod for its first real
compile. TileLang AOT hard-failed:

```
TileLang AOT failed to compile gdr_fq_kkt for sm_90.
ptxas ... error : Instruction 'setmaxnreg.inc' not supported on .target 'sm_90'
ptxas ... error : Instruction 'setmaxnreg.dec' not supported on .target 'sm_90'
Command: nvcc --cubin -O3 -arch=sm_90 ...
```

`gdr_fq_kkt` is not one of the ported kernels. It is the forward K·Kᵀ helper that
landed 2026-08-02 with a licensed −27% inference win.

## Root cause

`setmaxnreg` is the WGMMA warp-specialization register-realloc instruction; ptxas
accepts it only under the `sm_90a` variant. `build.rs:1289` built every TileLang
target as `format!("cuda -arch=sm_{sm_token}")` → plain `sm_90`.

So the kernel could never have compiled. It didn't have to: `tilelang_kernel_src_hash`
folds `flashqla_gdr.py`, `kernels.toml`, `build.rs` and `requirements-build.txt`
into the artifact identity, and the AOT step reuses a matching artifact. Every
build after the first hit that cache. The port changed `flashqla_gdr.py` **and**
`kernels.toml`, which invalidated the identity for every flashqla row — including
the three forward kernels that had been "fine" — and forced the first cold compile
this family has ever had on this box.

Confirmed by bisect, not inferred: `e675f031b`, the commit before any of this
session's work, built with `ARLE_CUDA_ENABLE_FLASHQLA_GDR=1` in a separate private
tree, fails **byte-identically** — same kernel, same two instructions, same target.
Same shared preserved venv, tilelang 0.1.12, matching the `requirements-build.txt`
pin.

The pin's own comment says "pod-verified 2026-07-23 (sm_90 AOT regen, BUILD_EXIT=0)".
That verification predates FlashQLA landing on 08-02 by ten days, so it never
covered these kernels.

## Fix

Scope `sm_90a` to the flashqla family in `build_tilelang_kernel`, mirroring the
precedent already in the same file (`build.rs:2808-2817` forces sm_90a-only on the
FlashMLA and FA3 TUs for the same WGMMA reason). Artifact directory and symbol
names keep the `sm90` token; only the compile target moves, so dispatch and the
runtime gate are unchanged. Other TileLang families keep plain `sm_90` — they
compile today and are measured; widening the change would put verified kernels
back through codegen for no reason.

Note this edit invalidates every TileLang artifact, since `build.rs` is in the
identity. The next pod build is a full cold AOT regen.

## Rule

A content-addressed artifact cache converts "this never built" into "this builds
fine" for as long as nobody perturbs the key. Two consequences:

- **A green build is evidence about the cache, not about the compiler**, unless
  something forced a cold path. When a kernel family lands, force one cold build
  before recording the win — otherwise the first unrelated edit to any file in the
  identity inherits a failure it did not cause.
- **Bisect before attributing a build break.** The obvious reading here was that
  the port broke the build; it added the only `set_max_nreg` calls in the diff. A
  control build at the pre-change commit cost one build slot and moved the defect
  three days earlier and into a different file.

See [`reference_pod_cuda_build_needs_tilelang_venv`], and the wins entry for the
port itself, `2026-08-05-flashqla-gdn-backward-training-route.md`.
