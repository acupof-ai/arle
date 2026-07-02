# Carried tvm-ffi 0.1.12 compat patch LICENSED — full stack upgrade, no ceiling

## Context

Dependabot #121's `apache-tvm-ffi` 0.1.11→0.1.12 bump hard-aborted
`import tilelang` (tilelang 0.1.11 is the latest release; upstream
tile-ai/tilelang#2367's only fix is pinning ≤0.1.11). ckl's direction: upgrade
the stack coherently and carry the compatibility patch ourselves — no version
ceiling. Two independent incompatibility surfaces were decoded case-by-case:

1. **C++**: tvm-ffi ≥0.1.12 registers builtin TypeAttr entries the bundled TVM
   of tilelang ≤0.1.11 still registers at load → duplicate-registration throw
   during dlopen (std::terminate, unreachable from Python).
2. **Python**: old TVM names DictAttrs' storage field literally `__dict__`;
   0.1.12 changed `_add_class_attrs`' guard from `hasattr` (0.1.11 — every
   class has `__dict__`, field skipped) to own-attrs-only → setattr on a
   type-unwritable name.

## What Worked

`crates/cuda-kernels/tools/tilelang/patches/tvm-ffi-legacy-tvm-compat.patch`
(2 hunks): first-wins skip for duplicate TypeAttr registration (the ffi
registration stays authoritative) + skip `__dict__`/`__weakref__` field names
(exact 0.1.11 behavior parity for legacy types). Applied by
`scripts/pod-tilelang-env.sh` (idempotent: gate = venv `import tilelang` +
installed tvm-ffi == the pin parsed from requirements-build.txt; provisioning
and patch-rebuild are separate steps so a version mismatch never re-pulls the
multi-GB torch chain). Kill condition was pre-declared: a third surface would
have reverted to the ≤0.1.11 pin — none appeared.

Pod verification (365796d284, GPU 1, run-patchgate-toy1r):

| gate | result |
|---|---|
| import | `tilelang 0.1.11 tvm-ffi 0.1.12 patched-ok` |
| full TileLang AOT regen + build | `BUILD_EXIT=0 (compiled 8 crates)`, zero regen errors |
| toy round vs run-sdpatrace-toy1r | forward 3.806s (vs 3.768s), backward 26.281s (vs 26.252s) — noise; fused=TAKEN 32/0; RUN_EXIT=0; mean_loss 0.3228 in band |

## Rule

- A dependency-set break is patched forward, not pinned back: decode each
  failure surface in a clean env, locate the raise point in upstream source,
  carry the minimal hunk, and pre-declare the kill condition (N+1th surface =
  revert with evidence).
- Upstream declared-compatibility (`requires_dist` ranges) is hypothesis, not
  fact — tilelang's own envelope admits the tvm-ffi version that aborts it.
- Drop the patch when a tilelang release imports cleanly against ≥0.1.12.
