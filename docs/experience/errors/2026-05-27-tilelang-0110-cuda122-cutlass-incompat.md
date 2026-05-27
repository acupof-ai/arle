# TileLang 0.1.10 + CUDA 12.2 — cutlass C++20 fold-expr break

## Context

Setting up a fresh pod (8 × H20, CUDA 12.2.140 / V12.2.r12.2) to verify
the B-3.3 native-deepep forward path. After pushing arle source via
`tn push` + cloning DeepEP and building deepep-sys cleanly, the full
`cargo build -p infer --features cuda,nccl --lib` failed during
TileLang AOT codegen with 30 nvcc errors from the bundled cutlass:

```
/root/tl-venv/lib/python3.10/site-packages/tilelang/3rdparty/cutlass/
  include/cute/util/type_traits.hpp(315): error: expected a "}"
    constexpr static bool value = (... || std::is_same_v<T, Us>);
                                                                ^
```

This is a C++17 unary fold expression. nvcc is being invoked with
`-arch=sm_90a -std=c++20`, but CUDA 12.2's nvcc doesn't parse the
fold-expression form in the device-code path correctly when cutlass
bumps to its newer cute headers.

## Root Cause

TileLang 0.1.10 ships a cutlass snapshot (cute v3+) that uses fold
expressions in device code. CUDA 12.2's nvcc parser handles these in
host code but rejects them in `--cubin` device-code compilation, even
under `-std=c++20`. The combination is broken upstream — needs CUDA
12.3+ to compose with this cutlass version.

## Fix

Pin TileLang to **0.1.9** in the venv used for AOT codegen:

```bash
/root/.local/bin/uv pip install --python /root/tl-venv/bin/python \
    --offline tilelang==0.1.9
```

0.1.9 ships an older cutlass snapshot that uses traits-style code
which nvcc 12.2 accepts. After the downgrade, the same TileLang AOT
build path succeeds.

The pyproject `optional-dependencies.tilelang = ["tilelang>=0.1"]`
spec is overly permissive — until CUDA 12.3+ is the floor or
TileLang upstream fixes the cute-cutlass-device issue, the safe pin
on CUDA-12.2-only environments is `tilelang>=0.1,<0.1.10`.

## Rule

When ARLE's CUDA hot-path build hits a tilelang nvcc parser error,
**don't reach for `-std=c++17` workarounds first** — the upstream is
calling for C++20, and the error is a CUDA-version mismatch with the
cutlass version that tilelang bundles. Verify the failing TileLang
version against the pod's nvcc version before any other diagnosis:

| Pod nvcc | Known-good TileLang range |
|---|---|
| CUDA 12.2 (V12.2.r12.2) | `>=0.1,<0.1.10` |
| CUDA 12.3+ | `>=0.1` (incl. 0.1.10) — verify on first use |

Cross-check `docs/plans/sm-coverage.md` if SM 7.0 is also in scope —
the SM-tier policy already documents the legacy-Volta route.
