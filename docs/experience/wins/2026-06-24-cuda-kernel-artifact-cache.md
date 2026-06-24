# Persistent hash-keyed CUDA kernel-artifact cache — from-scratch build 251s→53s (pod)

## Context

Tranche 3 of the kernel-pipeline redesign dropped ALL committed `generated/` artifacts
(the 93 per-SM `.c`, ~22.6M) and gitignored the dir
([plan](../../plans/2026-06-23-cuda-kernel-pipeline-redesign.md)). That traded git
bloat for build time: every clean/CI build now re-runs TileLang + nvcc instead of
consuming a committed `.c`. Measured the cost, then added a cache to recover it.
Commits `4069acc7` (cache) + the audit-driven hardening (this entry).

## Profiling — where the from-scratch build time goes (8×H20, timestamped phases)

`cargo clean` → full `--features cuda,nccl,deepep --bin arle`, registry/sccache warm.
**Total 133s.** Phase attribution (markers in the build log):

| Phase | Time | % |
|---|---|---|
| cuda-kernels: **TileLang AOT regen + per-kernel nvcc** | **59s** | 44% |
| cuda-kernels: DeepGEMM cutlass + deepep sidecar + cc/link | ~21s | 16% |
| downstream crates (infer-server/cli/autograd/agent-infer) + final link | 48s | 36% |
| deps → cuda-kernels start | 5s | — |

**Bottleneck = cuda-kernels build.rs (~80s, 60%), dominated by the TileLang AOT
regen+nvcc (59s)** — the part the committed `.c` used to skip. sccache caches rustc,
NOT nvcc, so the kernel regen repeats on every clean build.

## What Worked (pod A/B, 8×H20)

Persistent cache OUTSIDE `target/` (survives `cargo clean` + fresh clones), keyed on
`<src_hash>-<nvcc_tag>`. build.rs checks before regen, stores after.

- **Forced full regen 251s → warm-cache restore 53s (−79%)**, 31 kernel entries
  stored. The 53s residual is cc-compiling the restored `.c` + crate + link (not
  cacheable here). `ARLE_TILELANG_REGEN=1` for run1 (gates out restore); run2's env
  flip re-runs build.rs → marker-gated restore.
- Pod cache lives on `/host` (container `$HOME` is ephemeral), wired via
  `pod-build-env.sh` `ARLE_CUDA_KERNEL_CACHE_DIR`. Disable: `ARLE_CUDA_KERNEL_CACHE=0`.

## Hardening (adversarial audit `session-gap-audit`, all fixed pre-merge)

The first cut had three real holes; the audit caught them before they bit:

- **Key completeness**: `src_hash` hashed the kernel `.py` + params + SM but NOT the
  `kernels.toml` `[abi.*]` decls (an ABI-only edit → cached `.c` with the old
  signature → silent linker-accepts UB) NOR the tilelang library version (a tilelang
  bump with byte-identical `.py` → stale cubin). Fix: fold `public_decl`/`extern_decl`/
  `call_args` + the `requirements-build.txt` tilelang pin into the key
  (+`rerun-if-changed` on the pin).
- **Store atomicity**: the store did `remove_dir_all(entry)` then a non-atomic
  recursive copy into the live entry — on the pod's parallel-build model (shared
  `/host` cache) a concurrent restore could read a truncated `.c` (cubin embedded) →
  silent corrupt kernel; a killed store left a permanently-poisoned partial entry.
  Fix: stage in a per-pid temp dir, write a `.complete` marker LAST, atomic `rename`
  into place; **restore requires the marker, not just `.c`.is_file()** → a torn/
  killed/concurrent store reads as a miss (redundant regen), never corruption.

## Rule

- **Dropping committed build artifacts trades git size for build time — measure the
  regen cost and cache it before declaring the deletion a win.** The cache is the
  other half of "get the cubins out of git"; without it a clean build pays the full
  regen every time.
- **A build cache shared across parallel builds needs atomic publish (temp + marker-
  last + rename) and marker-gated reads.** `is_file()` on the payload is not enough —
  a mid-copy `.c` exists but is truncated; the embedded cubin makes that a silent
  wrong-kernel, not a build error.
- **A content-hash cache key must hash EVERYTHING that changes the output** — not just
  the obvious source file. The ABI decls (from `kernels.toml`) and the compiler/
  library versions (nvcc, tilelang) all change the cubin; omitting any is silent reuse
  of a stale artifact.
