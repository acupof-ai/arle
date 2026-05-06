# 2026-05-07 · 04-13 deferred bug closed + M1 telemetry landed + 3 strategic plans

## Goal
Close the long-deferred batched-decode B=1/B=3 greedy divergence
([`2026-04-13-batched-decode-high-concurrency.md`](../errors/2026-04-13-batched-decode-high-concurrency.md))
and land enough scaffolding to start the multi-week "world-#1 inference
runtime" push.

## Hypothesis
The 04-13 entry framed the bug as one phenomenon. We expected to find
one root cause; in practice it peeled into three layered issues
(graph-replay scalar staleness → autotune algo M-dependence → cuBLAS
internal kernel selection M-dependence), each requiring its own fix
before the layer below was visible.

## What worked

Nine commits landed in one session, split by ownership track:

| Commit | Track | Owner | Effect |
|---|---|---|---|
| `a2428af` | bug fix | Claude | `complete_stream` blocks until finish marker (e2e Phase 3 race) |
| `3f4c1b7` | build | Claude | `NVCC_CCBIN` env unblocks GCC≥16 hosts (CachyOS / Arch) |
| `4ddfa5d` | bug fix | Codex | TileLang decode `total_pages` uses static pool capacity (graph-stable scalar) |
| `9833068` | bug fix | Claude | `INFER_DETERMINISTIC=1` skips cublasLt autotune, B=1 matches HF baseline |
| `623e9ec` | plan | Claude | Long-ctx × spec-decode × TileLang × Tier-KV combo plan (M_a..M_e) |
| `bb648fe` | bug fix | Codex | Deterministic mode routes BF16 dense GEMM through per-row N=1 graph-safe call → B=1≡B=3 byte-exact |
| `ec54340` | plan | Codex | Backend-unification roadmap (M1-M7, 12 weeks) |
| `4001681` | plan | Codex | DSv4 small-scale full-method repro (1B from-scratch on 1×4070 Ti SUPER) |
| `92b5ba9` | M1 | Codex | Unified backend telemetry trait across CUDA + Metal |

## Final test state (RTX 4070 Ti SUPER, sm_89, CUDA 13.2, gcc-14 ccbin)

- `cargo test --release --workspace` — ✅ 53 passed (CPU)
- `cargo test --release -p infer --features cuda --test e2e` — ✅ all 4 phases pass
- `INFER_DETERMINISTIC=1 cargo test --release -p infer --features cuda --test greedy_consistency` — ✅ B=1 ≡ B=3 byte-exact, matches HF baseline
- `cargo clippy -p infer --features cuda -- -D warnings` — ✅ clean

## The bug peeling, in order

1. **Graph-replay scalar staleness** (fixed in `4ddfa5d`). TileLang paged
   attention took `total_pages` as a host scalar; CUDA Graph captured
   the warmup-time value (1 page from 1 dummy token) and replays
   silently rejected `KV_indices[i]` reads past that bound, gibberish
   after the first decode page boundary.
2. **Autotune algo cache keyed by `(M, N, K)`** (mitigated in `9833068`,
   superseded by `bb648fe`). With autotune on, B=1 and B=3 GEMMs land
   on different M, the autotune picks different algorithms with
   different fp accumulation paths, and greedy argmax flips. Skipping
   autotune restored B=1 to the HF baseline but left B=3 wrong.
3. **cuBLAS internal kernel selection still M-dependent** (fixed in
   `bb648fe`). Even with autotune off, `cublasLtMatmulAlgoGetHeuristic`
   and `cublasGemmEx` both pick different internal kernels for M=1 vs
   M=3, giving different per-row outputs. Resolution: in deterministic
   mode, route BF16 dense GEMM through the per-row N=1 graph-safe path
   so the kernel sees M=1 regardless of the actual batch — at the cost
   of B× kernel launches per layer. Production default keeps the fast
   batched path (autotune on).

## What we learned

- **One assertion failure can hide a stack of independent bugs.** The
  04-13 entry was framed as one issue. Bisecting *symptom* (gibberish
  vs coherent-but-different) at each fix exposed the next layer. Don't
  declare a bug "fixed" until the *next* failure mode also resolves.
- **CUDA Graph capture freezes scalar kernel arguments.** Anything
  that varies per call must be either device-side (read from a
  pointer) or made invariant via a static upper bound.
- **cuBLAS heuristic + cublasGemmEx are both M-dependent for batch
  invariance.** Algo selection by shape-key flips kernels at small M,
  fp drift cascades into greedy. The only kernel-determinism guarantee
  across batch sizes is "always launch with the same M".
- **Cooperative multi-agent pattern works for high-leverage parallel
  tracks.** Three parallel plans (backend-unification, DSv4 substrate,
  long-ctx combo) authored in one session by Codex + Claude with
  explicit ownership split. Each commits to its own files; shared
  paths reviewed via `codex review --uncommitted` before push.

## Rule

- When `greedy_consistency` (or any "B=1≡B>1" assertion) fails, peel
  in this order: (1) graph-stable launch scalars → (2) M-keyed algo
  cache / autotune → (3) cuBLAS kernel selection. Each layer can hide
  the next.
- Production paths keep the fast batched GEMM. The deterministic
  per-row N=1 fallback is only correct as a `INFER_DETERMINISTIC=1`
  diagnostic / parity proof; **do not** remove the fast path.
- Keep `INFER_DETERMINISTIC=1` available for any future correctness
  test that compares solo and concurrent decode trajectories — it is
  now the contract for byte-exact greedy across batch composition on
  CUDA.

## Bench
**pending-remote.** Recorded under
`docs/experience/wins/2026-05-07-cuda-greedy-consistency-deterministic-gemm.md`
(per-row N=1 path is slower; production default is unaffected, but the
delta should be measured on the next remote-host bench pass).

## Next moves (manager note)

- **Codex track**: M2 of `backend-unification.md` (Metal kv-tier policy
  adapter), 1 week budget.
- **Claude track**: M_a of `longctx-spec-tilelang-combo.md` (spec-decode
  benchmark harness), independent of unification, runs in parallel.
- **Joint review gate**: any change touching both tracks goes through
  `codex review --uncommitted` before push (this session's clippy bugs
  in M1 paths show review > self-attest).
