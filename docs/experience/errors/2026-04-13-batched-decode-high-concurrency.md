# 2026-04-13 · Pre-existing batched-decode regression under concurrency

## Context
During the 2026-04-13 remote CUDA validation batch (see
`docs/plans/tiered-kv-cache-remote-validation.md`), `cargo test --test
greedy_consistency` failed deterministically on L4 (CUDA 13.0, driver
580.82.07) at commit `876b986`:

Historical note (2026-04-21): this entry was later used to justify a
"retract the highest-KV-cost victim" decode heuristic. The active c16
sglang-alignment work moved away from that policy; current code retracts the
least-progressed request first, tie-breaking toward longer prompts. Keep this
file as the rationale for the old policy, not as a claim that the scheduler
may never change again.

- **Solo run** (B=1): ` about a person who is a master of disguise and
  how they use their skills to solve a mystery. Once upon a time, in a
  bustling city filled`
- **Concurrent run** (B=3): ` about a story about a\n\n\n\n\n\n\n\n\n\n
  \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n`

Same prompt, greedy sampling (`temperature=0`), determinism guaranteed
— batched decode emits a totally different (and degenerate) token
sequence than solo. The assertion at
`infer/tests/greedy_consistency.rs:169` fails.

Related symptom: `scripts/bench_throughput_sweep.py` full sweep at
`--num-slots 4` prints non-zero throughput for C=1..4, then **every
C≥8 row prints `0.0 t/s` with `errors=0`**. Server log shows the first
failure is:

```
ERROR infer::scheduler::cuda::decode: decode.rs:256 Batched sampling
      failed: Sync failed: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS,
      "an illegal memory access was encountered")
```

After the first illegal-address, the CUDA context is sticky-bad and
every subsequent request's prefix-cache KV migration also reports
`ERROR infer::scheduler::cuda::prefill:130 prefix KV migration to pool
failed: H2D page_indices failed: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS)`.
Clients see empty 200-OK streams with 0 tokens, which the bench script
counts as zero throughput (not errors).

## Bisection result — not caused by the 2026-04-13 local batch
Identical failure reproduces at **pre-A commit `37a8a82`** (the last
commit before the 2026-04-13 batch). Both solo and concurrent outputs
are byte-for-byte identical to the failing run on `876b986`:

```
solo:       " about a person who is a master of disguise and how they use their skills to solve a mystery. Once upon a time, in a bustling city filled"
concurrent: " about a story about a\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
```

Therefore commits A (`81f5fb0`), B (`5da8b67`), C (`eae8602`),
D (`c531315`), E (`ad4996e`), H (`b45798f`), F (`cf60261`),
bee467e (`bee467e`), G (`997d0b7`), and I (`8adec3c`) **are all
exonerated**. The regression exists somewhere earlier. Most plausible
suspects (from `git log --oneline infer/src/scheduler/`):

- `3e1d35f feat(policy): agent-aware SchedulerSignals + PrefixAwareAdmission`
- `82a19b1 feat(http): thread session_id through protocol → scheduler types`
- `8d9fba3 refactor(runtime): align policy wiring and metal observability`
- `5c5cf91 refactor(workspace): complete phase1 boundary cleanup`
- `a61281d refactor(control-plane): narrow engine facade and seal metal ids`
- `a291145 refactor(scheduler): isolate batch ids and lifecycle events`

None of these were validated against `greedy_consistency` on CUDA
at the time they landed — the test is `#[cfg(feature = "cuda")]`-gated
and the local Mac lane doesn't run it. The pre-existing win doc
`2026-04-09-bench-l4-qwen3-4b.md` already flagged a similar C=8+
CUDA_ERROR_ILLEGAL_ADDRESS under "Known Issue", pinned at "prefix
cache KV migration at high request rate" — likely the same bug.

## Root cause — unknown, investigation deferred
Evidence so far:
1. Greedy output at B=1 is fine; B=3 diverges. Points at batched
   sampling or batched decode kernel input, not the model forward pass
   per se.
2. The degenerate B=3 output is a prompt-independent `\n` token storm
   — strongly suggests the sampling step is reading the wrong address
   and producing the vocab-boundary token that hashes to `\n`.
3. Prefix cache KV migration errors only appear AFTER the sampling
   failure — they are secondary symptoms of the sticky bad CUDA
   context, not the root cause.

Not bisected further in this session; the remote validation task was
to verify A–I, and those are clean.

## 2026-05-06 update — CUDA Graph gibberish fixed, B=3 divergence remains

Re-tested on RTX 4070 Ti SUPER (`sm_89`, CUDA 13.2,
`NVCC_CCBIN=/usr/bin/g++-14`) after the Qwen3 BF16 TileLang decode path was
made default. Two distinct failures were present:

1. With CUDA Graph enabled, B=1 decode became gibberish after roughly the
   first page boundary. Root cause: the TileLang paged attention launch passed
   the current per-batch `total_pages` as a host scalar. CUDA Graph captures
   kernel launch scalars by value, so warmup/capture froze the bound at the
   dummy one-page shape; later replay rejected reads past that captured bound.
   Fix: pass the static KV-pool page capacity and rely on `kv_indptr` for
   per-request bounds.
2. With CUDA Graph disabled, B=1 and B=3 both produce coherent text, but still
   follow different greedy trajectories. Forcing the target request into row 0
   and serializing concurrent prefill did not remove the divergence, pointing
   at batch-size-sensitive decode numerics rather than a row-index or batched
   prefill bookkeeping bug. This matches the numerical-consistency lesson in
   `2026-04-15-e2e-phase3-replay-drift.md`; the exact-equality assertion in
   `greedy_consistency` is still not a valid proof target for the current
   batched TileLang path.

Validation after the graph fix:

- `cargo test --release -p infer --features cuda --test e2e -- --nocapture`
  passes on Qwen3-4B with CUDA Graph enabled.
- `cargo test --release -p infer --features cuda --test greedy_consistency -- --nocapture`
  still fails, but the outputs are coherent B=1/B=3 trajectories rather than
  the previous graph replay gibberish.

## Fix
Partially fixed. The CUDA Graph replay bug is fixed by using a graph-stable
TileLang page-capacity launch scalar. The B=3 exact greedy divergence remains
open as a separate numerical-consistency/test-contract issue.

Candidate investigation steps for the remaining B=3 lane, in order of cost:

1. Re-read `decode.rs:256` batched sampling buffer bookkeeping —
   look for stride mismatch introduced by the
   `typed DecodeContext` / `isolate batch ids and lifecycle events`
   refactors.
2. Disable CUDA Graph capture (`--cuda-graph=false`) and rerun
   `greedy_consistency`. If it passes with graph off, the bug is
   in the replay buffer binding; if it still fails, it is in the
   eager path.
3. `cuda-memcheck` / `compute-sanitizer` the `greedy_consistency`
   test binary to pinpoint the offending kernel.
4. Bisect the 6 suspect commits in order (all compile independently
   because none overlap in the scheduler file set).

## Update — 2026-05-07 — `INFER_DETERMINISTIC=1` fixes B=1, B=3 still wrong

After the graph-bounds fix, the next layer of the bug is the warmup
cublasLt autotune. `autotune_all_cached_gemms_cuda` keys the algo cache
by `(M, N, K)`; B=1 and B=3 GEMMs land on different M and pick different
fp accumulation paths, which is enough to flip greedy argmax across
batch sizes.

`infer/src/scheduler/cuda/core/warmup.rs` now reads `INFER_DETERMINISTIC`;
when set, it skips the autotune step and lets cublasLtMatmulAlgoGetHeuristic
return its default top-1 candidate. With `INFER_DETERMINISTIC=1` the
solo (B=1) `greedy_consistency` output recovers to the HF baseline in
`infer/test_data/Qwen3-4B.json`:

```
" about a young girl who is a talented artist, but she is not allowed
to paint because of her gender. What happens next? Please write in a"
```

This is byte-exact against the `tell_story` baseline. So the autotune
algo selection was the dominant numerical drift source for solo decode.

The concurrent (B=3) trajectory still diverges into the
`" about a young girl named Lila who is a talented painter, …"` lane
(the same wrong-path string also recorded under
[`2026-04-15-e2e-phase3-replay-drift.md`](2026-04-15-e2e-phase3-replay-drift.md)).
That residue is **not** in the autotune step (heuristic-only is in
effect for both runs); it must be in a kernel whose computation is
batch-shape sensitive even with a fixed cublasLt algo — most likely
candidates:

- `decode_prep_paged` batched fused QK-norm + RoPE + paged write —
  per-row in principle, but worth a memcheck pass with B=3 to verify
  no cross-row spill.
- TileLang prefill-as-decode alias kernel
  (`tilelang_batch_prefill_paged_hd128_q32_kv8`) — grid `(1, num_q_heads, B)`,
  per-block independent in the kernel source, but `total_q_tokens` /
  `qo_indptr` change with B and feed the kernel scalar args; if any
  block-shared shmem region or atomic counter is keyed on those, the
  per-row arithmetic could subtly differ.
- The `cublasLtMatmulAlgoGetHeuristic` top-1 itself is M-dependent —
  for `(N=2560, K=2560)` it may choose a different algo for M=1 vs
  M=3 even with autotune off. Pinning a single algo across M (e.g.
  via a custom shape-key normalizer in
  `crates/cuda-kernels/csrc/gemm/gemv.cu::GemmKey`) is the next concrete
  knob to try if `decode_prep_paged` and the TileLang alias come up
  clean under compute-sanitizer.

## Rule
- Never run `cargo test --release --test greedy_consistency` on L4 in
  isolation and trust "it passes" from an older win doc; the test was
  last green before the scheduler refactor storm (pre-`3e1d35f`).
- When adding a new local-lane PR that touches
  `infer/src/scheduler/cuda/*` or `infer/src/model/qwen3/batch_decode.rs`,
  the remote validation checklist MUST include
  `cargo test --release --test greedy_consistency` and
  `scripts/bench_throughput_sweep.py` — both are the only gate that
  exercises the B≥3 numerical path.
- When running `bench_throughput_sweep.py` on the remote box, restart
  the server between sweeps. Once CUDA gets stuck in the illegal-
  address state, every subsequent request returns empty 200 streams,
  and the bench summary underreports throughput silently.
- Until this is fixed, every benchmark that needs C≥8 data must carry
  a banner pointing at this file.
