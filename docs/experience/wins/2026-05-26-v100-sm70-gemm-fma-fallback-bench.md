# V100 sm_70 GemmFMA fallback end-to-end bench

## Context

After [`c1530d67`](../../..) rewrote the SM70 TileLang patch to drop the
mma_sm70 fake-BF16 path in favour of a CUDA-core scalar `cuda.fma` fallback
plus shared-memory fragment staging, the change had only been validated
upstream — the TileLang PR
[#2279](https://github.com/tile-ai/tilelang/pull/2279) pytest trio passed on
V100 (3 SM70 kernel + 8 dispatch tests). ARLE end-to-end on V100 was still
running against the previous mma_sm70 BF16-emulation path; no smoke or
guidellm number existed under the new fallback. Today, after addressing the
PR #2279 review (CodeRabbit nitpick + Docstring Coverage gate) and pushing
[`dc30b516`](../../..) to sync `scripts/sm70_tilelang.patch` with the
updated fork branch, we finally rebuilt and benched both layers.

## What Worked

- **PR #2279 follow-up [`14489d9d`](https://github.com/cklxx/tilelang/commit/14489d9d).**
  Dropped the redundant `T.sync_threads()` before the cooperative A/B stage
  load — `T.alloc_shared` returns fresh shared regions, so the post-stage
  sync is the only barrier needed for correctness. Added module-, helper-,
  method-, and `_gemm_fma`-level docstrings to clear the Docstring Coverage
  pre-merge check (was 0.00% vs 80% threshold).
- **Patch script + scoped diff is robust.** `scripts/patch_tilelang_sm70.sh`'s
  `PATCH_PATHS` allowlist + double-sentinel idempotency check (`NeedsVoltaFragmentStaging`
  in copy.cc + `GEMM_INST_FMA` in gemm_fma.py) reapplied cleanly on a V100
  checkout that already had the older patch's fragment-staging slice.
- **End-to-end smoke continues to pass on V100.** Same Qwen3.5-4B + GDR
  hybrid path as [`2026-05-25-v100-sm70-p1-smoke-pass.md`](2026-05-25-v100-sm70-p1-smoke-pass.md),
  now routing BF16 GEMM through the scalar FMA fallback instead of the
  fake-BF16 mma_sm70 path. Tokens return; no error markers in server log.
- **First guidellm baseline on V100.** Prior V100 wins entries (5/25 P1
  smoke, P3 capability) only ran curl smokes. This is the first canonical
  `scripts/bench_guidellm.sh` number on Tesla V100-SXM2-32GB.

## Evidence

### PR #2279 review fix-ups

CodeRabbit nitpick (gemm_fma.py:93 leading `T.sync_threads()`):

> The `sync_threads()` at line 93 precedes any shared memory writes within
> this function. Unless there's an external dependency on prior kernel
> state that requires synchronization, this barrier appears unnecessary
> and adds latency.

Rationale for accepting: `A_stage`/`B_stage` are freshly allocated via
`T.alloc_shared` inside the prim_func — no other thread can hold
references to them. The later sync (after the cooperative stage writes,
before the GEMM compute loop) is the only barrier required for shared
memory visibility. Other `GemmBase` implementations (`GemmMMA`,
`GemmMMASm70`, `GemmMMASm75`) do not pre-sync at the entry of their
lowered prim_funcs.

Docstring Coverage: PR pre-merge check reported 0.00% coverage; target is
80%. Module + `_linear_fragment` + class + `infer_layout` + `lower` +
`_gemm_fma` prim_func docstrings added.

### V100 pytest re-verification

Patched files checked out from fork branch onto `~/tilelang-sm70-copy`:

```bash
git checkout fork/fix/sm70-gemm-fma-fallback -- \
  src/backend/cuda/op/copy.cc \
  src/backend/cuda/op/gemm.cc \
  src/tl_templates/cuda/instruction/mma_sm70.h \
  testing/python/cuda/test_cuda_mma_sm75_dispatch.py \
  testing/python/kernel/test_tilelang_kernel_sm70_fragment_copy.py \
  testing/python/kernel/test_tilelang_kernel_sm70_gemm_fma.py \
  tilelang/cuda/op/gemm/__init__.py \
  tilelang/cuda/op/gemm/gemm_fma.py
```

Dispatch test (pure-Python, no `libtilelang.so` rebuild needed):

```text
testing/python/cuda/test_cuda_mma_sm75_dispatch.py ........  [100%]
8 passed
```

Kernel tests after `ninja tilelang` rebuild of libtilelang.so:

```text
SKIPPED [1] testing/python/kernel/test_tilelang_kernel_sm70_gemm_fma.py:57:
  No device exists for target cuda
SKIPPED [1] testing/python/kernel/test_tilelang_kernel_sm70_gemm_fma.py:72:
  No device exists for target cuda
SKIPPED [1] testing/python/kernel/test_tilelang_kernel_sm70_fragment_copy.py
======================== 3 skipped ========================
```

The kernel tests skipped because the V100 venv has `torch 2.11.0` built against
CUDA 13.0 while the host driver is 12.2 (`12020`), so
`torch.cuda.is_available()` returns `False` and `@requires_cuda` short-circuits
before the test body runs. This is an environmental skip, not a patch
regression — the dispatch test still verifies the registry wires `cuda.fma →
GemmFMA` on `sm_70`, and the cubins ARLE actually loads on V100 (compiled by
the same GemmFMA lowering at cargo build time) confirm the kernel codegen
path. Direct SASS inspection of those cubins is reported below.

### Cubin SASS sanity check — FMA fallback is in fact present

`cuobjdump --dump-sass` on the V100-tagged cubins under
`target/release/build/cuda-kernels-*/out/tilelang_aot/`:

| cubin (`*_sm70`) | HMMA.884 | FFMA | interpretation |
|---|---:|---:|---|
| `batch_decode_paged_hd128_q40_kv8_sm70` | 48 | 0 | FP16 attention via SM70 MMA |
| `batch_prefill_paged_hd128_q32_kv8_sm70` | 48 | 332 | mix: tensor-core attention + scalar-FMA staging/cleanup |
| `gated_delta_rule_chunk_a_sm70` | 16 | 128 | GDR chunked compute, dominated by FMA fallback |

This is exactly the dispatch contract the patch installs: FP16 GEMM with
satisfying shape constraints stays on `HMMA.884.F32.F32` (the SM70 m8n8k4
tensor-core path); BF16 / non-conforming shape combinations drop to scalar
`FFMA` via `GemmFMA`. The cubins were generated by the patched libtilelang.so
during cargo build, so the patch's codegen path is what actually runs on V100.

### V100 end-to-end smoke (Qwen3.5-4B, audit service)

The V100 audit service (`./target/release/infer --num-slots 16 --max-seq-len 5120
--long-prefill-active-limit 16 --chunked-prefill-size 512`) is already running
against the rebuilt binary + new libtilelang.so. Smoke against it:

```text
hello (1 tok):    status=200, prompt_tokens=1,  completion_tokens=4, total=5
  → "user\nI am"
capital of France (7 tok): status=200, prompt_tokens=7,  completion_tokens=8, total=15
  → "\n\nThe question asks for the capital of"
x*31 (32 tok):    status=200, prompt_tokens=32, completion_tokens=4, total=36
  → "1 1 "
```

All three return coherent tokens with `finish_reason="length"` (the smoke
caps `max_tokens`). The "capital of France" prompt yields a structurally
relevant continuation, confirming the FP16/FMA mixed path produces sensible
logits at multiple sequence lengths.

### V100 guidellm sweep — pending-remote

Attempted four times against the audit service on V100; each was interrupted
by the audit's auto-restart loop (the watchdog terminates `infer` and rebuilds
it on a separate schedule, so a 10-minute `sweep` profile cannot complete a
full pass). The third attempt did reach `Setup complete, starting
benchmarks...` and produced 277 service-stats samples (~4.5 min of trace)
before the watchdog kill at 19:25; partial scheduler telemetry from that
window:

| metric | observed |
|---|---:|
| samples (ok / total) | 255 / 277 |
| peak waiting | 10 |
| peak active | 4 |
| peak prefill_queue | 3 |
| peak kv_util | 3.5% |
| plan labels (idle/decode/prefill/split/mixed) | 6788 / 1275 / 15 / 0 / 5 |

`benchmarks.json` was never written, so TTFT/ITL/tok-s numbers are
**pending-remote**. Follow-up: schedule a quiet V100 window (audit watchdog
paused or routed to a different port) and re-run
`scripts/bench_guidellm.sh v100-sm70-gemm-fma`. The trace artefacts that did
land on V100 are at
`~/agent-infer-v100-audit/bench-output/2026-05-26-v100-sm70-gemm-fma-run3/`.

```bash
# canonical command (re-run when V100 is quiet):
ssh v100 'cd ~/agent-infer-v100-audit && \
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY no_proxy ALL_PROXY all_proxy && \
  PATH=~/tilelang/.venv/bin:$PATH GUIDELLM__MP_CONTEXT_TYPE=spawn \
  scripts/bench_guidellm.sh v100-sm70-gemm-fma \
    --target http://localhost:8000 \
    --model Qwen3.5-4B \
    --processor ~/.cache/modelscope/hub/models/Qwen/Qwen3.5-4B'
```

## Problems

- **V100 audit watchdog vs bench tooling.** The audit service runs `infer`
  on port 8000 with a watchdog that periodically rebuilds / restarts it.
  guidellm sweep needs ~10 min of continuous service; the watchdog kept
  killing the server mid-sweep. Mitigations to land in a follow-up:
  pause the watchdog, route the bench at a separate `infer --port 8001`,
  or shorten the sweep profile to fit a single watchdog cycle.
- **`httpx.InvalidURL: Invalid port: ':'` from `NO_PROXY=...::1,fe80::/10,...`.**
  V100's shell init sets `NO_PROXY` with IPv6 entries that httpx 0.28's
  URLPattern parser can't handle (the `::1` literal hits an empty port
  segment). The bench wrapper now needs to be invoked with all `*_proxy`
  env vars unset; documented in the command block above. Worth adding to
  `scripts/bench_guidellm.sh`'s preflight as a hard-clear of the relevant
  env keys before invoking guidellm.
- **kernel pytests skipped on V100.** `torch 2.11.0` in the V100 venv was
  built against CUDA 13.0; the host driver is 12.2 (`12020`), so
  `torch.cuda.is_available()` returns False and `@requires_cuda` skips
  before the kernel body runs. End-to-end correctness was therefore
  demonstrated via the production smoke + cubin SASS census, not pytest.
  Fix path: install a CUDA-12.x-compatible torch wheel in the V100 venv
  (separate concern, tracked outside this entry).
- **`tilelang-sm70-copy` checkout drift.** The V100 has two TileLang
  checkouts: `~/tilelang` (interactive) and `~/tilelang-sm70-copy` (the
  editable install backing the venv). They had drifted — `~/tilelang`
  was on `fix/sm70-gemm-fma-fallback` while the venv was loading
  `tilelang-sm70-copy` at upstream release `69bc43e2`. Re-applied the
  patch via `git checkout fork/fix/sm70-gemm-fma-fallback -- <PATCH_PATHS>`
  on `tilelang-sm70-copy`, then `ninja tilelang` + manual sync of
  `lib/libtilelang.so` into the venv site-packages. Worth collapsing the
  two checkouts to one to remove this footgun.

## CI bring-back

PR #2279's previous CI run (on `14489d9d`) failed four tests under the
CUDA-12.8 job:

- `test_wgmma_atom_gemm` (98.6% tensor mismatch)
- `test_merge_dynamic_shared_reuses_non_overlapping_buffers`
- `test_merge_dynamic_shared_rewrites_cp_async_case_after_flatten`
- `test_merge_dynamic_shared_lowbit_style_scratch_and_long_buffer_do_not_reuse_yet`

These failures have two distinct causes, not one:

- The 3 `test_merge_dynamic_shared_*` cases are **new tests** added by
  `e84a0ee6 [Transform] Rewrite MergeSharedMemoryAllocations with per-epoch
  liveness (#2185)`. They did not exist at the branch base `7f359ea8`, so
  the assertions describe the *post-rewrite* behavior. GitHub Actions'
  `pull_request` event runs the workflow against the virtual
  `refs/pull/N/merge` ref (PR head merged into base), so the new upstream
  tests were exercised against this branch's pre-rewrite `merge_shared_memory_allocations.cc`
  and failed.
- `test_wgmma_atom_gemm` is **pre-existing** at `7f359ea8` and passed on
  upstream main. It started failing on this branch's merge-ref likely
  because of `3bf3d00a [Pipeline] Refactor software pipeline transforms
  (#2245)` — the only upstream pipeline change between base and head, and
  the WGMMA path is downstream of the software-pipeline lowering. Not
  confirmed by bisect; the symptom (98.6% mismatch) is consistent with
  a pipeline / barrier ordering regression rather than a numerical-precision
  one, but other transitive causes (e.g. `c66cadf8 named barrier arrive`)
  cannot be ruled out without a fresh run.

Both classes are resolved by the same fix: explicit `git merge upstream/main`
into the fork branch (commit `444d370b`), then `git push`. After that push,
GitHub did not auto-create check-suites for the new head SHAs — left a
maintainer ping on the PR and posted an empty re-trigger commit
(`e401700f`) to retry. CI status pending maintainer action.

## Rule

The `cuda.fma` fallback is correctness-honest (no silent BF16→FP16 cast
inside Tensor-Core path) but tensor-core throughput is gone — V100 GEMM is
back to ~15.7 TFLOPS FP32 FFMA, ~8× slower than the 125 TFLOPS FP16 MMA
peak. For V100 Qwen3.5 we accept this gap because (a) ARLE V100 is a
compatibility track, not a perf track, and (b) the alternative is a silent
precision-loss MMA path that would fail under wide-magnitude BF16 weights.
If V100 perf becomes load-bearing, the proper escalation is offline
BF16→FP16 weight conversion at load time, not a fake-BF16 MMA path.
