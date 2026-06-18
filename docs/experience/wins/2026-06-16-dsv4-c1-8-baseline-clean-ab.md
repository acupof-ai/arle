# DSv4 8×H20 decode c1–8 — CLEAN baseline + compressor-batch A/B (profiling OFF)

Supersedes the c1–8 table in
[[2026-06-16-dsv4-c1-8-baseline-snapshot]], whose numbers were measured with
`ARLE_DSV4_DECODE_PHASE_TIME=1 + ARLE_DSV4_LINEAR_PROFILE=1` (the
`serve_bench_62.sh` launcher) — each step pays a `cudaStreamSynchronize`, which
**understated tok/s ~25–35%** (the "c1=31.9" was the artifact ckl flagged). This
entry re-measures with **profiling OFF** and adds a same-session gate OFF→ON A/B.

## Goal
Fix the contaminated c1–8 baseline and produce the standing clean anchor for the
next lever (compute/comm overlap on the batched path). "baseline" = current-best
all-non-MTP-opts-ON config ([[feedback_baseline_means_current_best_all_opts_on]]).

## Config
.62 (192.168.12.62), 8×H20 97GB, CUDA 12.9, glibc 2.28. Model
`/data01/models/DeepSeek-V4-Flash`. Binary origin/main@`3e3e50e0` + DSv4
(`/data01/arle-build`). deepgemm JIT cache warm (`DG_JIT_CACHE_DIR=/data01/deepgemm-warm`,
[[reference_dsv4_deepgemm_jit_cache_persist_62]]). nccl-cu12 2.27, clang-11 deepgemm-JIT.

- TP=8, `num-slots 64`, `max-total-tokens 4096`, `chunked_prefill_size 64`.
- `MOE_BACKEND=allreduce`, `EXPERT_BACKEND=deepgemm` (native), `INCREMENTAL_KV=1`,
  `FUSED_DISPATCH_PAYLOAD=1`, batched-FlashMLA decode (default-on c≥4), fused-wqkv,
  decode-proj DeepGEMM, CUDA decode graph — all code-default ON.
- **profiling OFF** (NO `DECODE_PHASE_TIME` / `LINEAR_PROFILE`).
- **NO MTP** — the MTP-head deepgemm JIT hangs on .62's forced clang-11 host
  compiler (toolchain artifact, not a decode regression; needs a gcc≥10 build host
  for the real ~53). See the next-steps plan.
- **A/B = same binary, same session, two configs back-to-back**, only flip:
  `ARLE_DSV4_DECODE_COMPRESSOR_BATCH` (OFF `serve_bench_clean.sh` → ON
  `serve_bench_baseline.sh`).

## Params
Non-streaming `/v1/completions`, `max_tokens=128`, `temperature=0`, one ~28-token
prompt, 2 warmup reqs, c ∈ {1,2,4,8} (c concurrent identical, aggregate
wall-clock). Token count from response `usage.completion_tokens`. Driver
`/data01/run_baseline_ab.sh`, sweep `/data01/sweep.py`. **Single sweep per c**
(see caveat).

## Results — clean A/B (profiling OFF)

| c | OFF agg tok/s | ON agg tok/s | Δ ON vs OFF | ON per-req tok/s |
|---|---------------|--------------|-------------|------------------|
| 1 | 43.8 | 44.9 | +2.5% | 44.9 |
| 2 | 44.1 | 44.8 | +1.6% | 22.4 |
| 4 | 44.2 | 69.8 | **+58%** | 17.4 |
| 8 | 74.0 | 77.6 | +4.9% | 9.7 |

**Headline:** clean c=1 is **~44 tok/s** (ON 44.9 / OFF 43.8), NOT the contaminated
31.9. The OFF column **replicates the prior clean session** (plan doc: 43.0 / 45.0 /
45.0 / 74.8) within noise → the clean gate-OFF baseline is solid cross-session.

**Lever (compressor-batch):** biggest marginal win at **c=4 (+58%)** — where the
per-row compressor GEMVs still dominate OFF (44.2) but ON batches them (69.8). At
c=8 both configs already get broad batching (batched-FlashMLA + natural GEMM/MoE
batch), so the lever's marginal gain narrows to +5% (77.6 vs 74.0). Consistent with
"gain ∝ n until other batched paths saturate."

Roofline note: this entry is a same-session tok/s A/B anchor only. It does not
compute achieved-vs-peak; per `docs/bench-and-trace-spec.md` §7.7, roofline is
deferred to the next nsys/ncu component pass before using these rows as a
kernel-efficiency claim.

## MTP arm — MEASURED on .61 (gcc-13), root cause of the .62 "hang" confirmed

The `.62` MTP-head deepgemm JIT does NOT hang because of MTP — it hangs because
`.62` has only gcc-8.3 + clang-11/7 (no c++20 compiler better than clang-11;
enumerated 2026-06-16) and the forced `-ccbin clang++-11` chokes on the MTP-head
shapes. **On `.61` (glibc 2.39, gcc-13.3 default host compiler, no clang forcing),
MTP serves in 136s with zero new JIT** (cache stayed 33, jit procs=0, 8 workers
alive). So "MTP can't be tested" was a `.62`-toolchain artifact, **confirmed by
direct test**: gcc-13 compiles what clang-11 hangs on.

**`.61` same-binary same-session A/B (commit `2f021c0`, profiling OFF, gcc-13,
num-slots 64, ~28-tok prompt, max_tokens=128):**

| c | no-MTP | MTP | Δ |
|---|--------|-----|---|
| 1 | 43.4 | 47.9 | **+10%** |
| 2 | 43.6 | 48.1 | +10% |
| 4 | 67.2 | 48.1 | **−28%** |
| 8 | 73.5 | 79.1 | +8% |

MTP **+10% @c=1** (B=1), but **−28% @c=4**: MTP is flat ~48 c1→c4 then jumps to
79 @c8, while no-MTP scales c4→67. On this binary (`2f021c0`, NOT my
`3e3e50e0`+compressor-batch) the MTP lane looks per-row-plateaued at low c and
only the c=8 batched wave passes no-MTP — consistent with the per-row-MTP-plateau
finding ([[2026-06-15-dsv4-batched-mtp-prod-shape-flip]]). The `2f021c0` no-MTP
column matches the .62 OFF column at c1/2/8 (43/44/74) but diverges at c4 (67 vs
44) → `2f021c0` carries a c4 batched-decode path my OFF build lacks. So this is a
clean MTP-vs-no-MTP A/B **on the .61 binary**, NOT cross-comparable to the .62
compressor-batch table above.

**Prior-session MTP envelope** (different base/prompt/slots — cross-reference, not
a row-by-row delta):

**B=1 (single-stream), same-session ×3 ([[2026-06-13-dsv4-mtp-d2-chain-fold-53]]):**

| arm | B=1 tok/s |
|-----|-----------|
| no-spec base | 44.5 |
| **MTP d2 chain-fold (default-on)** | **52.8–53.3 (×3, σ≈0.04), +18–20%** |

**Concurrency — batched MTP (default-on at c≥4), prod ~2400-tok shape, num-slots 16,
same-session ([[2026-06-15-dsv4-batched-mtp-prod-shape-flip]]):**

| c | batched MTP | per-row MTP | Δ |
|---|-------------|-------------|---|
| 4 | 47.9 | 41.7 | — |
| 8 | **76.7** | 43.4 | **+77%** |
| 12 | 78.7 | 46.1 | +71% |

(Short-prompt c=12 +81%, [[2026-06-15-dsv4-batched-mtp-fold-win]]; batched-MTP-draft
sub-lever default-on adds +6.8%@c8/+11.1%@c16 after the batched lm_head fix,
[[2026-06-15-dsv4-batched-mtp-draft-default-on]].) The Δ here is **batched vs per-row
MTP**, not vs no-MTP: per-row MTP plateaus ~42–46 (sequential per-slot spec_step);
batched runs one amortized wave → scales to ~77.

**Done above** — the clean same-session MTP-vs-no-MTP c1–8 A/B now exists, on `.61`
(gcc-13). The remaining gap is a single-host run of *my* `3e3e50e0`+compressor-batch
binary WITH MTP on `.61` (the `.62` build can't serve MTP at all), to get a
compressor-batch × MTP combined number on one binary.

## MTP on REAL prompts (ShareGPT) + draft-depth d2 vs d3 (.61, gcc-13)

Synthetic fixed-prompt tok/s hides MTP's prompt-dependent acceptance. Re-ran on 12
varied real **ShareGPT** first-turn prompts (89–864 chars, from the local
`ShareGPT_V3_unfiltered_cleaned_split.json`), `max_tokens=256`, same `.61` binary
`2f021c0`, profiling OFF. B=1 = per-prompt median tok/s (the clean signal; the c=1
aggregate is dragged by one early-EOS prompt + one outlier).

| arm | B=1 median tok/s | c=8 agg tok/s |
|-----|------------------|---------------|
| no-MTP | 43.7 | 82.8 |
| **MTP d2** (`--mtp-draft-tokens 2`) | **49.7** (+14%) | 80.1 |
| MTP d3 (`--mtp-draft-tokens 3`) | 45.3 (+4%) | 73.7 |

- **MTP d2 = +14% B=1 on real prompts** (vs +10% on the synthetic prompt) — realistic
  redundancy lifts acceptance. Per-prompt range 41–59: a java-coding prompt hit 58.7
  (predictable → high accept), prose hit 41 (low accept). Acceptance IS prompt-dependent
  — the synthetic single-prompt number was an underestimate of the favorable case and an
  overestimate of the prose case.
- **d3 is WORSE than d2** (45.3 vs 49.7, −9%): depth-3 over-drafts — the 3rd draft
  token's marginal acceptance doesn't pay for the extra draft+verify forward. **d2 is the
  sweet spot**, confirming [[2026-06-13-dsv4-mtp-d2-chain-fold-53]]. d3 degrades further
  at c=8 (73.7) — deeper draft hurts more under concurrency.
- **At c=8, MTP does not help** (no-MTP 82.8 ≥ d2 80.1 ≥ d3 73.7): batched no-MTP already
  saturates the decode wave, so MTP's per-step draft+verify overhead is net-negative on
  this binary. MTP winning at c≥4 needs the batched-MTP lane
  ([[2026-06-15-dsv4-batched-mtp-prod-shape-flip]]), not fully engaged in `2f021c0`.
  **MTP is a B=1 / low-concurrency lever here.**

## Problems / caveats
- **Single sweep per c — the c=4 +58% vs c=8 +5% non-monotonicity is not yet
  pinned to a CI.** The direction (lever helps at c≥4) is solid and matches the
  high-c A/B (n=22, c=64: +38%, [[2026-06-16-dsv4-batched-compressor-prepass]]),
  but the exact per-c magnitude needs ≥3 repeats / median before it's enshrined.
  Labeled hypothesis on magnitude, evidence on direction + sign.
- No TTFT/ITL (streaming `/v1/completions` → HTTP 400 on this build; non-streaming
  only). guidellm not installed on .62.
- Sub-linear aggregate scaling persists (per-req 44.9→9.7 from c1→c8): the decode
  step is ∝ n. Next lever = compute/comm overlap on the batched lane.

## Rule
- A committed bench number whose launcher had `DECODE_PHASE_TIME`/`LINEAR_PROFILE`
  is profiling-contaminated (per-step sync, −25–35%) and is NOT a throughput
  baseline — re-measure profiling-OFF before citing. Always state the bench's
  profiling state in the config block.
- **A serve that "hangs / can't load MTP" on a box with only clang-11 is a JIT
  host-compiler artifact, not a model/code failure.** DSv4's deepgemm bridge needs
  `-std=c++20`; gcc-8.3 is too old → clang-11 forced on `.62` hangs the MTP-head
  shapes. The fix is the toolchain: `.61` (gcc-13.3) serves MTP in 136s, no hang.
  Verify "X can't run here" against a second host before declaring it a property of
  X — the `.62`/`.61` split is the controlled experiment.
