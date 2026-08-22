# The decode lever board, rebuilt — the Marlin GEMM is not the top lever

> Status: Survey + code map + correction. No new measurement. Re-ranks the
> decode levers against the 2026-08-04 W8A16 kernel budget, a full read of the
> vendored Marlin launch path, and the external state of the art.

## Context

Asked where the next large decode win is. The answer given was: the W8A16
Marlin GEMM runs at 51% of peak HBM at m=1, so roughly 2× of bandwidth headroom
exists, so the lever is a Marlin config sweep.

The premise was two measurements out of date, the prize was overstated ~3×, and
the item was not the largest one on the board.

## Root Cause

The 51% comes from `ncu` on 2026-08-02
(`/host/marlin-bench/ncu/marlin_roofline_noclk.csv`, recorded in
`reference_w8a16_marlin_decode_occupancy_bound_not_memory_wall`): 2.04 TB/s,
warp occupancy 12.5%, `blocks_per_sm ≡ 1`.

[`wins/2026-08-04-w8a16-decode-step-kernel-budget.md`](../wins/2026-08-04-w8a16-decode-step-kernel-budget.md)
is two days newer, measures the same kernel on the same shapes on the same GPU,
in situ under `nsys --cuda-graph-trace=node`, and reports **11.709 ms per step**
across 256 launches — within 0.8% of SGLang's run of the identical kernel.

Two reasons the older number is the wrong one to rank against:

1. `ncu` serializes launches and flushes caches between replays. Its bandwidth
   figure is kernel-isolation, not steady-state.
2. The memory index is organised by topic, so the topic-matching file surfaces
   first and the newest measurement does not. The lever board lives in the dated
   entry, not in the index.

Occupancy 12.5% and a healthy achieved bandwidth are both true. Low occupancy is
not the binding constraint it looked like in isolation — see the launch-argument
mechanism below.

## The Marlin denominator needs re-checking

The 2026-08-04 budget scores Marlin against **30 GB/step**, giving 2.56 TB/s
(73% of the 3.5 TB/s achievable). Per-shape arithmetic does not reproduce that
number. Five of the seven decode GEMM shapes are Marlin-eligible (`in_proj_ba`
fails `N % 64`; `lm_head` is loaded BF16 and never repacked):

| shape (n, k) | launches/step | int8 MB each | GB/step |
|---|---:|---:|---:|
| gate_up (34816, 5120) | 64 | 178.3 | 11.41 |
| down (5120, 17408) | 64 | 89.1 | 5.70 |
| in_proj_qkvz (16384, 5120) | 48 | 83.9 | 4.03 |
| o_proj + out_proj (5120, 6144) | 64 | 31.5 | 2.01 |
| qkv (14336, 5120) | 16 | 73.4 | 1.17 |
| **Σ** | **256** | | **24.32** |

That reconciles with the parameter count: 27B total − 1.27B embeddings −
1.27B `lm_head` = 24.35B transformer parameters, and `lm_head`'s BF16 traffic is
already a separate row in the same budget table. **At 24.3 GB the Marlin GEMM
achieves 2.08 TB/s = 59% of achievable, floor 6.95 ms, headroom 4.76 ms** — not
3.14. The 30 GB figure looks like it double-counts the embedding/`lm_head`
bytes. Confirm before either number is quoted again.

## The corrected c=1 board

Qwen3.6-27B W8A16, 1×H20, c=1, 33K context, shipped champion. Times from the
2026-08-04 budget; floors at 3.5 TB/s achievable read.

| bucket | ms | share | bytes | achieved | of achievable | floor | headroom |
|---|---:|---:|---:|---:|---:|---:|---:|
| Marlin W8A16 GEMM (256) | 11.709 | 70% | 24.3 GB | 2.08 TB/s | **59%** | 6.95 | 4.76 |
| FA3 decode + combine (32) | 2.433 | 15% | 2.16 GB | 0.89 TB/s | **25%** | 0.618 | 1.82 |
| lm_head (1) | 0.666 | 4% | 1.5 GB* | 2.24 TB/s | 64% | 0.43 | 0.24 |
| ~750 small kernels | 1.53 | 9% | ≈0 | — | latency-bound | ≈0 | 1.53 |
| Σ | 16.65 | | | | | 8.0 | **8.35** |

\* the `lm_head` byte figure carries the same unreconciled-denominator caveat
(248320 × 5120 × 2 B = 2.54 GB, not 1.5).

## What actually dominates at concurrency

`docs/baselines.md` fits verify to **22 ms intercept + 2.48 ms/row**, rising to
5.18 ms/row at 33K. Reading the code, those three terms are three different
mechanisms:

| term | what it is | c=1 | c=16 |
|---|---|---:|---:|
| 22 ms intercept | the weight read, shared by the whole batch | 22 | 22 |
| 2.48 ms/row | **per-row kernel launches, not bytes** | 2.5 | 39.7 |
| +2.7 ms/row at 33K | the full-attention KV read | 2.7 | 43.2 |

**The 2.48 ms/row is a serialized launch chain.** During verify the GDN advance
runs `LinearCore::Rows` — a host loop over rows issuing conv1d + GDR launches
inside each of the 48 linear layers (`qwen35.rs:7052-7115`). At 5.9 µs per token
per layer that is 48 × 7 × 5.9 µs = **1.98 ms, ~80% of the term**. The byte
floor for the same work is 302 MB/row = 0.086 ms, 3.5% of it. A batched
single-launch sibling exists — `LinearCore::Tables`, `qwen35.rs:7117-7182` — and
fires only for pure 1-token decode. The full-attention prep is also still
per-row during verify, because `meta.seq_len == 7 ≠ 1` puts it on the non-decode
branch (`qwen35.rs:6432`, `:6548-6578`): 16 layers × 16 rows = 256 more launches
FA3's own batching did not remove.

**The +2.7 ms/row is FA3 reading the KV cache at ~0.8 TB/s.** It is the only
per-row term in the code that scales with context:
`33,000 × 16 layers × 4 kv_heads × 256 × 2 (K,V) × 2 B` = 2.163 GB/row = 618 µs
at 3.5 TB/s, so the kernel runs **4.4× off roofline**. Independently
corroborated: [`wins/2026-07-28-fa3-one-launch-per-layer.md`](../wins/2026-07-28-fa3-one-launch-per-layer.md)
measured the same kernel 5.5× off its batch roofline on the MoE twin after
batching, 17× before.

At c=16 the verify step is therefore roughly **22 weights / 40 per-row launches
/ 43 KV read**, and the two structural terms are each larger than the GEMM.
Neither is the item the old board ranked first.

## What the outside stacks have

**Machete is not the answer at m=1.** Red Hat's own figures put it ahead only
from batch / prefill length 128 upward, and the announcement lists "optimizing
low batch size performance" as future work. It is a Hopper `wgmma` design;
`wgmma` wants m ≥ 64.

**TensorRT-LLM and LMDeploy TurboMind both dispatch a separate CUDA-core
small-M kernel for M=1,2**, reserving the tensor-core GEMM for M ≥ 128 with a
measured-latency choice in between. vLLM, SGLang and ARLE send every M through
Marlin. TurboMind's microbenchmark margin over vLLM+Marlin is **19.2% average /
25.5% maximum on GEMM** and 7.6% on decode-phase operator latency — ~2.2 ms
against our 11.709, inside the measured headroom either way. The stated reason
matches what the source shows: "MARLIN requires manual specifications of warp
layouts and tile sizes, necessitating extensive tuning for different GPU
architectures."

**D-cut** ranks draft tokens globally across the batch by drafter confidence,
keeps high-confidence prefixes, prunes low-confidence suffixes, and reallocates
verification budget across requests instead of verifying a uniform depth.
DFlash at batch 32: 1.26× → 1.65×; at batch 64 temperature 1 it holds 1.15×
where DFlash drops below plain decode.

**We already ship that mechanism.** `dspark_verify_keeps`
(`qwen35/dspark.rs:1565-1655`) runs a per-drafted-token sigmoid acceptance head,
cumprods it into a survival curve, and `dspark_verify_lens`
(`qwen35-spec/src/lib.rs:1313-1336`) takes the goodput-optimal admission cut.
Cost is one small GEMM plus one D2H per tick. So D-cut is not a feature to
build; the open question is whether its cost model is right at every operating
point — see below.

**MagicDec's premise holds for this model.** Beyond a critical sequence length
decode stays memory-bound even at large batch, because KV traffic scales with
batch while weight traffic does not. Our KV is 64 KB/token, so 33K is 2.16 GB
per sequence and c=16 is **34.6 GB against 24.3 GB of weights**. The regime it
describes is the regime we serve.

## The Marlin launch path, read end to end

Worth recording because it explains the 12.5% occupancy and scopes the cheap
probe.

- `determine_exec_config` hardcodes `return {1, th_config}`
  (`marlin/gptq_marlin.cuh:506`); the explicit-thread_k/n branch also forces 1
  (`:631`). `blocks = sms * blocks_per_sm` at `:661`.
- **At `blocks_per_sm == 1` the launch requests the entire 232448 B opt-in
  dynamic shared memory while the chosen config's layout needs 82944 B.** The
  smem division at `:662` only applies when `blocks_per_sm > 1`, so the
  over-request alone pins residency at one block per SM: `floor(233472 / 232448)
  = 1`. The kernel's own `max_shared_mem` parameter is dead —
  `marlin_template.h:304` declares it and nothing reads it; every buffer is
  sized `constexpr`. **The 12.5% occupancy is a launch-argument artifact, not a
  kernel design limit.**
- Nothing in the kernel body assumes `grid == sms`. The striped scheduler is
  fully `gridDim.x`-parameterized (`marlin_template.h:356`, `:367-368`), the
  lock protocol is `barrier_acquire`/`barrier_release` and self-resetting, and
  the lock buffer is already 4× oversized (`marlin_w8a16.cu:168` returns
  `sms * 4`).
- **`C_tmp` is the real constraint, and it is scoped.** Both `C_tmp` and the
  locks are indexed by `locks_off ≤ gridDim.x - 1`, but the serving allocation
  is sized once at `marlin_w8a16_c_tmp_floats(64, sms)`
  (`quant_linear.rs:606-620`) and shared by decode and prefill. At m=1 the
  per-block slot is 2048 floats, so that allocation holds 8 × sms slots — safe
  to `blocks_per_sm = 8`. At m ≥ 64 (prefill, and the batched verify at m=112)
  the slot is exactly 16384 floats and the allocation is exactly sms slots, so
  `blocks_per_sm = 2` would write 2× past the end. **A small-M-only raise is
  safe with today's allocation; a global flip is not.**
- Shared-memory budgets for the three instantiated m=1 configs are 82944 /
  41984 / 49664 B. At `blocks_per_sm = 2` the budget is 115200 B and the
  current config still passes with no other change. At 4 it is 57088 B and only
  the two 128-thread configs fit. At ≥ 3 the code as written **throws**:
  `determine_exec_config` is handed the full `max_shared_mem` (`:649`) while the
  post-hoc `RuntimeCheck` uses the reduced one (`:667-680`).
- None of the tuning knobs reach Rust: `thread_k/n` are hardcoded `-1, -1` and
  `use_atomic_add` / `use_fp32_reduce` are hardcoded at `marlin_w8a16.cu:142-145`;
  `blocks_per_sm` is not a `marlin_mm` parameter at all.

## Ranked

| # | Lever | c=1 | c=16 | Kind | Status |
|---|---|---|---|---|---|
| 1 | Batch the GDN advance + attention prep across verify rows | — | ~32 ms of a ~105 ms verify | Reuse the existing `LinearCore::Tables` path | **KILLED**, measured below |
| 2 | FA3 decode KV read, 25% → 70–80% | 1.8 ms of 16.65 (11%) | ~30 ms of a ~43 ms KV term | Probe first, then kernel/config | now #1 |
| 3 | Small-M GEMM path (TRT-LLM / TurboMind shape-aware dispatch) | ~2.2–4.8 ms | amortizes away | Kernel port | open |
| 4 | Marlin smem over-request → `blocks_per_sm` 2 at small M | unknown, cheap to test | same | ~4-line additive edit, decode-only | open |
| 5 | ~750 small kernels, 1.53 ms of pure latency | 1.53 ms (9%) | same | Register-resident fusion only | open |

## #1 measured, and killed

The per-row term is real. The attribution to the GDN lane was wrong.

Phase split of the DSpark tick — c=16, 33K prompts, ThinkingCap-Qwen3.6-27B-FP8
on 1× H20 (GPU 6), steady-state samples only:

| rows | chain_rows | draft | snap | verify | commit | tick |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 96 | 47.94 | 1.55 | 109.29 | 3.21 | **162.0 ms** |
| 1 | 6 | 2.29 | 0.12 | 24.00 | 0.44 | **26.9 ms** |

Slopes across the two points: verify **5.69 ms/row**, draft **3.04 ms/row**.
85% of the tick scales with rows, which is what the model predicted.

Then the A/B — same binary, only the GDN lane swapped
(`--qwen35-gdr-chunked false`):

| arm | draft | snap | verify | commit | tick |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunked (shipped) | 47.94 | 1.55 | 109.29 | 3.21 | 162.0 |
| recurrent | 48.05 | 1.55 | 110.14 | 3.21 | 162.9 |
| Δ | +0.2% | 0 | **+0.8%** | 0 | +0.6% |

The swap changes both the algorithm and the per-(row, layer) launch count by
2–4×, and verify moved 0.8%. The arm is provably engaged, so this is a real
null, not a dead flag.
**The per-row verify cost is neither the GDN kernel nor its launch overhead.**
That kills both proposed fixes: the chain-length threshold and the
batch-across-rows rewrite.

It also settles where the chunked-GDR win lives. The tick is identical across
arms, so all 30% is prefill and none of it is decode.

With the GDN lane excluded, the ~91 ms per-row verify term is dominated by the
FA3 KV read: 16 rows × 2.16 GB at 0.93 TB/s ≈ 51 ms. #1 collapses into #2.

## Unverified

- **FA3's achieved bandwidth on this exact shape** (hd256, 4 kv_heads, qlen 7,
  16 rows, splits 8, 33K KV). Everything above infers it from the fit slope
  minus the other terms. `ncu dram__bytes.sum` + duration on
  `qwen/full_paged/attention` distinguishes one inefficient read from a second
  unmodeled context term — different fixes.
  [`wins/2026-08-04-fa3-decode-splits-fill-the-sms.md`](../wins/2026-08-04-fa3-decode-splits-fill-the-sms.md)
  already named this probe and records two failed `ncu` attempts: the first
  matched nothing because `--kernel-name-base function` strips template
  arguments from `cutlass::device_kernel<...>`, the second stalled after graph
  capture.
- **Which kernel owns the ~91 ms per-row verify term.** The GDN lane is
  excluded by measurement (above); the FA3 KV read accounts for ~51 ms by
  arithmetic; the remaining ~40 ms is unattributed. An nsys kernel-name
  histogram of one verify step settles it. Two attempts failed: session mode
  (`nsys launch --session-new` + `nsys start`) produced a report with no CUDA
  kernel data and an `nsys stats` that died on
  `boost::filesystem::current_path`; `--delay 260 --duration 40` missed the
  window because the bench finished at 231 s. A third attempt should use
  `--delay 150`.
- **The Marlin byte denominator** — 24.3 GB by arithmetic vs 30 GB in the
  budget entry. Efficiency is 59% or 73% depending on which is right, and the
  headroom 4.8 ms or 3.1 ms.
- **`blocks_per_sm > 1` has never been run.** A wall-clock delta alone would not
  prove the arm engaged; it needs an occupancy read. L2 is also a trap for any
  microbenchmark here — `o_proj` is 31.5 MB of INT8 and fits entirely in an H20's
  L2, so back-to-back iterations would report a fictitious GB/s. There is no
  L2-flush helper in the tree.
- **`scripts/marlin_serve.sh` is a no-op A/B.** Its only knob,
  `ARLE_W8A16_DISABLE_MARLIN`, is read by no code in the tree, so both arms run
  identical binaries. Any prior end-to-end marlin-vs-scalar number from that
  script should be discarded.
- **The DSpark goodput cost model is a c=16 whole-tick fit** — `bias_ms 211.0`
  = trunk 116 + draft 95, `row_ms 0.53` (`qwen35-spec/src/lib.rs:1290-1301`). It
  reproduces the measured c=16 burst (211 + 0.53 × 112 = 270 ms vs 262.7 ms
  recorded) so it is not mis-fit, but it is a single constant pair applied at
  every concurrency and context length. Whether re-fitting it per operating
  point moves the admission cut is untested. It also implies the draft phase is
  ~95 ms of a ~270 ms c=16 tick — a third of the tick, and nothing in
  `docs/experience/` attributes it.
- **No TensorRT-LLM comparison on H20.** TurboMind reports +118.9% throughput
  over TensorRT-LLM on L40S/A100, so the spread between outside stacks is far
  wider than our 2.8% margin over SGLang. Where the real bar sits is unmeasured.

## Rule

**Rank levers against the newest dated entry for the exact binary and dtype, not
against the topic-matching memory.** The memory index answers "what do we know
about X"; only the dated entry answers "what is X worth today". Two days and one
shipped fix were enough to reverse this board.

**Re-derive the denominator before quoting an efficiency.** Three different
percentages (51 / 59 / 73) for one kernel came from one measured duration and
three different byte counts. The duration was never in doubt.

**An `ncu` bandwidth figure is not an in-situ bandwidth figure.** The profiler
serializes and flushes; use it to explain *why* a kernel is slow and a
graph-node trace to decide *whether* it is.

Related: [[feedback_measured_floor_is_not_physical_floor]],
[[feedback_no_ungrounded_estimates]],
[[feedback_read_scaling_curve_before_kernel_rewrite]],
[[feedback_dont_file_hypothesis_as_root_cause]].
