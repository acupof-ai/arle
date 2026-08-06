# The decode lever board, rebuilt — Marlin at m=1 is 73% of achievable, not 51%

> Status: Survey + correction. No new measurement. Re-ranks the decode levers
> against the 2026-08-04 W8A16 kernel budget and against what the outside
> stacks actually ship.

## Context

Asked where the next large decode win is. The answer given was: the W8A16
Marlin GEMM runs at 51% of peak HBM at m=1, so roughly 2× of bandwidth
headroom exists, so the lever is a Marlin config sweep (`blocks_per_sm`, the
`small_batch_thread_configs` table).

The premise was two measurements out of date and the ranking built on it was
wrong.

## Root Cause

The 51% comes from `ncu` on 2026-08-02
(`/host/marlin-bench/ncu/marlin_roofline_noclk.csv`, recorded in
`reference_w8a16_marlin_decode_occupancy_bound_not_memory_wall`): 2.04 TB/s,
warp occupancy 12.5%, `blocks_per_sm≡1`.

[`wins/2026-08-04-w8a16-decode-step-kernel-budget.md`](../wins/2026-08-04-w8a16-decode-step-kernel-budget.md)
is two days newer, measures the same kernel on the same shapes on the same GPU,
in situ under `nsys --cuda-graph-trace=node`, and reports **11.709 ms for 30 GB
= 2.56 TB/s = 73% of the 3.5 TB/s achievable**. It also finds Marlin within
0.8% of SGLang's own run of the identical kernel.

Two reasons the older number is the wrong one to rank against:

1. `ncu` serializes launches and flushes caches between replays. Its bandwidth
   figure is a kernel-isolation figure, not a steady-state one; here it
   understates by 25%.
2. The memory index is organised by topic, so the *topic-matching* file surfaces
   first and the *newest measurement* does not. The lever board lives in the
   dated entry, not in the index.

Occupancy 12.5% and 73% of achievable bandwidth are both true. Low occupancy did
not turn out to be the binding constraint it looked like in isolation.

## The corrected board

Qwen3.6-27B W8A16, 1×H20, c=1, 33K context, shipped champion. All rows from the
2026-08-04 budget; floors at 3.5 TB/s achievable read.

| bucket | ms | share | achieved | of achievable | floor | headroom |
|---|---:|---:|---:|---:|---:|---:|
| Marlin W8A16 GEMM (256 launches) | 11.709 | 70% | 2.56 TB/s | **73%** | 8.57 | 3.14 |
| FA3 decode + combine (32) | 2.433 | 15% | 0.93 TB/s | **27%** | 0.618 | 1.82 |
| lm_head (1) | 0.666 | 4% | 2.24 TB/s | 64% | 0.43 | 0.24 |
| ~750 small kernels | 1.53 | 9% | — | latency-bound | ≈0 | 1.53 |
| Σ | 16.65 | | | | 9.9 | **6.73** |

The row that owns 70% of the step has the least available time per unit of work
it is required to do. The worst efficiency in the table is the attention.

## What the outside stacks actually have

**Machete is not the answer at m=1.** Red Hat's own figures put it ahead only
from batch / prefill length 128 upward, and the announcement lists "optimizing
low batch size performance" as future work. It is a Hopper `wgmma` design;
`wgmma` wants m ≥ 64.

**TensorRT-LLM and LMDeploy TurboMind both dispatch a separate CUDA-core
small-M kernel for M=1,2**, with the tensor-core GEMM reserved for M ≥ 128 and
a measured-latency choice in between. vLLM, SGLang and ARLE send every M through
Marlin. That is a real structural difference, and it is the one thing the two
fastest closed-ish stacks do that we do not.

The size of the prize is published and it is not 2×. TurboMind's own
microbenchmark against vLLM+Marlin reports GEMM gains of **19.2% average, 25.5%
maximum**, and 7.6% average on decode-phase operator latency. Against our
11.709 ms that is ~2.2 ms — inside the 3.14 ms of headroom the budget shows, and
consistent with it. The TurboMind paper's stated reason is the same one the
Marlin source shows: "MARLIN requires manual specifications of warp layouts and
tile sizes, necessitating extensive tuning for different GPU architectures."

**Speculative decode at concurrency has a 2026 answer.** D-cut ranks draft
tokens globally across the whole batch by drafter confidence, keeps
high-confidence prefixes, prunes low-confidence suffixes, and reallocates the
verification budget across requests instead of verifying a uniform depth for
everyone. Against DFlash at batch 32 it moves 1.26× → 1.65×; at batch 64 with
temperature-1 sampling it holds 1.15× where DFlash drops below plain
autoregressive decode. Our c=16 row has the same signature: `accept 0.280`,
`tok/row 0.400`, DSpark 2.9× at c=1 collapsing to 1.1×.

**MagicDec's premise holds for this model.** It argues that beyond a critical
sequence length, decode stays memory-bound even at large batch, because KV
traffic scales with batch while weight traffic does not — and that a draft with
a fixed small context window therefore keeps paying at high throughput (up to
2.51× at batch 32–256 on Llama3.1-8B). Our KV is
`4 kv_heads × 256 head_dim × 2 (K,V) × 2 B × 16 full-attn layers` =
**64 KB/token**, so 33K is 2.16 GB per sequence and c=16 is **34.6 GB against
29 GB of weights**. The regime it describes is the regime we serve.

## Where the two axes actually land

**The GEMM axis shrinks.** 3.14 ms of a 16.65 ms step (19%) at c=1, and it
amortizes away at concurrency because the weights are read once per step
regardless of batch. It is also a kernel port, not a config flip — the
literature's own margin is ~20%, not the 2× a 51%-to-100% roofline argument
implied.

**The attention axis and the speculative-decode axis are the same axis.** The
verify slope recorded in `docs/baselines.md` is 2.48 ms/row short-context rising
to 5.18 ms/row at 33K; the +2.7 ms of context-dependent per-row cost is
numerically the same thing as FA3's own 2.433 ms/row at c=1 reading 2.16 GB at
0.93 TB/s. Both point at one kernel running at 27% of achievable bandwidth on a
term that scales with batch.

Projected (not measured) at c=16, 33K: 16 × 2.16 GB = 34.6 GB of KV, which is
37 ms at the current 0.93 TB/s and 9.9 ms at 3.5 TB/s. That is the term that
makes verify expensive at concurrency, and it is the term that pushes DSpark
from 2.9× to 1.1×.

## Ranked

| # | Lever | c=1 prize | c=16 prize | Kind |
|---|---|---|---|---|
| 1 | FA3 decode KV read, 27% → 70–80% | 1.8 ms of 16.65 (11%) | ~27 ms of a projected ~37 ms FA3 term | Kernel/config, probe first |
| 2 | Small-M GEMM path (TurboMind/TRT-LLM shape-aware dispatch) | ~2.2 ms (13%) | amortizes away | Kernel port |
| 3 | D-cut adaptive verify depth | none | DSpark 1.1× → ~1.5× projected | Scheduling, reuses the existing confidence head |
| 4 | ~750 small kernels, 1.53 ms of pure latency | 1.53 ms (9%) | same | Register-resident fusion only |

#1 pays on both axes at once and is the only item where the efficiency gap is
large enough that a config or launch-geometry fix might reach it. It is also
already the recorded open question:
[`wins/2026-08-04-fa3-decode-splits-fill-the-sms.md`](../wins/2026-08-04-fa3-decode-splits-fill-the-sms.md)
ends by asking whether the kernel reads the required 2.16 GB inefficiently or
reads several times that, and names `ncu` with the decode graph off as the
probe.

## Unverified

- **That probe has never run.** Two `ncu` attempts failed: the first matched
  nothing because `--kernel-name-base function` strips template arguments from
  `cutlass::device_kernel<...>`, the second stalled after graph capture. Until
  it runs, "27% efficiency" does not distinguish a slow read of the right bytes
  from a correct read of 3.7× too many bytes, and those have different fixes.
- **`blocks_per_sm > 1` has never been tried.** Upstream supports it
  (`marlin/gptq_marlin.cuh:662` halves the per-block smem budget), the lock
  buffer is already oversized for it (`marlin_w8a16.cu`,
  `marlin_w8a16_workspace_ints` returns `sms * 4`), and
  `determine_exec_config` hardcodes `return {1, th_config}` at
  `gptq_marlin.cuh:519`. `c_tmp` is sized `sms * max_m_block * max_thread_n` —
  per SM, not per block — so any arm above 1 needs that indexing checked before
  it is trusted. At 73% achieved the expected prize is small; this is a cheap
  probe, not a program.
- **No TensorRT-LLM comparison on H20.** TurboMind reports +118.9% throughput
  over TensorRT-LLM on L40S/A100, so the spread between the outside stacks is
  far wider than our 2.8% margin over SGLang. Where the real bar sits is
  unmeasured.

## Rule

**Rank levers against the newest budget entry for the exact binary and dtype,
not against the topic-matching memory.** The memory index answers "what do we
know about X"; only the dated entry answers "what is X worth today". Two days
and one shipped fix were enough to move this number by 22 points and to reverse
the ranking built on it.

**An `ncu` bandwidth figure is not an in-situ bandwidth figure.** The profiler
serializes and flushes; use it to explain *why* a kernel is slow, and a graph-node
trace to decide *whether* it is.

Related: [[feedback_measured_floor_is_not_physical_floor]],
[[feedback_no_ungrounded_estimates]],
[[feedback_read_scaling_curve_before_kernel_rewrite]],
[[feedback_dont_file_hypothesis_as_root_cause]].
