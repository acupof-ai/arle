# The W8A16 27B decode step, kernel by kernel — CUDA, 2026-08-04

> Status: Measurement. First per-kernel budget for the W8A16 champion; the
> 2026-08-01 budget is the FP8 model and its weight layout does not transfer.

## Goal

Split the 16.9 ms decode step at 33K context into kernels, and score each
against the bytes it has to read.

## A whole-step graph makes the default trace blind

`nsys profile --trace=cuda` on the champion reports **one kernel**:

```
Time(%)  Total(ns)   Instances  Avg(ns)  Name
100.0    25,121,062  1176       21,361   argmax_kernel_fast(...)
```

Every other kernel is inside the captured decode graph, and nsys times a graph
replay as a single unit by default. `argmax` is the only kernel left outside
it, in the sampling tail. The window is correctly placed — 1176 argmax calls
in 20 s is 17 ms/step — and it still shows nothing.

**`--cuda-graph-trace=node` is mandatory once a whole-step graph exists.** It
times each graph node separately and the step opens up.

## Parameters

```bash
nsys launch --session-new=... --trace=cuda --cuda-graph-trace=node env ... arle serve ...
# collection started 45 s in (past the 33K prefill), 20 s window, 3000-token completion
nsys stats --report cuda_gpu_kern_sum --force-export=true
```

- H20 GPU 6, Qwen3.6-27B W8A16, 33K prompt, c=1, shipped defaults
- Steps in window: 1147 (`embedding_batched_native_kernel`, exactly 1/step)
- In-process `submit` over the same run: 17.272 ms

## Results

Per decode step, total ÷ 1147:

| kernel | launches | ms | % |
|---|---:|---:|---:|
| Marlin W8A16 GEMM | 256 | 11.709 | 70.3 |
| FA3 decode attention | 16 | 2.373 | 14.3 |
| lm_head (`nvjet_tst_128x8`) | 1 | 0.666 | 4.0 |
| `rms_norm_batched_offset` | 128 | 0.408 | 2.5 |
| `gated_delta_rule_decode` | 48 | 0.228 | 1.4 |
| in_proj GEMV (`nvjet_tst_64x8` splitK) | 48 | 0.224 | 1.3 |
| `add_native` | 128 | 0.176 | 1.1 |
| `conv1d_prefill` | 48 | 0.124 | 0.7 |
| `split2` | 96 | 0.112 | 0.7 |
| `decode_prep_paged_hd256` | 16 | 0.111 | 0.7 |
| `silu_mul_fused` | 64 | 0.104 | 0.6 |
| `rms_norm_gated` | 48 | 0.079 | 0.5 |
| `conv1d_state_update` | 48 | 0.078 | 0.5 |
| `splitKreduce` | 48 | 0.071 | 0.4 |
| FA3 combine | 16 | 0.060 | 0.4 |
| `prepare_varlen_num_blocks` | 16 | 0.054 | 0.3 |
| 5 more, each ≤ 0.03 | 51 | 0.074 | 0.4 |
| **Σ kernel** | **~1150** | **16.651** | |

Σ kernel 16.651 against `submit` 17.272 leaves 0.62 ms (3.6%) of gap, which
includes the nsys tracing overhead present in this run and absent from the
17.272.

## Scored against the bytes

H20 achievable read is 3.5 TB/s (measured, `docs/baselines.md`).

| bucket | bytes/step | floor | measured | achieved | of achievable |
|---|---:|---:|---:|---:|---:|
| Marlin (int8 weights) | 30 GB | 8.57 ms | 11.71 | 2.56 TB/s | 73% |
| FA3 (KV at ~34K) | 2.2 GB | 0.63 | 2.37 | 0.93 TB/s | 27% |
| lm_head (bf16) | 1.5 GB | 0.43 | 0.67 | 2.24 TB/s | 64% |
| ~750 small kernels | ≈ 0 | ≈ 0 | 1.53 | — | latency |

Three separate problems, and they do not rank the way the time column does:

- **Marlin owns 70% of the step and is already at 73% of achievable.** Only
  3.1 ms exists there and it costs a kernel rewrite.
- **FA3 is the worst efficiency at 27%**, but after the split-ceiling fix it is
  down to 2.37 ms, so the whole prize is 1.7 ms.
- **~750 small kernels cost 1.53 ms while reading almost nothing.**
  `rms_norm_batched_offset` takes 3.19 µs for 10 KB in and 10 KB out — 6 ns of
  traffic at the achievable rate. This bucket is entirely per-kernel latency.

## Inside the replay the kernels are back to back

Node start/end timestamps for one step, taken between two consecutive
`embedding_batched_native_kernel` calls:

| | |
|---|---:|
| kernel nodes | 1060 |
| first start → last end | 16.683 ms |
| Σ node duration | 16.631 ms |
| inter-node gap, total | 88.6 µs (0.084 µs/node) |
| overlap, total | 36.7 µs |

The step is 99.7% busy. Launch overhead inside a captured graph is 0.084 µs
per node, so removing nodes can return at most that — which is why fusing 192
residual-add + norm pairs moved the wall 0.00 ms
([entry](../errors/2026-08-04-launch-count-is-not-the-decode-lever.md)).

That does not make the 1.53 ms small-kernel bucket unreachable, and it explains
why that particular fusion missed it. A fused kernel replaces two *durations*
with one, worth `add_native`'s 0.176 ms — but the one built there re-read the
sum from global memory in its second pass, so its traffic equalled the unfused
pair's and its duration did too. It bought one 0.084 µs launch and paid a full
global read. **The bucket is reclaimable only by fusion that keeps the second
pass in registers** (the [T6 GDN](2026-08-03-t6-gdn-decode-kernel.md) pattern);
traffic-preserving fusion is zero by construction.

## Against SGLang, kernel by kernel

Same GPU, same int8 values (mechanical GPTQ v1 repack), same `gptq_marlin`
kernel, SGLang 0.5.13 traced the same way (957 steps):

| kernel | ARLE ms | SGLang ms | Δ |
|---|---:|---:|---:|
| Marlin GEMM (256 each) | 11.709 | 11.611 | +0.098 |
| FA3 decode (16 each) | 2.373 | 2.507 | −0.134 |
| lm_head (1 each) | 0.666 | 0.666 | 0.000 |
| GDN decode (48 each) | 0.228 | 0.277 | −0.049 |
| in_proj splitK (48 each) | 0.224 | 0.305 | −0.081 |
| norms | 0.487 | 0.503 | −0.016 |
| conv1d | 0.202 | 0.125 | +0.077 |
| remaining tail | 0.586 | 0.533 | +0.053 |
| **Σ kernel** | **16.651** | **16.527** | **+0.124 (+0.7%)** |

**The decode step is at parity**, from 1.57× behind on
[2026-08-02](2026-08-02-w8a16-sglang-matched-ab.md). Marlin matches to 0.8%,
which is what running the same kernel on the same bytes should look like. FA3
is 5% *faster* here after the split-ceiling fix. The one row behind was conv1d,
since [closed](2026-08-04-conv1d-decode-fusion.md).

End to end on the same 16×256-token c=1 protocol, SGLang delivers 10.5 tok/s
against ARLE's 7.6. Subtracting the measured decode time (256 × 16.95 ms) puts
TTFT at **~29.4 s vs ~20.2 s**, the same 1.45× as 2026-08-02 — **prefill has
not moved while decode closed.** Those TTFTs are derived from throughput, not
measured; the direct measurement is the next step.

## Learnings

The step is 100% kernel time (host is 0.061 ms,
[entry](2026-08-04-decode-step-has-no-host-tail.md)), and now it is 100%
attributed.

**Rule: a whole-step CUDA graph silently empties the default profile.** One
kernel at 100% is not a fast step, it is a graph the profiler declined to open.
Any trace of a graphed path needs `--cuda-graph-trace=node`, and the check is
that the kernel count matches the step's known launch count.

**Rule: rank optimization targets by headroom, not by share.** The largest row
here has the least available time per unit of work required, and the cheapest
win sits in a bucket worth 9% of the step.

**Rule: a fusion is worth a kernel duration, not a launch.** Inside a graph the
launch is 0.084 µs and the duration is microseconds. A fusion that leaves the
byte count unchanged returns the launch and nothing else.

Related: [[feedback_measured_floor_is_not_physical_floor]],
[[feedback_path_probe_before_perf_claim]].
