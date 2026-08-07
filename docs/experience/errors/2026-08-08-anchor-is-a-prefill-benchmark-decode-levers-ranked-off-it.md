# The anchor is a prefill benchmark, and every decode lever was ranked off it — CUDA, 2026-08-08

## Context

`docs/perf-qwen36-27b.md` had named FA3 decode-verify KV bandwidth its #1 open
item, on a derivation: at c=16 and 32K the per-row verify term is ~5.69 ms/row,
a same-binary A/B has excluded the GDN lane, the KV traffic is 34.6 GB whose
floor is 9.9 ms, so the read looked ~9× off roofline. The derivation was never
measured, because **no capture in that document was taken above batch 1**.

One `nsys` capture at the served shape settled it. GPU 6, `70760bc09`, c=16 on
the 32K anchor dataset, a 30 s window at bench elapsed 118–148 s.

## What was confirmed

**FA3 decode-verify runs at 29.2% of achievable bandwidth.**

```
KV traffic / tick   9 rows x 32550 tok x 65536 B  =  19.20 GB
FA3 time / tick                                      18.79 ms
achieved                                             1.02 TB/s  = 29.2% of 3.5
per-layer cross-check  9 x 32550 x 4096 B / 1.174 ms = 1.02 TB/s
```

Inside a decode tick it is the largest line, 42.5% of tick GPU-busy. The
derived range of 0.38–0.68 TB/s was too pessimistic; the mechanism is real.

## Root cause of the mis-ranking

**All decode together is a 2–6% share of GPU time on the workload the ranking
used.**

```
window: GPU busy 28676 ms of 29642 ms wall (96.7%), idle 3.26%
  FP8 GEMM, prefill shapes        59.4%
  FA3 prefill                     16.3%
  norms + pack_quantize           12.3%
  GDN / gated-delta               10.6%
  FA3 decode-verify                0.39%
  all decode ticks together        0.98%
```

**The window undersamples decode 2–6×, and the two decode rows above must not
be quoted.** The bench has a queueing ramp (TTFT p50 4.2 s, p99 164 s) and the
window sits at bench elapsed 118–148 s, where many requests are still in their
first prefill. Reconciling against run totals — 15,981 generated tokens and 414
steps, so decode ticks are bounded by `15981 / max-tokens-per-tick ≤ D ≤ 414`,
i.e. **143 ≤ D ≤ 414**:

| | window | run-level bound |
|---|---:|---:|
| all decode ticks | 0.98% | **2.1 – 6.1%** |
| FA3 decode-verify | 0.39% | **0.86 – 2.5%** |

Removing FA3 decode-verify entirely moves this row by **1–3%**. The 29.2%
bandwidth figure is unaffected — it is internal to a tick and independent of
how many ticks the window caught.

**This is a second instance of the error the perf chain's own reading rules
warn about**, made on the same day the rules were tightened: a window selected
without checking it against the run.

The cause of the mis-ranking is the dataset shape, not the runtime. Over the
full 128-request run the anchor is **4,452,150 prompt tokens against 15,965
output — 279:1**, or **38.6:1** counting only real prefill work after the
measured 0.876 prefix-cache hit rate (616,303 prefill tokens). Long-agent 32K
× 8 turns is, by GPU work, a prefill benchmark. Every decode number in the perf
chain, and every decode lever in its register, was ranked against it.

**The dataset is not the mistake.** `gen_bench_prompts.py` states it models
measured coding-agent traces (TraceLab, 4,265 Claude Code / Codex sessions:
119K prefix, 875 append, 214 output, 8.8 steps per request). For the workload
ARLE targets, prefill genuinely dominates. What was wrong was pricing decode
levers on it.

The latency view hid this. The document read "decode is 95.1% of a c=16
request's latency" off the anchor, but with 128 requests queued on 16 slots
that figure is queueing: `itl_p50` is **0.0197 ms** against `itl_mean`
**176.99 ms**, and `e2e_p99` is 199 s.

## Also corrected by the same capture

| finding | measurement |
|---|---|
| rows per tick is **9**, not the nominal 16 | three independent counts: `rms_norm_batched_offset` gridX 54 ÷ block 6; `prefill_attention_paged_hd256` 144/tick = 9 × 16 layers; `add_native` gridX 270. Most slots are in prefill. Repeats an earlier 11.0-at-nominal-16. |
| `gated_delta_rule_prefill_recurrent_kernel` is **live on the decode path** | 7.75 ms/tick, 16.6% of tick busy, 48 launches — despite the name, and despite being retired from the prefill path on 08-02 |
| `accept_rate` **0.478** | against 0.31 recorded 08-06; acceptance is not the constraint here |
| decode ticks have no launch-gap problem | 11,422 gaps all in 0–5 µs; every bin ≥20 µs empty inside a tick |
| the prefix sidecar still fires | 1,123 D2H at max payload 3,145,728 B, 1.69 GB in 30 s |

Decode tick composition, 6 ticks agreeing to ±0.1%:

```
tick span 50.12 ms | busy 46.62 ms | 1489 kernels
FA3 full-attn        19.81 ms  42.5%   n=192
FP8 GEMM             14.64 ms  31.4%   n=352
GDN / gated-delta     8.80 ms  18.9%   n=144
norms + elementwise   3.37 ms   7.2%   n=689
```

## Fix

The perf chain now carries the measured window with its run-level correction,
the 38.6:1 post-cache ratio, and a re-ranked open-items list whose #1 is
re-anchoring decode on a decode-shaped workload with the capture window placed
after the queueing ramp. Its provenance table records, per measurement table, the date, the
batch, and that all of them run the anchor.

## Rules

**A kernel's distance from roofline and its share of GPU time are different
numbers, and only the second one ranks it.** FA3 decode-verify is 29.2% of
roofline and a 0.86–2.5% share of GPU time. Both were true the whole way
through; the ranking used the first.

**Reconcile a capture window against the run totals before quoting a share.**
This window caught 0.20 decode ticks/s against a run rate of at least 0.44, and
the check is one division against numbers the bench already prints. Writing the
rule down in the morning did not stop me from breaking it in the evening; the
rule needs to be a step in the analysis, not a caution in a preamble.

**Check the token ratio of a benchmark before ranking a phase off it.** One
division — prompt tokens over output tokens — would have shown 279:1 at any
point, and it is printed in every bench CSV this repo produces.

**A derivation that has never been measured is not evidence, however many
independent facts agree with it.** Four adversarial reviews and three
repo-internal measurements all pointed at the per-row KV read, and they were
right about the mechanism and wrong about the size by roughly an order of
magnitude, because none of them fixed the denominator.
