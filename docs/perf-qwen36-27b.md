# Qwen3.6-27B — performance chain, 1×H20

Where a request's wall clock goes, stage by stage, with the measurement behind
each number. Companion to [`architecture-dsv4.md`](architecture-dsv4.md), which
describes the DSv4 execution paths; this document describes the Qwen3.6-27B
**cost** of those paths.

**Reading rules.**

- Every number carries its date, its commit, and how it was obtained. A number
  without those three is not in this document.
- Prefill numbers state **cold or warm**. The same 33K prompt runs 35.1 s cold
  and 0.525 s warm through the prefix cache (2026-08-01) — a prefill figure
  without that label means nothing.
- Decode and prefill are opposite regimes and never share a conclusion. Prefill
  is compute-bound at 83.9% GPU-busy (§1.1); a plain decode **step** is 16.7%
  idle (§2.1), while the 50.5% figure in §4.1 is a bench **window** whose idle
  is inter-request stalls — the two are not interchangeable. A measurement that
  does not name its phase supports no claim about the other
  ([error](experience/errors/2026-08-07-measured-prefill-concluded-about-decode.md)).
- Shares are **of the window that was measured**, and a window is not quotable
  until it is reconciled against the run totals — the 08-08 anchor capture
  undersampled decode 2–6× by landing in a queueing ramp
  ([correction](experience/errors/2026-08-07-named-a-call-site-whose-gate-was-off.md),
  [entry](experience/errors/2026-08-08-anchor-is-a-prefill-benchmark-decode-levers-ranked-off-it.md)).
- **A kernel's distance from roofline and its share of GPU time are different
  numbers, and only the second ranks it.** Both failure directions are now on
  record: draft attention was priced out at a 4.3% share where it is 30.5%
  (§2.0), and the FP8 GEMM was held open at 64–67% of peak where it runs 93%
  (§1.3). Before pricing any lever, name the workload and the batch.

**Provenance of every measurement table.** The recurring failure in this
document has been a table measured at one configuration and read as current.
Each row below carries the date, the batch, and whether a later default flip
invalidated it. **Check this table before ranking anything off a number.**

| § | table | date | batch | context | status |
|---|---|---|---|---|---|
| §1.1 | FP8 prefill budget | 08-01 | 1 | 33K cold | **stale** — pre chunked-GDR default flip; use §1.2 |
| §1.2 | W8A16 prefill vs SGLang | 08-05 | 1 | 33K cold | current |
| §2.1 | plain decode step budget | 08-01 | 1 | short | **stale** — its #2 kernel was deleted 08-03; only the 8.9 ms weight-read floor survives |
| §2.2 | W8A16 decode ledger vs SGLang | 08-03 | 1 | short | **before-state only** — the program it motivated shipped the same day, 26.88 → 21.37 ms |
| §2.3 | decode throughput vs batch | 08-07 `70760bc09` | 1–16 | 32K | current; the column is `B / TPOT`, decode-only, not end-to-end `out tok/s` |
| §2.4 | sampling penalty | 08-07 `7b8a66603` | 1, 8, 16 | 32K | current |
| §3 | DSpark tick phase split | 08-07 `7b8a66603` | 11.0 rows at nominal 16 | short | split current; the `22 ms + 2.48/row` fit is superseded by a per-row fit |
| §4.1 | launch-gap histogram | 08-07 | c=16 window | short | current, and it is a **window** not a step |
| §4.2 | prefix sidecar | 08-07 | 1–16 | short | current |
| §4.4 | memory ledger | 08-07 | 16 slots | — | current |
| §5.0 | anchor row | 08-07 `70760bc09` | 1, 8, 16 | 32K | current |
| §5.2 | vs SGLang, W8A16 | 08-06 | — | 33K | current |
| ceiling | c=16 window, anchor workload | 08-08 `70760bc09` | 9 rows/tick at nominal 16 | 32.5K | current, but the window undersamples decode 2–6× — see the note there |
| §2.0 | c=16 window, **decode-shaped** workload | 08-08 `70760bc09` | **16.0 rows/tick**, four counts | 2.5K | **current — the decode baseline**, window reconciled to +6.8% |
| §1.3 | FP8 GEMM decomposed, per layer | 08-08 `70760bc09` | c=16 | 32.5K anchor | **current — the prefill baseline.** Read off the anchor trace, no new GPU time |
| model | exact prefill/decode ledger, floors, tail bandwidth | 08-09 `70760bc09` | c=16 | 32.5K anchor | **current — supersedes the 251.9 µs/token estimate.** A partition of the window, not a fit; same trace, no new GPU time |

The stale and before-state rows remain as history. Current prefill decisions use
the anchor partition and §1.3; current decode decisions use §2.0 and the c-sweep
rows. The anchor is 279:1 prompt:output and all decode together is 2–6% of its
GPU time. The anchor and decode-shaped workloads disagree by 17× on
full-attention's share of a decode tick, so a decode lever must be priced at the
context it will run at.

**Model and device constants** used throughout:

| | |
|---|---|
| layers | 64 = **16 full-attention** + **48 linear** (gated delta) |
| full-attn KV cell | 65536 B/token (16 layers × 4 kv-heads × 256 head-dim) |
| recurrent state | 146.8 MiB per slot = 48 × (3 MiB gdr f32 + 60 KiB conv bf16) |
| weights, FP8 | 31.2 GB |
| H20 SMs | 78 |
| H20 HBM read | **3.5 TB/s achievable** (measured 2026-07-10), 4.02 TB/s spec |
| H20 FP8 / BF16 peak | ~296 / ~148 TFLOPS |

---

## The model

This model accounts for GPU kernel work. End-to-end wall remains the acceptance
metric because queueing, CPU work, copies, and GPU kernels overlap; they cannot
be added without a full-run critical-path timeline.

```
T_kernel = sum_s c_prefill(Q_s, L_s, shape_s)
         + sum_t c_tick(rows_t, accepted_t, L_t, mode_t)

s          one issued prefill segment
Q_s        query tokens in that segment
L_s        effective KV depth for that segment
shape_s    GEMM and recurrent shapes selected for that segment
t          one observed decode tick
rows_t     rows present in that tick
accepted_t accepted tokens for each row in that tick
mode_t     plain or speculative decode
```

The sums use the issued segment and tick sequence from the trace or runtime
counters. `generated / (mean rows x mean accepted)` is invalid when rows and
acceptance co-vary, as they do during ramp, mixed prefill/decode, and drain.

For a treatment with full-run effective share `s_run` and speedup `f`, the
matched-run prediction is `delta_wall = s_run x (1 - 1/f)`. `s_run` is specific
to that treatment and workload. A narrow-window share is diagnostic evidence
until a full-run trace or matched A/B establishes its run-level value.

### The anchor window, resolved exactly — 2026-08-09, `70760bc09`

Every kernel in the `nsys` c=16 capture is assigned to prefill or decode by
whether its start falls inside one of the seven decode windows, so the ledger
below is a **partition of the window, not a fit**. Nothing is unattributed.

| | |
|---|---|
| wall | 29,642 ms |
| GPU busy | 28,676 ms (idle 966 ms, 3.3%) |
| kernel time | 28,601 ms = **28,168 prefill + 433 decode** |
| prefill tokens | **90,208** in 44 chunks of 2048, launched as 55 segments |
| decode tokens | **349** in 6 passes |
| `c_prefill` | **312.3 µs/token** |

The chunk structure is read off `silu_mul`, whose grid is `(68, tokens)` with
`68 x 256 = 17408 = intermediate_size`, cross-checked against `split_qkv` and
`paged_hd256`, whose `gridY` sums are identical to within 0.01%. A chunk of 2048
tokens spanning two sequences is issued as two segments, which is why 44 chunks
produce 55 prefill segments and `977 = 16 x (55 + 6)` full-attention launches.

**This replaces two earlier counts, both wrong.** §1.3 said 176 chunks, inferred
from duration clusters; a later note said 33, which was a GEMM launch count read
as a `silu_mul` count. The three-way cross-check above settles it at 44.

### `pack_quantize` run-level calibration

Every share below comes from a 30 s steady-state window. The matched A/B below
calibrates one term, `pack_quantize`, against the full run.

`pack_quantize` was 7.75% of the window's kernel time. It was then made **5.12x
faster in situ** (confirmed by capture, below), and a matched counterbalanced A/B
on the full 409 s run measured **wall -2.98%**. A term of run-level share `s`
sped up `f` times returns `s(1 - 1/f)` of wall:

```
1 - 1/5.12 = 0.805        2.98% / 0.805  =  s = 3.70% run-level
window share of wall = 7.75% x 96.4% busy = 7.47%
7.47 / 3.70  =  2.02x
```

This makes `pack_quantize` 2.02x more concentrated in the selected window than
in the run. The A/B's 0.5% BASE spread puts that ratio in a 1.7-2.4x band.

No reusable `0.5` multiplier follows from one treatment. FA3 changes with KV
depth, decode changes with rows, and a wall delta can change the schedule. The
full-run kernel/host/queueing split remains unknown until a full-run timeline is
partitioned. The capture placement provides a hypothesis for the mismatch: it
sits 31-38% into the run with all 16 slots saturated on prefill, while ramp and
drain run at lower batch.

### `c_prefill` — the exact per-token ledger

Prefill kernel time divided by 90,208 tokens. `n` is launches; a GEMM's `K` is
read from the `pack_quantize` that feeds it and its `N` from the consumer's grid.

| term | n | ms | µs/token | share |
|---|---:|---:|---:|---:|
| `gate_up` (K 5120 → N 34816) | 3523 | 7562.6 | 83.8 | **26.9%** |
| FA3 | 881 | 4646.4 | 51.5 | **16.5%** |
| `down_proj` (K 17408 → N 5120) | 2819 | 3807.0 | 42.2 | 13.5% |
| linear-attn `in_proj` (K 5120 → N 16384) | 2115 | 2583.5 | 28.6 | 9.2% |
| ~~`pack_quantize`~~ | 14095 | 2205.2 | 24.4 | ~~7.8%~~ → **1.54%** (fixed, see below) |
| **`fq_fwd` (FlashQLA)** | 2643 | 1617.6 | 17.9 | **5.7%** |
| `out_proj` (K 6144 → N 5120) | 2820 | 1361.9 | 15.1 | 4.8% |
| `qkv` (K 5120 → N 14336) | 881 | 800.5 | 8.9 | 2.8% |
| `conv1d` | 2931 | 587.5 | 6.5 | 2.1% |
| `silu_mul` | 3523 | 586.1 | 6.5 | 2.1% |
| `split2` | 5286 | 357.7 | 4.0 | 1.3% |
| `fq_kkt` | 2643 | 279.7 | 3.1 | 1.0% |
| `rms_norm_gated` | 2643 | 257.3 | 2.9 | 0.9% |
| `gdr_fq_prep` | 2643 | 207.3 | 2.3 | 0.7% |
| remainder (small GEMM variants, `nvjet`, norms, adds, paged gate) | — | 1308.1 | 14.5 | 4.6% |
| | | **28,168.4** | **312.3** | 100% |

The earlier version of this table was a 251.9 µs/token point estimate that closed
only 79% of the window. It is superseded: the residual was not one missing effect
but the small-`M` GEMM variants, the elementwise tail, and a `pack_quantize`
count that was low by 40%.

### The floor layer — what the cost *must* be

A share says how big a term is. It does not say whether the term can be made
smaller, and ranking by share alone is how this document repeatedly picked work
with no headroom in it. Each term therefore carries a floor: the same work
against whichever hardware limit binds it.

**Arithmetic.** GEMM floors are `2·K·N·layers / 296 TFLOPS` per token — config
dimensions and the FP8 peak, no fitting.

| term | floor µs/tok | measured | of FP8 peak | headroom |
|---|---:|---:|---:|---:|
| `in_proj` | 27.2 | 28.6 | **95.1%** | 1.4 |
| `gate_up` | 77.1 | 83.8 | **92.0%** | 6.7 |
| `qkv` | 7.9 | 8.9 | 88.8% | 1.0 |
| `down_proj` | 38.5 | 44.6 | 86.3% | 6.1 |
| `out_proj` | 13.6 | 15.8 | 86.1% | 2.2 |
| | **164.3** | **181.7** | | **17.4 = 1570 ms** |

**Attention.** FA3's cost is dead linear in KV depth, which resolves the
denominator the persistent grid hides. Sorting the 977 launches by their segment
length gives a clean ladder: at `Q = 2048` each additional 2048 tokens of KV
depth costs **732 µs**, over nine consecutive rungs, ±0.8%.

```
FLOP per rung  =  4 x 2048 x 2048 x 24 heads x 256 head_dim  =  1.031e11
                  / 732 us  =  140.8 TFLOPS  =  95.1% of the 148 bf16 peak
```

Mean effective KV depth over the prefill segments is **18.0K**, and per-launch
depths run 5.9K–30.5K, consistent with a 32K anchor. **FA3 has 227 ms of
headroom and the open item is closed** — the previous version of this section
called this "the highest-value measurement in the document" and offered 14.8K as
the break-even depth; the measured depth is above it, so the kernel is at peak.

**FlashQLA `fq_fwd` — 13.5% of the bf16 roofline, and the gap is structural.**
Derived 2026-08-09 from `tools/tilelang/flashqla_gdr.py` plus the trace grid. The
grid `(96, 1, 512)` is `ceildiv(DV 128, block_DV 64) × H`, so **H = 48 heads** —
cross-checked against `rms_norm_gated`'s grid, which puts the gated-delta output
at 48 × 128 = 6144. Six GEMMs per 64-token chunk per CTA:

| GEMM | M×K×N | MACs |
|---|---|---:|
| `h += kᵀ @ vd` | 128×64×64 | 524,288 |
| `u = k @ h` | 64×128×64 | 524,288 |
| `a @ u` | 64×64×64 | 262,144 |
| `p = q @ kᵀ` | 64×128×64 | 524,288 |
| `o = q @ h` | 64×128×64 | 524,288 |
| `o += p @ vd` | 64×64×64 | 262,144 |

```
7.864e6 FLOP/token/layer x 48 layers x 90,208 tokens = 3.405e13 FLOP
  / 148 TFLOPS bf16 = 230 ms floor      measured 1708 ms = 13.5% of peak
```

**But most of that 1478 ms is not waste.** Two structural limits, both derived,
neither measured:

- **Wave quantization.** 96 CTAs on 78 SMs: 18 SMs take two CTAs and set the
  critical path, so the ceiling is `96 / (2 × 78) = 62%`, lifting the floor to
  371 ms. This one is checkable and changeable — `block_DV = 32` gives
  `4 × H = 192` CTAs, 82% ceiling, at the cost of computing the DV-independent
  `p = q @ kᵀ` four times instead of twice (+10% work for +32% wave efficiency).
- **Dependency chain.** A 2048-token segment is **32 serial chunks**, each six
  dependent GEMMs at 64×128×64 — Hopper `wgmma`'s minimum M. The recurrence
  `h → u → o` carries almost no ILP. Per chunk step: 3.4 µs ideal against 19.1 µs
  measured, **18% efficient**.

**`fq_fwd` is `T_stall`'s only large candidate and the bucket has never been
measured.** Its 1478 ms nominal gap is larger than any remaining individual tail
row. Its dependency structure provides a concrete hypothesis; the reclaimable
fraction remains unknown until `ncu` splits the warp stall reasons.

**Consequence: prefill arithmetic is finished.** Every GEMM is at 86–95% of FP8
peak and attention is at 95% of bf16 peak. Together they are 74.6% of prefill
time with **1797 ms of headroom, 6.1% of selected-window wall**. Nothing in this
document should propose a faster matrix kernel again.

### The tail — the same trace, with traffic lower bounds

The kernels outside GEMM and attention are 23.3% of prefill time. They quantize,
split, normalize, convolve, and add. `bytes / 3.5 TB/s` is a traffic lower bound;
it establishes neither the binding resource nor reclaimable headroom.

| kernel | ms | GB moved | effective TB/s | of HBM | traffic lower bound ms |
|---|---:|---:|---:|---:|---:|
| ~~`pack_quantize`~~ **fixed `5cfe8494f`** | 2216 → **441** | 593 | 0.27 → **1.35** | 7.6% → **38.5%** | 170 |
| `silu_mul` | 589 | 605 | 1.03 | 29.4% | 173 |
| `conv1d` | 590 | 222 | 0.38 | 10.7% | 63 |
| `split2` | 360 | 289 | 0.80 | 23.0% | 83 |
| `rms_norm_gated` | 259 | 160 | 0.62 | 17.7% | 46 |
| `gdr_fq_prep` | 207 | 107 | 0.51 | 14.7% | 30 |
| `add_native` | 117 | 46 | 0.39 | 11.3% | 13 |
| `split_qkv` | 101 | 83 | 0.83 | 23.6% | 24 |
| `rms_norm_batched` | 105 | 6 | 0.06 | 1.6% | 2 |
| | **4544** → **2851** | 2111 | | | **603** |

`pack_quantize` removed 1775 ms from the selected window. The other eight rows
sum to **2248 ms, 7.9% of window kernel time**. Their run-level share and
reclaimable fraction are unmeasured. Byte counts come from grid dimensions and
the read/write pattern; `conv1d` and `split2` carry the largest traffic error.

`pack_quantize` at `dsv4_deepgemm_ops.cu:65` used one 128-thread block per
128-element quantization block, one `uint16_t` per thread, a shared-memory
reduction, and a second input read. The shipped 16-lane form uses 16 B loads,
register reuse, shuffle reduction, and packed FP8 conversion: **5.13× on the
microbench, 5.12× in situ, bit-identical**.

The source audit separates the remaining rows into different mechanisms:

| kernels | current implementation | next evidence |
|---|---|---|
| `silu_mul`, `add_native` | four bf16 per thread via `uint2`; SiLU also executes FP32 `exp` | `ncu` instruction and pipe split |
| `split2`, `split_qkv` | eight bf16 per thread via `uint4` | achieved bandwidth and launch cost |
| `conv1d` | one channel/token per thread, causal loop plus separate state update | memory pattern, instruction mix, state-update share |
| `rms_norm_*`, `gdr_fq_prep` | reductions and recurrent preparation | per-kernel stall and instruction split |

The `pack_quantize` repair is specific to that kernel. Each row needs one
profile at the traced shape before implementation. Fusion remains a
candidate only where an existing producer can emit the required layout without
adding a new intermediate or changing numerical order.

### The gap layer — four buckets, four distinct remedies

A floor says what the work must cost. It does not say where the difference went,
and "where it went" is what picks the next task. The machine is, at every
instant, either doing required work, doing work the implementation added, or not
issuing at all:

```
T_actual  =  T_floor  +  T_inflation  +  T_stall  +  T_idle

T_floor       max over resources of required_work / capacity — irreducible
T_inflation   work the implementation added: quantization round trips,
              materialized intermediates, re-reads, recompute
T_stall       warps resident but not issuable — the data they need is in
              flight from HBM and there is not enough concurrency to cover it
T_idle        nothing resident: launch gaps, syncs, host waits
```

The buckets matter because their remedies do not overlap. **Inflation** is paid
down by fusion, dtype choice, and caching. **Stall** by async copy / TMA, warp
specialization, deeper software pipelining, more occupancy. **Idle** by CUDA
graphs, persistent kernels, stream overlap, larger batches. **Floor** only by
doing less work.

This is also why almost any change appears to help. Fusion attacks inflation and
idle at once, and most kernels have slack in both, so a win arrives without
anyone learning which bucket paid — which is indistinguishable from guessing.

### The buckets, measured on the anchor window

Two captures, same 30 s steady-state window, same analyzer: `70760bc09` (before)
and `5cfe8494f` (after the `pack_quantize` fix). Wall 29,642 / 29,693 ms, GPU
busy 96.5% / 96.4%, kernel 28,601 / 28,611 ms — like for like.

| bucket | selected-window amount | of kernel time | run-level status | how it was obtained |
|---|---:|---:|---|---|
| tail, 8 kernels | **2248 ms observed** | 7.9% | unmeasured | trace sum; binding resources unclassified |
| `fq_fwd` nominal gap | **1478 ms derived** | 5.2% | unmeasured | 13.5% of bf16 roofline; reclaimable fraction unknown |
| floor gap — all GEMM | 1570 ms derived | 5.5% | unmeasured | 17.4 µs/token × 90,208 |
| idle | 966 ms observed | 3.4% | selected window only | GPU busy 28,676 of 29,642 ms |
| floor gap — FA3 | 227 ms derived | 0.8% | unmeasured | 4.9% short of the 148 TFLOPS bf16 peak |

`pack_quantize` is the only tail row with a classified gap and a full-run A/B.
It removed 1775 ms from the selected window and moved full-run wall by −2.98%.
Its gap was `T_inflation`; `ncu` settled that against the initial memory-stall
hypothesis:

| metric | before | after |
|---|---:|---:|
| duration | 100.7 µs | **20.6 µs (4.89×)** |
| executed instructions | 46.61 M | **7.95 M (5.87× fewer)** |
| SM (compute) throughput | 81.5% | 69.2% |
| **DRAM throughput** | **5.3%** | 25.7% |
| achieved occupancy | 90.3% | 84.6% |
| executed IPC | 3.32 | 3.07 |

**The speedup equals the instruction reduction at every step.** Both versions run
the SM at 69–81% of issue throughput with IPC above 3.0 and occupancy near 90% —
the kernel is not waiting on memory, it is issuing address arithmetic, reduction,
and synchronization instructions at nearly the hardware rate. DRAM at 5.3% means
the memory system is close to idle while the old version runs.

For `pack_quantize`, TMA, async copy, and warp specialization target the wrong
bucket. Its binding floor is instruction issue: `instructions / issue_rate`.
The traffic lower bound above remains useful as a sanity check. The other eight
rows have no bucket assignment yet.

**`fq_fwd` is the opposite case and the reason the buckets are kept separate.**
Same nominal size, no demonstrated remedy, and a dependency structure that
`T_stall` is exactly the name for. Measuring it is the next thing this document
needs.

### What this model is for, and what it forbids

It ranks work by `share x achievable improvement`, both of which it now makes
explicit. Three consequences follow:

- **Prefill arithmetic is at the hardware floor.** GEMM 86–95% of FP8 peak, FA3
  95% of bf16 peak, 74.6% of prefill time, 6.1% of selected-window wall in
  headroom. Only `P` can make it smaller.
- **The data-prep tail contains the largest unclassified kernel time.**
  `pack_quantize` was 7.8% of prefill time and is now 1.54%. Eight kernels with
  distinct implementations remain; their observed time is 7.9% of this window.
- **A window share is diagnostic.** Full-run prediction requires the lever's
  own run-level share. `pack_quantize` established 3.70% for itself and no
  multiplier for other terms.
- **A decode-side lever is summed over the observed ticks.** The selected anchor
  window contains only 433 ms of decode kernel work in 28,601 ms. This is why
  slot-batched draft attention is −10.4% on a decode-shaped workload and a null
  here; the model predicted it before the A/B ran.

It also forbids the error this document made three times in two days: a
coefficient measured at one `(c, L, chunk)` cannot be read at another. Every row
above carries its shape.

---

## Where the ceiling is — the roofline of the shape production actually runs

**This section is a derivation, written when every kernel measurement in the
document was batch 1** (§2.1 plain single-row decode steps, §2.2 against SGLang
at `bs=1` in a graph) and the served c=16 / 32K shape had never been captured.
It is kept because the derivation is what motivated the captures, and because
comparing it against them is the point.

**Three c=16 captures now exist and they settle it**: the anchor window below,
§2.0 (decode-shaped), and §1.3 (the FP8 GEMM decomposed). Where this section's
derivation disagrees with them, they win — read the "Measured" subsection
before using any number above it.

**Scope of the metric.** FLOPs below are GEMM only — `2 × 22.3e9` per token,
using the same 22.3 B GEMM-parameter count §1.2 uses for the prefill roofline.
Attention FLOPs are excluded: they run bf16 against a 148 TFLOPS peak and cannot
be scored against the 296 TFLOPS FP8 denominator. Bandwidth is the 3.5 TB/s
achievable figure from the constants table.

### The batching crossover is at 7 rows, and c=16 is past it

DSpark verify pushes `rows × 6` draft tokens through **one** weight read. A GEMM
stops being memory-bound when compute time overtakes that read:

```
N* = 296 TFLOPS / (2 x 3.5 TB/s) = 42.3 tokens = 7.05 rows at block 6

rows  tokens   weight read   GEMM compute   bound
   1       6       8.91 ms        0.90 ms   memory
   8      48       8.91 ms        7.23 ms   memory
  11      66       8.91 ms        9.94 ms   COMPUTE
  16      96       8.91 ms       14.46 ms   COMPUTE
```

**Past ~7 rows, adding a row stops amortizing anything and starts costing real
FLOPs.** The anchor runs at 16 rows — 2.3× past the crossover. The headroom in
"share the weight read across more tokens" is largely spent on this workload.

### At the anchor's context, most of the traffic was never amortizable

Per decode step at c=16 and 32K, from the constants table and §4.4:

```
                                                  GB   amortizable?
weights, FP8                                    31.2   yes, shared by all rows
full-attn KV  65536 B/tok x 32768 x 16 rows     34.4   no, per row
recurrent state  146.8 MiB x 16                  2.5   no, per row
                                          total 68.0   46% amortizable
```

**The per-row KV read already exceeds the weight read.** Decode at the anchor
workload is not weight-bound; it is KV-bound, and no amount of batching touches
the larger half. The measured verify slope doubling — 2.48 ms/row at short
context, **5.18 ms/row at 33K** (§3) — is this term appearing.

### Consequence: the headroom is roofline at the batched shape, not batching

Measured verify against its own floor, `max(weight read, GEMM compute)` plus the
per-row KV read where context makes it non-negligible:

```
                measured   floor   off roofline
c=1   short       24.5 ms  8.9 ms      2.7x
c=11  short       49.3 ms  9.9 ms      5.0x
c=16  short       61.7 ms 14.5 ms      4.3x
c=16  32K        104.9 ms 18.7 ms      5.6x
```

The gap grows with batch. That is the opposite of what a batch-independent cost
produces, and it was invisible in every capture this document contained when
this was written — all of which were batch 1.

### Measured: the mechanism is confirmed and the size is not

`nsys`, 2026-08-08, `70760bc09`, GPU 6, c=16 on the 32K anchor dataset, a 30 s
window at bench elapsed 118–148 s. **This is the document's first decode
capture above batch 1.** Full entry:
[`errors/2026-08-08-anchor-is-a-prefill-benchmark-decode-levers-ranked-off-it.md`](experience/errors/2026-08-08-anchor-is-a-prefill-benchmark-decode-levers-ranked-off-it.md).

**FA3 decode-verify runs at 29.2% of achievable bandwidth.**

```
KV traffic / tick   9 rows x 32550 tok x 65536 B  =  19.20 GB
FA3 time / tick                                      18.79 ms
achieved                                             1.02 TB/s   = 29.2% of 3.5
per-layer cross-check  9 x 32550 x 4096 B / 1.174 ms = 1.02 TB/s  (agrees)
```

The off-roofline half of the claim holds, and the earlier derived range of
0.38–0.68 TB/s was too pessimistic.

**The size does not hold.** In the same window:

```
GPU busy 28676 ms of 29642 ms wall (96.7%), idle 3.26%
  FP8 GEMM, prefill shapes        59.4%  ███████████████████████████████
  FA3 prefill                     16.3%  ████████
  norms + pack_quantize           12.3%  ██████
  GDN / gated-delta               10.6%  █████
  FA3 decode-verify                0.39% ▏
  all decode ticks together        0.98% ▏
```

> **The window undersamples decode 2–6×; use the run-level bound, not these
> two rows.** The bench has a queueing ramp — TTFT p50 4.2 s against p99
> 164 s — and this window sits at bench elapsed 118–148 s, where many requests
> are still in their first prefill. Reconciling against run totals (15,981
> generated tokens, 414 steps, so decode ticks are bounded by
> `15981 / max-tokens-per-tick ≤ D ≤ 414`, i.e. **143 ≤ D ≤ 414**):
>
> | | this window | run-level bound |
> |---|---:|---:|
> | all decode ticks | 0.98% | **2.1 – 6.1%** |
> | FA3 decode-verify | 0.39% | **0.86 – 2.5%** |
>
> The conclusion survives — decode is a single-digit share of GPU time on this
> workload — but the window figures are 2–6× low and must not be quoted. The
> 29.2% bandwidth measurement is unaffected: it is internal to a tick and does
> not depend on how many ticks the window caught. This is the same error the
> reading rules at the top of this document warn about, made again.

Six decode ticks against ~55 chunked-prefill passes, a ratio the run-level
bound says is inverted relative to the whole run. **Removing FA3 decode-verify
entirely would move this row by 1–3%.**

### The anchor workload is prefill-bound, and this document ranks decode off it

The cause is in the dataset, not the runtime. Over the full 128-request run:

| | |
|---|---|
| prompt tokens | 4,452,150 (mean 34,782/req) |
| output tokens | 15,965 |
| ratio | **279 : 1** |

Long-agent 32K × 8 turns is, by GPU work, a **prefill benchmark**. §2 and the
ceiling analysis above are both ranked against a workload where all decode
together is a **2–6%** share of GPU time.

The dataset is not a mistake: `gen_bench_prompts.py` states it models measured
coding-agent traces (TraceLab, 4,265 Claude Code / Codex sessions — 119K prefix,
875 append, 214 output, 8.8 steps per request). Prefix caching recovers most of
it — measured hit rate **0.876**, so 4.45 M prompt tokens become **616 K** of
real prefill work, still **38.6 : 1** against output. **For the workload ARLE
targets, prefill genuinely dominates.** The error was not the benchmark; it was
pricing decode levers on it.

§0.1's "decode is 95.1% of a c=16 request's latency" measures wall clock after
first token with 128 requests queued on 16 slots — `itl_p50` is **0.0197 ms**
against `itl_mean` **176.99 ms**, and `e2e_p99` is 199 s. That is queueing, not
decode compute. The two views must not be added to one conclusion.

**What follows.** Decode levers must be priced on a decode-shaped workload
(short prompts, long generations) and prefill levers on this one. Ranking both
off the anchor is how a 29%-of-roofline kernel became a ~0.4% lever.

### What the capture also corrected

| finding | measurement |
|---|---|
| **rows per tick is 9, not the nominal 16** | three independent counts: `rms_norm_batched_offset` gridX 54 ÷ block 6; `prefill_attention_paged_hd256` 144/tick = 9 × 16 layers; `add_native` gridX 270 = 54 × 5. Most slots are in prefill, not decode. Repeats the earlier 11.0-at-nominal-16. |
| **`gated_delta_rule_prefill_recurrent_kernel` is alive on the decode path** | 7.75 ms/tick, 16.6% of tick busy, 48 launches. The §1.1 stale note is right about prefill and wrong if read as "this kernel no longer runs". |
| `accept_rate` **0.478** | against 0.31 recorded 2026-08-06; acceptance is not the constraint here |
| decode ticks have no launch-gap problem | 11,422 gaps all in 0–5 µs; every bin ≥20 µs is empty inside a tick |
| the sidecar still fires | 1,123 D2H, max payload 3,145,728 B, 1.69 GB in the window |

Inside one decode tick, 6 ticks agreeing to ±0.1%:

```
tick span 50.12 ms | busy 46.62 ms | 1489 kernels
FA3 full-attn        19.81 ms  42.5%   n=192  ██████████████████████
FP8 GEMM             14.64 ms  31.4%   n=352  ████████████████
GDN / gated-delta     8.80 ms  18.9%   n=144  ██████████
norms + elementwise   3.37 ms   7.2%   n=689  ████
```

### Corrections to the previous version of this section

`b3198fb22` framed performance as a product of three factors — time occupancy,
roofline efficiency, arithmetic intensity — and ranked levers by factor. Ten
things in it were wrong, found by four adversarial reviews of that commit:

| claim | defect |
|---|---|
| "a product of three factors" | never formable as tabulated: two cells were levels, the third (`1.23×`) an elasticity. The product was never written because it does not close. |
| ③ ceiling ~12× | `32.8% / 2.6%` divides an all-three-ideal by a fully-measured state, so it multiplied batching headroom by occupancy and roofline headroom. ③'s own share is ~3×, and the KV term above cuts it further. |
| ① = 50.5% busy | that is §4.1's 19.92 s **window**, whose idle is 79 inter-request stalls; the per-step figure is 16.7% (§2.1). The section advertised one and computed with the other — violating this document's own reading rule at the top. |
| "every >2× win is in ③" | false in this file: FlashQLA took linear attention **7.231 → 0.441 s, 16.4×** (§1.2), and it is a kernel replacement. |
| "rank by factor before size" | refuted by that same example: 16.4× on a module worth 23% of prefill yielded −26% end to end. **Factor size × share governs.** A lever's ceiling belongs to the lever, not to a category. |
| "22 ms intercept is the part adding rows does not amortize" | inverted. Per-token verify cost is `22/(R·A) + slope/A`; the intercept **is** the term that falls as rows grow. The slope is what does not. |
| the `22 ms + 2.48 ms/row` model itself | superseded on the current binary (`baselines.md:162`) by a pure per-row fit, **5.69 verify + 3.04 draft**, 85% of the tick. §3 now carries the current one. |
| "36% of verify at c=16" | no measurement produces it: it multiplies a short-context slope by a nominal row count the same capture measured as 11.0. |
| 27e9 params for FLOPs | §1.2 uses 22.3e9 GEMM params for the same purpose. |
| "acceptance may collapse with batch" | already withdrawn in `baselines.md:137` — `accept` tracks cache state, not concurrency (0.532 vs 0.313 at matched c=16). Do not re-file it. |

The ①②③ tags are dropped from §6 for the reason in the "rank by factor" row.
What survives from
that section is the physical observation it was built on: a batch-1 decoder at
100% of achievable bandwidth is still a nearly idle machine. What was wrong was
concluding that batching therefore had the headroom — on this workload it is
already past the crossover, and the headroom moved to the per-row KV read that
batching cannot touch.

---

## 0. The chain

**How to read the figures.** Three things have to be visible at once — what a
stage contains, how long it takes, and how it sits inside the end-to-end time —
and they are carried by two different figure types, which must not be confused:

| type | when | encoding |
|---|---|---|
| **timeline** | the stages really are sequential (§0.1 request, §3 tick) | bar **start** is start time, bar **length** is duration; the row is the whole |
| **budget** | the parts interleave (§1.1 prefill, §2.1 decode step) | every bar starts at the same origin; only **length** means anything |

Drawing an interleaved budget as a timeline would assert an execution order
that does not exist, so those figures are labelled at the point of use. The
flowchart carries containment and routing only — box size means nothing there,
and percentages inside nodes are labels.

**Every share in this document is taken against the sum of the rows shown**, and
the sum is printed. Where that differs from a previously published share, the
difference and its cause are stated on the spot.

**What is in this document.** A number earns a place here if it can change a
decision — what to work on next, or what not to. Everything else stays in the
wins/errors entry it came from and is linked. The four largest numbers in the
chain are all unattributed, and are marked as such rather than filled with a
hypothesis; see *Measurement debt*.

```mermaid
flowchart TB
    subgraph ADMIT["Admission — infer-core/src/planner.rs"]
        A1["HTTP → tokenize"] --> A2["radix prefix match"]
        A2 --> A3["build_forward_plan<br/>budget 16384 tok/tick, ≤16 rows"]
    end

    A3 --> P{"ForwardMode"}

    subgraph PREFILL["Prefill — 94% of GPU time on the anchor (279:1 prompt:output)"]
        direction TB
        B1["chunk 2048 tok"] --> B2["FP8 GEMM — 57.7% of all kernel time<br/>gate_up 33.9% at 93% of peak<br/>down 24.2% at 88% — AT THE FLOOR"]
        B2 --> B3["full attention ×16<br/>FA3 / TileLang — 16.3%"]
        B3 --> B4["linear attention ×48<br/>FlashQLA chunked GDR — 10.6%"]
        B4 --> B5["pack_quantize + norms — 12.3%<br/>prefix sidecar 146.8 MiB per stride"]
    end

    subgraph DECODE["Decode — 2.1-6.1% of GPU time on the anchor, 100% of a decode-shaped one"]
        direction TB
        C1["draft — DFlash backbone, block 6<br/>attention 30.5% of a decode-shaped tick<br/>slot-batched 3a8f99b1f"] --> C2["snapshot recurrent"]
        C2 --> C3["verify — trunk forward<br/>FP8 GEMM 28.8% · GDN 21.0%<br/>full-attn 1.8% at 2.5K ctx, 42.5% at 32.5K"]
        C3 --> C4["accept + commit"]
        C4 --> C5["rollback replay<br/>batched varlen"]
    end

    P -->|Prefill / Mixed| PREFILL
    P -->|Decode / Mixed| DECODE
    PREFILL --> OUT["detokenize → SSE"]
    DECODE --> OUT
    C5 -.->|next tick| C1
```

Prefill and decode rows share a tick (`ForwardMode::Mixed`), but the executor
still decomposes the mixed plan into per-row prefill submissions followed by a
batched decode dispatch (`infer-cuda/src/executor/qwen35.rs:2932`).

### 0.1 Where a request's own latency goes

Anchor workload, 214 output tokens, warm prefix. Bar position is start time and
bar length is duration, both to scale — these two stages really are sequential.

```
c=1                       s   share
prefill (TTFT warm)    0.84   31.7%  ███████████████████
decode 214 x 8.46 ms   1.81   68.3%                     █████████████████████████████████████████
                                     total 2.65 s

c=16                      s   share
prefill (TTFT warm)    1.22    4.9%  ███
decode 214 x 110.52 ms 23.65  95.1%     █████████████████████████████████████████████████████████
                                     total 24.87 s
```

Concurrency moves a request's latency almost entirely into decode. Aggregate
*throughput* on the same workload runs the other way — prefill tokens dominate
the token count, which is why total tok/s scales 10453 → 33780 while decode
tok/s scales 118 → 145 (§2.3). The two views answer different questions and
neither substitutes for the other.

**The four largest costs are measured and none of them is attributed.** Each is
sized precisely and its cause is unknown. Quantized GEMM, DeepGEMM FP8, and
launch gaps have all been measured and priced out (§6) — the FP8 GEMM
decisively so, at 87–93% of peak (§1.3).

| cost | size | what is known | what is not | § |
|---|---|---|---|---|
| per-row verify term at c=16 | **5.69 ms/row, 85% of the tick, 9.2× off roofline** | measured 2026-08-08: FA3 decode-verify achieves **1.02 TB/s = 29.2%** at 32.5K/9 rows and 47% at 2.5K/16 rows | why the achieved bandwidth falls with context when row count moved too — two points cannot separate them | ceiling, §2.0 |
| sampling penalty | −30 to −40% decode tok/s at c ≥ 8 | the concurrency shape is a per-row loop; acceptance-vs-concurrency is withdrawn | the size of the batched-draft gate's share | §2.4 |
| prefill GPU idle | 3.97 s vs SGLang 0.19 s | GPU-busy is within 0.93 s on identical kernels | what the idle is waiting on | §1.2 |
| sidecar writes | 9.4% of wall, 83 GB per bench | the cost, exactly | the restore hit rate, so whether it is earned | §4.2 |

They are ranked by size at the batch that is served, which is why the per-row
verify term leads despite the prefill idle being the larger raw number — see
*Where the ceiling is* above.

---

## 1. Prefill

### 1.1 FP8 weights — 33K cold, single request

`nsys`, 2026-08-01, 28.6 s wall / 24.0 s GPU-busy (**83.9%**).

> **STALE — do not rank prefill work off this table.** Measured before chunked
> GDR went default-on (`c2eb5de9e`, 08-02). Its #1 row is a kernel that no
> longer runs **on the prefill path** at the shipped defaults: FlashQLA chunked
> replaced `gated_delta_rule_prefill_recurrent` there and measured **1.06 s
> against its 9.37 s**, which moves 33% of the budget and reorders everything
> below it. The kernel itself is still live — the 08-08 capture measures it at
> **7.75 ms per decode tick, 16.6%**, 48 launches, despite the name.
> Two more prefill changes landed after (`0ac780495`, `301d0c074`), and
> `b0368426a` routed batch==1 prefill to FA3, which also moves the TileLang
> row. Same flag as [`baselines.md`](baselines.md) carries. A re-measure needs
> one `nsys` capture. **§1.2 is the current prefill decomposition** — it is
> dated 08-05 and has the FlashQLA column.
>
> One consequence is already readable: removing ~8.3 s of GPU-busy while the
> 4.60 s idle row stays put takes idle from 16.3% to **~21% of prefill**. The
> idle item got relatively larger, not smaller.

These kernels interleave across the prefill, so this is a **budget**, not a
timeline — every bar starts at the same origin and only its length is meaningful.

```
                                     s   share   launches
gated_delta_rule_prefill_recurrent 9.37  33.1%   1152  ████████████████████
DeepGEMM FP8, all shapes           8.33  29.5%   7936  ██████████████████
GPU idle (incl. host tokenization) 4.60  16.3%      —  ██████████
TileLang full attention            3.93  13.9%    368  ████████
pack_quantize bf16 -> fp8          1.50   5.3%   9600  ███
conv1d / norm / silu               0.55   1.9%   3840  █
                                                       total 28.28 s
```

Plus **2328 `cuMemcpyDtoH` costing 1.58 s** — host round-trips inside the
prefill loop.

Per-part ceilings decide where work is worth doing:

| part | achieved | ceiling | verdict |
|---|---|---|---|
| DeepGEMM `gate_up` / `down` | 199 / 189 TFLOPS | ~296 FP8 | 64–67% at THIS shape; **93 / 88% at the served c=16 shape — see §1.3** |
| full attention | 54 TFLOPS | ~148 BF16 | 36%, and on TileLang rather than the FA3 decode already uses |
| `gated_delta_rule_prefill_recurrent` | — | — | not compute-bound: `<<<48, …>>>` on **78 SMs**, scanning the sequence serially inside each block |

The recurrence was the largest single line and the only one with headroom.
FlashQLA chunked GDR, parameterized over (H, Hg), took 33K cold prefill
**28.95 → 21.63 s (−26%)** and is default-on since `c2eb5de9e`; `b0368426a`
then routed batch==1 prefill to FA3 for a further −4%.

### 1.2 W8A16 weights — 33K cold, versus SGLang

`nsys --cuda-graph-trace=node`, 2026-08-05, H20 GPU 6, SGLang 0.5.13 on a
mechanically repacked GPTQ twin: **same int8 values, same `gptq_marlin`
kernel**. Prefill is not captured into a graph on either stack.

| bucket | ARLE (stub) | ARLE (FlashQLA) | | SGLang | |
|---|---:|---:|---|---:|---|
| Marlin GEMM (8448 launches) | 18.660 s | 18.660 | `████████████████████████████` | 18.675 | `████████████████████████████` |
| GPU idle | **3.877** | **3.967** | `██████` | **0.190** | `▏` |
| full attention (FA3) | 1.632 | 1.633 | `██` | 1.529 | `██` |
| other | 0.361 | 1.108 | `██` | 0.422 | `▊` |
| linear attention + conv1d | **7.231** | **0.441** | `▊` | 0.314 | `▌` |
| **wall** | 31.76 | **25.84** | | **21.13** | |

The two bar columns are on the same scale, so the only visible difference
between the stacks is the idle row.

Three conclusions that hold for both weight paths:

- **Quantized GEMM is not a lever.** Identical kernel, identical launch count,
  15 ms apart across stacks. It is 58–88% of prefill on both.
- **`--chunked-prefill-size` is not a lever.** ARLE 2048 vs 4096 and SGLang
  4096 vs 8192 all land inside 0.07 s TTFT.
- **The remaining gap is 3.8 s of GPU idle** — ARLE 3.97 s against SGLang
  0.19 s, with GPU-busy time within 0.93 s. Scheduling or host-side, not a
  kernel.

Roofline: ~1.68 PFLOP for a 33K prefill (22.3 B GEMM params × 2 × 33e3, plus
0.21 PFLOP causal full attention over 16 layers) against 148 TFLOPS BF16 →
**11.4 s floor**. SGLang 54% MFU, ARLE 46% after the FlashQLA fix.

---

### 1.3 The anchor's biggest line, decomposed — it is at ~90% of FP8 peak

The `nsys` c=16 anchor capture, `70760bc09`, same window as §"Measured". Read
off the trace with no new GPU time; the launch sequence is exactly periodic, so
each GEMM is identified by the kernel it sits between and by the dimension of
the `pack_quantize` that feeds it.

`sm90_fp8_gemm_1d2d_impl` is **57.7% of all kernel time** (16.51 s of 28.60 s),
in one launch shape — grid (78, 1, 1), i.e. one persistent block per SM, which
is DeepGEMM's design and not a starved grid. 14,094 launches resolve into
**four GEMMs per layer**, the periodic unit being a 2048-token prefill chunk:

```
pack K=5120   -> GEMM  503.7 us   attention out_proj   (K 6144 -> N 5120)
pack K=5120   -> GEMM 2646.8 us   gate_up              (K 5120 -> N 34816)
                 silu_mul (68, 2048)
pack K=17408  -> GEMM 1409.0 us   down_proj            (K 17408 -> N 5120)
pack K=5120   -> GEMM 1256.7 us   linear-attn in_proj
                 split2 -> conv1d -> fq_fwd
```

`silu_mul` grid (68, 2048) pins the shape: 68 × 256 = 17408 = `intermediate_size`
per token over a 2048-token chunk, so `gate_up` is the fused 2×17408 projection.
`hidden_size` 5120, `intermediate_size` 17408, 64 layers.

**`silu_mul`'s `gridY` is the token counter, and it closes the window's ledger.**
An earlier version of this section said "176 prefill chunks", inferred from
duration-cluster counts; that was wrong. The grids give it directly:

| `silu_mul` grid | launches | tokens each | what |
|---|---:|---:|---|
| (68, 2048) | 2115 | 2048 | full prefill chunks |
| (68, 1712) + (68, 336) | 512 + 512 | 2048 together | chunk split across two requests |
| (68, 1696) + (68, 352) | 192 + 192 | 2048 together | same |
| (68, 54) | 414 | 54 | **decode** — 9 rows × block 6 |

Σ (`gridY` × launches) ÷ 64 layers = **90,208 prefill tokens and 349 decode
tokens — 258:1**, an independent confirmation of the anchor's 279:1 derived from
kernel grids rather than the bench CSV.

| GEMM | share of FP8 GEMM | TFLOPS | of ~296 FP8 peak |
|---|---:|---:|---:|
| `gate_up` | **33.9%** | 275.9 | **93.2%** |
| `down_proj` | 24.2% | 259.1 | 87.5% |
| attention `out_proj` | 8.7% | 255.8 | 86.4% |
| linear-attn `in_proj` | 21.6% | 273.5 | 92.4% (N 16384, from `split2`'s grid) |

MLP alone (`gate_up` + `down_proj`) is **69.7% of the FP8 GEMM time and ~40% of
all GPU kernel time** on the anchor.

**The two layer types, read off the same periodic sequence:**

```
linear layer (×48)   out_proj 503.6 -> gate_up 2645 -> silu -> down 1409
                     -> in_proj 1255.7 -> conv1d 277 -> fq_fwd 773 us
full-attn layer (×16) qkv 1117.0 -> paged_hd256 109.5 -> FA3 7231.7
                     -> out_proj 503.6 -> gate_up 2645 -> silu -> down 1409
```

`device_kernel` — 16.6% of kernel time, the second-largest line — demangles to
`cutlass::device_kernel<flash::FlashAttnFwdSm90<…>>`, i.e. **FA3**, one launch
per full-attention layer per chunk. 977 launches, p50 **5.03 ms**, spanning
1.18 → 10.92 ms as the chunk's context depth grows.

**Scored 2026-08-09: 140.8 TFLOPS, 95.1% of the bf16 peak.** The grid is
(78, 1, 1) — persistent, one block per SM — so sequence lengths are absent from
the trace and the denominator had to come from elsewhere. Joining each launch to
the `split_qkv` that precedes it gives its `Q`; sorting the `Q = 2048` launches
by duration gives a ladder whose step is **732 µs per 2048 tokens of KV depth**,
constant to ±0.8% over nine rungs. That step is one clean rectangle of attention
work, `4 × 2048 × 2048 × 24 × 256 = 1.031e11` FLOP, so the rate follows without
knowing any absolute depth. Mean effective depth is **18.0K**, per-launch range
5.9K–30.5K.

The apparent chunk-count disagreement — `silu_mul` 2115/64 = **33**, FA3 977/16 =
**61** — was two different quantities. 33 is full 2048-token chunks; 61 is
*segments*, because a chunk spanning two requests is issued as two launches:
33 + 8 + 8 + 3 + 3 prefill + 6 decode = 61, and 44 chunks in total.

**This kills the largest open item.** §1.1 records `gate_up` / `down` at 199 /
189 TFLOPS, "64–67% — leave alone", and that number was measured at **33K cold,
single request**. At the served c=16 shape the same kernels run at **87–93% of
FP8 peak**. The anchor's dominant cost is not inefficient; it is the hardware
floor. There is no GEMM lever here — the only way to reduce it is to do less
matrix work (prefix cache hit rate, sparsity, shorter effective context), not
faster matrix work.

Same shape error as the draft attention, in the opposite direction: a number
measured at one shape was governing a decision at another. There it understated
a lever by 7×; here it overstated one.

---

## 2. Decode

### 2.0 Decode-shaped workload, c=16 — the current decode baseline

`nsys`, 2026-08-08, `70760bc09`, GPU 6, 16 sessions × 1 turn, ~1.1K prompt /
1.8K generated (**1 : 1.62** prompt:output), 32 requests, 60 s window at bench
elapsed 40–100 s. **Window reconciled against run totals: 484 ticks in the
window at 8.07/s against a run-level 7.56/s, +6.8%** — representative, unlike
the anchor capture (§ *Where the ceiling is*). Rows per tick measured **16.0**
by four independent counts.

Full entry:
[`wins/2026-08-08-decode-shaped-reanchor-draft-attention-is-30pct.md`](experience/wins/2026-08-08-decode-shaped-reanchor-draft-attention-is-30pct.md).
Budget, not a timeline.

```
tick span 112.33 ms | GPU-busy 96.88 ms (86.2%) | period 123.9 ms
draft attention        29.57 ms  30.5%  ███████████████
FP8 GEMM               27.87 ms  28.8%  ██████████████
GDN / gated-delta      20.30 ms  21.0%  ██████████
dense GEMM (cuBLAS nvjet + splitK)
                       11.68 ms  12.1%  ██████
norms + elementwise     5.67 ms   5.9%  ███
full-attn paged         1.75 ms   1.8%  █
sampling                0.06 ms   0.1%
                                        total 96.88 ms
```

Row: 407.97 out tok/s, 659.55 total tok/s, ITL mean 31.08 ms, TTFT p50 1.42 s.
`accept_rate` 0.475, empirical chain length 3.37 tokens.

**A decode lever's share is workload-dependent, and the two captures disagree
by 17×.** Full-attention is 1.8% of a tick here at ~2.5K context and 42.5% at
32.5K on the anchor. Neither is "the" decode budget; a decode lever must be
priced at the context it will run at, and both numbers belong in this document.

**Draft attention is the largest line, and §6 had it priced out.** The register
carried `DSpark draft attention — 1.5 ms of 35 ms (4.3%) — priced out, 3
rewrites failed to transfer`. That verdict came from a shape where it was 4.3%;
here it is **30.5%**. The kernel is `nonpaged_prefill_attention_kernel`, 39,690
launches in the window — the DFlash draft has no paged KV pool, so the draft
lane takes the nonpaged branch while the trunk takes
`prefill_attention_paged_hd256_kernel`.

Every one of those launches carries grid **(32, 6, 1) = 192 blocks**, on one
stream, 7.5 µs apart: 16 slots × 5 draft layers, one launch per slot. The cost
was the launch structure, not the arithmetic — see open item #1.

**Idle inside a tick is 13.8%**, and 61.4% of that gap time sits in **53 gaps
over 1 ms**. On the anchor capture decode ticks were gap-free below 20 µs; this
shape is not. Cause unknown. Host round-trips are visible next to it: HtoD
moves 0.08 GB per tick in **0.486 ms** — latency, not bandwidth.

Achieved KV bandwidth on the paged full-attn kernel:

```
16 rows x 2484.3 tok x 65536 B = 2.605 GB / 1.590 ms = 1.638 TB/s = 47% of 3.5
```

Against 1.02 TB/s (29.2%) at 32.5K context and 9 rows on the anchor. Longer
context measured *lower* achieved bandwidth, but row count moved too (16 vs 9)
and the two effects are not separable from these two points.


### 2.1 Plain decode step, FP8

`nsys` over 59 plain-decode steps, 2026-08-01, per step.

> **STALE — do not rank decode work off this table, and note it is batch 1.**
> Two defects. First, its #2 row is a kernel that was deleted from the default
> path on 08-03: every bf16 N≤4 GEMM rides cuBLASLt now
> ([`wins/2026-08-03-t5b-lmhead-cublas.md`](experience/wins/2026-08-03-t5b-lmhead-cublas.md)),
> and the decode program that followed took the step **26.88 → 21.37 ms
> (−20.5%)** (§2.2). The 23.89 ms sum and the composition under it both predate
> that. Second, a plain single-row decode step is not the shape served —
> production runs DSpark verify at c=16, where 85% of the tick scales with rows
> (§ *Where the ceiling is*). Kept here because the **weight-read floor** it
> establishes is a device constant and does not go stale.

Budget, not a timeline — 1094 launches per step interleave.

```
                                     ms   share   launches
fp8_gemv_batch_kernel             13.80   57.8%    400  ███████████████████████████████████
gemv_handwritten_kernel (bf16)     4.30   18.0%     97  ███████████
GPU idle between launches          4.00   16.7%      —  ██████████
gated_delta_rule_decode            0.80    3.3%     48  ██
rms_norm / add / silu              0.79    3.3%   ~250  ██
flash attention                    0.20    0.8%     16  █
                                                        total 23.89 ms
```

Shares are of the 23.89 ms sum. The originally published shares (66 / 21 / 16 /
4 / 4 / 1) were taken against the ~21 ms measured step, and the idle row against
a third base; the ms column is the measurement and the sum is the denominator
used here.

Weight-read floor at 31.2 GB / 3.5 TB/s = **8.9 ms** — the one durable number
in this table. GEMV measured 18.1 ms against it, **~49% of achievable
bandwidth**, reproducing the 51% found in July; but 4.30 ms of that 18.1 is the
since-deleted `gemv_handwritten_kernel`, so treat 49% as a batch-1 figure from a
superseded configuration, not as the current roofline gap.

### 2.2 Module ledger versus SGLang, W8A16

2026-08-03, both stacks `nsys`-decomposed, columns summing to measured ITL.
ARLE 25.08 ms/step versus SGLang 17.07 ms/step:

| module | ARLE | | SGLang | | Δ | attributed to |
|---|---:|---|---:|---|---:|---|
| marlin | 13.20 (357) | `██████████████████████████` | 12.31 (270) | `████████████████████████` | 0.89 | qkv fusion, fixed-grid prologue × launches |
| idle | 5.66 | `███████████` | 2.08 | `████` | **3.58** | whole-step CUDA graph |
| bf16 gemv | 3.17 (52) | `██████` | 1.11 (nvjet) | `██` | 2.06 | `gemv_handwritten_kernel` on fused `[96,5120]` in_proj_ba: ~52 µs at ~19 GB/s, one-block-per-row grid cannot fill 78 SMs; cuBLAS splitK does the shape in ~8 µs |
| GDN chain | 1.21 | `██` | 0.53 | `█` | 0.68 | kernel quality (fla-style) |
| FA3 chain | 0.93 | `██` | 0.45 | `█` | 0.48 | decode config |
| norms | ~parity | | ~parity | | ~0 | — |
| **total** | **25.08** | | **17.07** | | **8.01** | |

The kernel was exonerated by construction: SGLang decodes bs=1 inside a
whole-step captured CUDA graph while ARLE launched ~1094 kernels eagerly.

**Program outcome, all shipped and default-on the same day: 26.88 → 21.37 ms
(−20.5%).** `gemv → cuBLASLt` −1.28, qkv/qkvz fusion −0.59, whole-step decode
graph under paged KV −1.84 (per-slot persistent `PageMeta` refreshed outside
the graph, FA3 `seqlen_k` pinned to capacity, TileLang fallback refuses
capture). `--qwen35-decode-graph` is default-on for serve with an MMLU 84/100
license.

Remaining against SGLang's 17.07: ~4.3 ms = host tail (8 refresh H2Ds +
sampling D2H/sync + scheduler) + GDN kernel 0.7 + FA3 decode config 0.5 +
marlin prologue residue.

### 2.3 Aggregate decode throughput is nearly flat in batch size

From the anchor row, per-request decode tok/s and the aggregate `B / TPOT`:

| c | TPOT ms | per-request tok/s | aggregate decode tok/s |
|---:|---:|---:|---:|
| 1 | 8.46 | 118.1 | 118.2 |
| 2 | 18.70 | 53.5 | 106.9 |
| 4 | 33.77 | 29.6 | 118.4 |
| 8 | 62.49 | 16.0 | 128.0 |
| 16 | 110.52 | 9.0 | **144.8** |

**Sixteen times the batch buys 1.23× the decode throughput.** An earlier
version of this section attributed that to a batch-independent intercept. That
attribution is withdrawn: on the current binary the tick is **85% per-row**
(verify 5.69 ms/row), so the flat curve is the per-row term, not a fixed cost.
Two cautions on the number itself — this column is `B / TPOT`, a decode-only
aggregate, while §2.4 and §5.1 measure end-to-end `out tok/s` and scale 3.3×
over the same range; and c=16 is offered concurrency, against a measured mean of
11.0 rows per tick (§3).
Aggregate *total* throughput still scales (10453 → 33780 tok/s) because prefill
tokens dominate the token count on the anchor workload.

### 2.4 Sampling costs 30–40% of decode throughput at c ≥ 8

Counterbalanced greedy/sampled sweep, 2026-08-07, `7b8a66603`, long-agent 32K,
128 requests per point, order greedy, sampled, sampled, greedy.

| c | greedy (temp 0) | sampled (temp 0.7) | Δ |
|---:|---:|---:|---:|
| 1 | 34.8 | 33.55 | **−3.6%** |
| 8 | 108.65 | 65.4 | **−39.8%** |
| 16 | 113.8 | 78.6 | **−30.9%** |

out tok/s, each cell the mean of that arm's two sweeps. Within-arm spread is
8.4% at c=16 and 6.4% at c=8, so the effect clears its own noise by 4–6×.
Greedy completed 128/128 everywhere; sampled completed 120/128 at every point,
cause unknown.

**The concurrency shape is the evidence.** At c=1 sampling costs 3.6%; at c=8 it
costs 39.8%. A lower acceptance rate under temperature would cost roughly the
same fraction at every concurrency. A per-row host loop costs nothing at c=1 and
grows with row count — which is the shape measured.

The candidate mechanism is the batched-draft gate at
`infer-cuda/src/executor/qwen35.rs:1984`:

```rust
if idx.len() >= 2 && decode_rows.iter().all(|r| r.params.is_greedy())
```

`idx` is the seeded rows, but the greedy test sweeps **all** rows, so a single
sampled request drops every greedy row to per-row drafting. In this sweep every
row is sampled, so the batched path never fires.

**Not yet isolated:** the sampled arm has no `accept_rate` figure, so the gate's
share is unattributed. Note the rival hypothesis is already dead — "accept
halves at concurrency" was withdrawn in `baselines.md:137`; `accept` tracks
prefix-cache state, not `c` (0.532 vs 0.313 at matched c=16). This stays a
top-ranked item because production serving is sampled while every optimization
on this row was measured at temp 0.

---

## 3. Speculative decode — the DSpark tick

`ARLE_DSPARK_PHASE=1`, 2026-08-07, `7b8a66603`, c=16, short prompts, 293 ticks,
mean 11.0 rows/tick.

The tick is sequential, so position and length are both to scale:

```
                 ms   share
draft         12.75   19.9%  ████████████
snapshot       1.21    1.9%              █
verify        42.64   66.6%               ████████████████████████████████████████
commit         2.40    3.7%                                                       ██
rollback       5.03    7.9%                                                         █████
                                          total 64.03 ms
```

Shares are of the 64.03 ms sum. The phase table published earlier used 59.00 ms
(the four `[dspark-phase]` fields, excluding the separately-logged rollback) as
its base, which is why `draft` reads 21.6% there and 19.9% here. Rollback is
real tick time, so the sum is the honest denominator.


`commit` splits into tap 0.42, accept 0.02, cap 0.01, trunc 0.01, ext 1.94.
`rollback` splits into restore 0.85, replay 4.18.

**The phase timer synchronizes at each lap, so the total is inflated and only
the split is meaningful.** The same run measured 149.1 out tok/s against
236–243 unprofiled.

Two structural facts:

- **draft + verify = 86.5% of the tick.** Orchestration — snapshot, commit,
  rollback — is 13.5%. Per-row host bookkeeping inside commit is negligible
  (accept 0.02 ms, cap and trunc 0.01 ms each).
- **The per-row term is the wall, not the intercept.** An earlier fit on this
  short-prompt capture read verify as `22 ms intercept + 2.48 ms/row`. On the
  current binary at c=16 / 33K the tick is a pure per-row fit — **verify
  5.69 ms/row, draft 3.04 ms/row, 85% of the tick scaling with rows**
  ([`baselines.md:162`](baselines.md), measurement in
  [`errors/2026-08-06-…-gemm-is-not-the-top-lever.md`](experience/errors/2026-08-06-decode-lever-board-rebuilt-gemm-is-not-the-top-lever.md)).
  Use the per-row fit. An intercept, where one exists, is the shared weight read
  — the term batching **does** amortize; the slope is what it cannot.

Consequence for kernel work: price a kernel against the per-row term at the
batch you serve, not against a batch-1 step. DSpark draft attention was priced
at 1.5 ms of a 35 ms step (4.3%), which caps a −33% microbench win at −1.4% end
to end. On a decode-shaped c=16 workload the same kernel is 30.5% of a tick
(§2.0), and the cost there is the per-slot launch, not the inner loop — the
same shape error that made the three rewrites miss.

---

## 4. Costs outside the forward

Measured on the decode-heavy short-prompt shape, 2026-08-07.

### 4.1 GPU idle is not launch gaps

One 19.92 s decode window, GPU busy 10.07 s (50.5%), idle 9.86 s. Binning every
gap on the unified kernel+memcpy timeline:

| gap size | n | total | share of idle |
|---|---:|---:|---:|
| 0–5 µs | 410485 | 0.59 s | 6.0% |
| 5–20 µs | 18045 | 0.14 s | 1.5% |
| 20–50 µs | 643 | 0.02 s | 0.2% |
| 50–200 µs | 1328 | 0.12 s | 1.2% |
| **>1 ms** | **79** | **8.98 s** | **91.1%** |

**All 430k sub-millisecond gaps together are 0.87 s, 4.4% of the window** —
the four bins above, not only the 0–5 µs one. That is the entire budget a CUDA
graph on this path can recover. (No bin covers 200 µs–1 ms; the column sums to
9.85 s against 9.86 s stated idle, so that range is empty in this capture.) 91% of the idle is 79
stalls averaging 114 ms, of which **7.45 s sits in no CUDA API call at all**.

### 4.2 The prefix sidecar

`Qwen35RecurrentSnapshot` writes the whole recurrent state at every stride
boundary of every prefill so a later conversation can restore the hybrid
prefix. The payload is fixed at **146.8 MiB** by the model's 48 linear layers,
independent of how much prefix is cached.

| | per snapshot | per 512 s bench |
|---|---:|---:|
| count | — | 578 |
| payload | 146.8 MiB | **83 GB** |
| serialize, per element (`d626a1b03^`) | 84.45 ms | 48.25 s = **9.4% of wall** |
| serialize, bulk copy (`d626a1b03`) | 76.40 ms | 43.65 s |

Bulk copy is **−9.5% on the operation and 0.9% of wall** — an end-to-end null,
kept because it is strictly less work
([bench](experience/wins/2026-08-07-prefix-sidecar-serialize-bulk-copy.md)).
146.8 MiB in 76 ms is 1.9 GB/s, so the residual is allocating and
first-touching fresh heap; making this materially cheaper means not making the
copy.

**Open:** the sidecar's restore hit rate is unmeasured, so whether 83 GB per
bench is earned is unknown. This is the largest unpriced item in the chain.

### 4.3 Whole-slot park

`admit_via_oversubscription` parks the longest-running decode into the KV tier
whenever a waiter exists. Both park routes are **unreachable in a default
serve** — `--kv-oversubscription` defaults off, and the other route requires
`kv_tier_capacity() == 0` while the L2 host tier is on. The same per-element
serialization was fixed there in `a546ba80a` and remains **unmeasured** for
that reason. Park and promote now log elapsed ms and a running count.

---

### 4.4 Memory ledger

From a serve start, 16 slots, `mem_fraction_static 0.9`, FP8 weights
(2026-08-07 serve log):

| | |
|---|---|
| total VRAM | 97508 MB |
| free after weights | 64731 MB |
| recurrent reservation | **3127 MB** = 16 slots × 195 MB |
| free after recurrent | 61604 MB |
| full-attn KV pool | **51853 pages** @ page_size 16 = 829648 tokens, 54.4 GB |
| per-slot budget | 195 MB = gdr 144 + conv 2 + draft 48 (K+V is paged, 0) |
| L2 host DRAM tier | 862 GB budget (`dram_fraction 0.5`), features: `prefix` |
| L3 (SSD) | off by default |

Two consequences the chain depends on:

- **The recurrent state, not the KV cache, is the per-slot cost.** 195 MB per
  slot is fixed by the 48 linear layers and is independent of context length,
  while full-attn KV is paged at 65536 B/token across only 16 layers.
- **The device pool is not the binding resource at these workloads.** 829648
  tokens against 16 concurrent rows means a 16 × 1750-token workload occupies
  3.4% of the pool, which is why no KV-pressure preempt fires and why §4.3's
  park routes stay unreachable.

---

## 5. Anchor numbers

### 5.0 Current row

Long-agent 32K × 8 turns, DSpark, `70760bc09`, 2026-08-07 — the row
[`baselines.md`](baselines.md) tracks:

| c | TTFT cold | TTFT warm | TPOT | total tok/s |
|---|---:|---:|---:|---:|
| 1 | 10.82 s | 0.84 s | 8.46 ms | 10453.0 |
| 8 | 1.60 s | 0.79 s | 62.49 ms | 31334.5 |
| 16 | 2.90 s | 1.22 s | **110.52 ms** | **33780.3** |

### 5.1 Day delta — what the 08-07 decode work moved

Counterbalanced A/D/D/A, `010af0ede` (morning) against `7b8a66603` (evening),
same anchor workload, 128/128 complete at every point in all four sweeps. Each
cell is the mean of that arm's two sweeps.

| c | A out tok/s | D out tok/s | Δ | Δ total tok/s |
|---:|---:|---:|---:|---:|
| 1 | 34.15 | 35.35 | +3.5% | −0.2% |
| 2 | 75.20 | 78.70 | +4.7% | +2.9% |
| 4 | 83.95 | 96.25 | **+14.7%** | +9.6% |
| 8 | 91.70 | 111.40 | **+21.5%** | +10.0% |
| 16 | 104.80 | 118.40 | **+13.0%** | **+22.3%** |

The gain appears at c ≥ 4 and is ~flat at c = 1, which is the signature of the
two mechanisms that were fixed: both were per-row host loops whose cost scales
with row count and vanishes at a single row.

### 5.2 Versus SGLang

W8A16 against SGLang on identical weights and kernel, 2026-08-06:

| arm | TTFT p50 | prefill tok/s | ITL p50 | ITL p99 | e2e p50 |
|---|---:|---:|---:|---:|---:|
| ARLE | 23.01 s | 1434 | **16.70** | 20.50 | 27.4 s |
| SGLang | **21.03 s** | **1568** | 17.16 | **19.19** | **25.44 s** |

---

## 6. Lever register

Every lever that has been priced, with the measurement that settled it, and the
**batch it was measured at**. That last column is the one that has misled this
document: a lever measured at batch 1 carries no information about the shape
production serves.

Ranking rule: **effect size × share of the step you actually run.** An earlier
version of this table ranked by a factor category instead. That rule is refuted
here — FlashQLA is a 16.4× kernel win (§1.2) that yielded −26% end to end
because it sat on a 23% module, while DSpark's ~2.5× sat on the whole step.
Ceilings belong to levers, not to categories.

**Register, ordered by measured size**

| lever | phase | measured at | size | status |
|---|---|---|---|---|
| **FP8 GEMM, prefill shapes** | prefill | **c=16 anchor** | **57.7% of ALL kernel time** | **CLOSED (§1.3)** — `gate_up` 93.2% of FP8 peak, `down` 87.5%. At the floor; only less math helps |
| **DSpark draft attention** | decode | **c=16, 2.5K ctx** | **30.5% of a decode tick** | **FIXED** `3a8f99b1f` — per-slot 192-block launch; −69% pinned, ITL −10.4% decode-shaped, **null on the anchor** |
| FP8 GEMM on the verify shape | decode | c=16, 2.5K ctx | 28.8% of a decode tick | open — but the prefill shapes are at 90% of peak (§1.3), so expect little |
| GDN / gated-delta at c=16 | decode | c=16, 2.5K ctx | 21.0% of a decode tick | open — a same-binary A/B nulled it at 33K, untested here |
| >1 ms gaps inside decode ticks | decode | c=16, 2.5K ctx | 13.8% of tick span, 53 gaps | open, cause unknown |
| full-attn KV bandwidth | decode | c=16 | **47% of achievable at 2.5K, 29.2% at 32.5K** | open, but only 1.8% of a tick at short context |
| batched-draft gate under sampling | decode | c=1…16, 32K | −30 to −40% decode tok/s at c ≥ 8 | **open, #2** — `executor/qwen35.rs:1984` tests all rows greedy; one-line narrowing to `idx` |
| prefill GPU idle | prefill | c=1, 33K | 3.97 vs SGLang 0.19 s | **open, #3** — largest single gap, own SLO (TTFT) |
| sidecar write policy | prefill | c=1…16 | 83 GB / 9.4% of wall | open — hit rate unmeasured |
| host tail (refresh H2D, sampling sync) | decode | **batch 1** | part of ~4.3 ms residual | open, needs re-pricing at c=16 |
| GDN decode kernel | decode | **batch 1** | 1.21 vs 0.53 ms | open at batch 1; the GDN lane is a measured null at c=16 |

**Shipped**

| lever | phase | measured at | result |
|---|---|---|---|
| DSpark speculation | decode | c=1…16 matched on/off | **2.9× at c=1 decaying to 1.1× at c=16** (`baselines.md:144`) |
| FlashQLA chunked GDR | prefill | c=1, 33K | linear attention **7.231 → 0.441 s (16.4×)**, 33K cold −26% |
| whole-step CUDA graph | decode | batch 1 | −1.84 ms, default-on |
| `gemv_handwritten` → cuBLASLt | decode | batch 1 | −1.28 ms |
| qkv/qkvz fusion | decode | batch 1 | −0.59 ms |
| batched DSpark verify core | decode | c=8 | TPOT −12.7% |
| batched rollback replay | decode | c=16 | TPOT −11.4% |
| sidecar bulk serialize | prefill | c=1…16 | −9.5% on the op, end-to-end null |

**Priced out**

| lever | phase | measured at | why |
|---|---|---|---|
| quantized GEMM kernel | prefill | c=1, 33K | identical to SGLang ±15 ms |
| DeepGEMM FP8 | prefill | c=1, 33K | 64–67% of peak |
| CUDA graph on the spec path | decode | c=16 window | all sub-ms gaps 0.87 s, 4.4% of window |
| prefill–prefill fusion | prefill | c=1 | ~3% (15 redundant weight reads) |
| `--chunked-prefill-size` | prefill | c=1, 33K | ±0.07 s TTFT |
| `--max-num-batched-tokens` | prefill | c=1…16 | budget never binds; 16384 stays |
| pinned readback staging | decode | c=1…16 | wash on both phases, kept |

**What the batch column exposes.** Nine of the levers above were measured at
batch 1, including every open kernel item. The decode work this document ranks
runs at c=16, where the tick is 85% per-row and the per-row term is 9.2× off
its roofline. Re-pricing the batch-1 rows at c=16 is a prerequisite for
trusting any of their sizes.

---

## 7. Reproducing

Bench parameters, gates, and the A/B contract live in
[`bench-and-trace-spec.md`](bench-and-trace-spec.md). Two notes specific to this
chain:

- **`ARLE_QWEN35_PROFILE` parent ranges are inflated.** Each leaf ends in
  `stop.synchronize()` and a parent absorbs every child's sync bubble. Only
  leaves are real. Count forwards as `input_norm` instances / 64.
- **`nsys` is not required to read its own database.** Every gap and API figure
  in §4.1 came from `sqlite3` over the `.sqlite` an earlier capture wrote —
  `CUPTI_ACTIVITY_KIND_KERNEL`, `_MEMCPY`, `_RUNTIME` are plain tables.

---

## Next work and supporting evidence

Two rules the 08-08 captures produced. **Price a lever on a workload where the
phase it touches is a large share of GPU time** — full-attention decode is 47%
of roofline and 1.8% of a decode tick at short context; both are true and only
the second ranks it. And **reconcile a capture window against run totals before
quoting a share** — the anchor capture undersampled decode 2–6× by landing in a
queueing ramp.

Execute in this order:

| priority | measurement | decision it unlocks |
|---:|---|---|
| 1 | full-run anchor phase counters or timeline: issued prefill segments, decode ticks, rows, accepted tokens, GPU busy, and queue depth | run-level shares and a closed prediction for the next treatment |
| 2 | prefix-sidecar write count and restore-hit count on the same run | retain, reduce, or delete the 83 GB write path |
| 3 | `ncu` on `fq_fwd` at the traced 2048-token shape | test `block_DV=32` only if wave occupancy or dependency stalls bind |
| 4 | post-fix decode-shaped `nsys`, then sampled-row A/B with `accept_rate` and seeded-row counts | locate the missing 43% of the draft-attention prediction; isolate the sampling gate |
| 5 | one `ncu` profile per tail row, descending by observed time: `conv1d`, `silu_mul`, `split2`, then norms/prep | choose a row-specific vectorization or fusion; no batch port of `pack_quantize` |

The first two items rank wall-clock work. Items 3–5 rank kernel work after the
run-level denominator is known.

**Decode** — priced on §2.0, the decode-shaped c=16 capture.

1. **DSpark draft attention, 30.5% of a decode tick — fixed at `3a8f99b1f`.
   ITL mean 31.05 → 27.81 ms, −10.4% on the decode-shaped workload, and a null
   on the anchor** (TPOT +0.8%, inside the trial spread, 3 trials per arm).
   Correctness clean: 11/11 needle rungs, MMLU 0/50 disagreements. It was the
   largest single line, and §6 had it **priced out at 4.3%** from a shape where
   it was small — but the anchor result is the reminder that a 30.5% share on
   one workload is 2–6% on another, and the fix is worth what the share is.

   The cost was never in the kernel's arithmetic. All 39,690 launches in the
   window carry grid **(32, 6, 1) = 192 blocks**, serialized on one stream 7.5
   µs apart — 16 slots × 5 draft layers per tick, one launch each. The batched
   draft (`dspark_draft_blocks`) batched the GEMMs and left the ring kernels
   per-slot, which its own doc comment said. 192 blocks is ~2.5 per SM on an
   H20; the two 08-01 rewrites were tuned by `ncu` at **3072** blocks, where
   the kernel really is ALU-bound, and that is why neither transferred.

   Folding the slot axis into `blockIdx.z` gives grid (32, 6, 16) = 3072
   blocks. Pinned-shape A/B, output bit-identical in every arm:

   | kv_len | per-slot | batched | Δ |
   |---:|---:|---:|---:|
   | 512 | 3.545 ms | 1.023 | −71.1% |
   | 1024 | 7.203 | 2.057 | −71.4% |
   | 1376 | 9.013 | 2.766 | **−69.3%** |
   | 2048 | 13.283 | 4.197 | −68.4% |

   The win is flat across the window, so it is structural. The harness at
   `kv_len` 1376 costs 563 µs per 192-block launch against the serve's 558 µs
   mode, so it sits on the measured operating point.

   **Only 57% of it transferred even on the decode-shaped workload, and that is
   the open part.** The decomposition projects −18.2% on tick span (29.57 →
   9.07 ms saves 20.5 ms of a 112.33 ms span); the serve measured −10.4% on ITL
   mean. Cause unknown. Settle it with an `nsys` capture on the new binary at
   the same shape — confirm the draft line actually fell to ~9 ms, and locate
   the residual. Item #4 below, the 53 gaps over 1 ms, is the first place to
   look.
2. **FP8 GEMM on the verify shape, 28.8%.** Priced out at 64–67% of peak on
   *prefill* shapes (§1.1). The verify shape is different and has never been
   decomposed.
3. **GDN / gated-delta, 21.0%.** A same-binary A/B nulled this lane at 33K
   context; that A/B has not been repeated at short context, where its share is
   larger.
4. **Gaps over 1 ms inside decode ticks** — 13.8% of tick span, 61.4% of it in
   53 gaps. Decode ticks on the anchor capture had nothing above 20 µs. Cause
   unknown. Next to it, HtoD moves 0.08 GB per tick in 0.486 ms, which is
   latency.
5. **The trunk verify attention is per-slot too, and it is small.**
   `prefill_attention_paged_hd256_kernel` runs grid **(4, 6, 1) = 24 blocks**,
   126,948 launches — the same one-launch-per-slot shape as #1. It costs 1.6%
   of the decode window here and 5.4 ms of the anchor window, so it does not
   earn the same fix today. Recorded because the mechanism is identical and its
   share grows with context.

**Prefill** — priced on the anchor, which is what it models.

5. **The anchor's FP8 GEMM is decomposed and it is CLOSED — ~90% of FP8 peak
   (§1.3).** It is 57.7% of all kernel time, and `gate_up` runs at **93.2%** of
   the 296 TFLOPS peak, `down_proj` at 87.5%, attention `out_proj` at 86.4%.
   MLP alone is ~40% of all GPU kernel time and sits on the hardware floor. The
   "64–67% — leave alone" in §1.1 was right about the verdict and wrong about
   the number, having been measured at 33K cold single-request.

   **The consequence re-ranks the project: on the anchor, the dominant cost
   cannot be made faster, only smaller.** Levers are prefix-cache hit rate,
   sparsity, and effective context length — not kernels.

6. **Prefill FA3 is CLOSED — 140.8 TFLOPS, 95.1% of the bf16 peak (2026-08-09).**
   The persistent (78, 1, 1) grid hides sequence lengths, so the rate came from
   the *slope* instead: 732 µs per 2048×2048 block of attention work, constant to
   ±0.8% over nine rungs (§1.3). Mean effective KV depth 18.0K. 227 ms of
   headroom in 4646 ms. **With this and item 5, prefill arithmetic has 1797 ms of
   headroom total — 6.1% of selected-window wall — and no kernel lever remains
   in it.**

7. **The data-prep tail — `pack_quantize` DONE, eight kernels remain.**
   `pack_quantize` shipped at `5cfe8494f`: 2216 → 441 ms, **5.12× in situ**,
   bit-identical, wall −2.98%. The remaining rows total **2248 ms, 7.9% of the
   selected window's kernel time**. Their implementations differ: `silu_mul`
   and `add_native` already use `uint2`, `split2` and `split_qkv` use `uint4`,
   `conv1d` is a causal compute loop, and norms/prep contain reductions. Their
   run-level shares, binding resources, and reclaimable fractions are open.

8. **`fq_fwd` (FlashQLA) is now the largest non-GEMM term — 1708 ms, 5.97%, at
   13.5% of the bf16 roofline.** Floor derived 2026-08-09 (§ the floor layer):
   230 ms ideal, 371 ms after wave quantization, **1478 ms nominal headroom**.
   **`T_stall`'s only large candidate, and that bucket has never been measured.**
   Two derived limits, neither confirmed: 96 CTAs on 78 SMs caps utilization at
   62% (`block_DV = 32` would give 192 CTAs and 82% — a concrete, falsifiable
   A/B), and a 32-chunk serial recurrence of six dependent 64×128×64 GEMMs runs
   at 18% per chunk step. **`ncu` warp-stall reasons decide how much of the 1478
   ms is reachable; until then it is not comparable to item 7's 2248 ms.**

9. **Sampling costs 30–40% of decode throughput at c ≥ 8** (§2.4) — measured on
   the anchor, so its size needs re-measuring on §2.0's shape. The gate at
   `executor/qwen35.rs:1984` tests *all* rows greedy. Do **not** re-open
   "acceptance collapses with concurrency": withdrawn (`baselines.md:137`), and
   both 08-08 captures measured `accept_rate` ≈ 0.475.
10. **Sidecar write policy** — restore hit rate still unmeasured.

## Measurement debt

Facts this chain rests on that have not been measured:

| item | why it matters |
|---|---|
| **`fq_fwd` warp-stall split** | 1478 ms nominal headroom at 13.5% of roofline; `ncu` decides how much is reachable, and it is the only large `T_stall` candidate in the document |
| **full-run phase accounting** | the selected window over-represents `pack_quantize` by 2.02x; the run-level shares of every other term and the critical-path split remain unknown |
| prefix sidecar restore hit rate | decides whether 9.4% of wall is earned |
| acceptance rate under temperature | the sampled arm has no matching `accept_rate`; its 30–40% throughput penalty is still unattributed |
| whole-slot park cost | `a546ba80a` shipped unmeasured; both routes default-off |
| tokenize / detokenize share | folded into "GPU idle" in every prefill capture |
| TP > 1 | every number here is single-GPU |
| the 8/128 incomplete requests under sampling | uniform across arms, cause unknown |
