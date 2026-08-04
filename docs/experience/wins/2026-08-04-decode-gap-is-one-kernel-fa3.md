# The whole decode gap vs SGLang is one kernel: FA3 decode attention — 2026-08-04

> Status: **Measured.** Champion W8A16, c=1, 33K context, decode-only window,
> nsys on the live serve. There is **no idle problem** — ARLE's GPU occupancy
> is 0.95 against SGLang's 0.90. The entire gap is kernel time, and 4.71 ms of
> the 4.75 ms of it is a single kernel.

## The measurement that dissolved two days of wrong leads

| per decode step | ARLE champion | SGLang | Δ |
|---|---:|---:|---:|
| wall | 20.120 ms | 16.115 | +4.01 |
| **GPU busy (Σ kernel)** | **19.184** | 14.437 | **+4.75** |
| GPU idle | 0.94 | 1.68 | **−0.74 (ARLE better)** |
| launches | 996 | 928 | +68 |
| **occupancy** | **0.95** | 0.90 | |

Per kernel:

| kernel | ARLE | SGLang | Δ |
|---|---:|---:|---:|
| marlin GEMM ×256 | 11.614 | 11.635 | parity (control) |
| **FlashAttn ×16** | **5.001** | **0.295** | **+4.71** |
| lm_head ×1 | 0.664 | 0.666 | parity |
| GDN decode ×48 | 0.226 | 0.277 | −0.05 (ARLE faster) |
| linear in_proj ×48 | 0.222 | 0.305 | −0.08 (ARLE faster) |
| norm + residual | 0.606 (×192) | 0.504 (×209) | +0.10 |

**313 µs per full-attn layer.** It is 25% of the decode step on its own.

**Do not read the SGLang column here as 17×.** Config gives `head_dim 256`,
`num_key_value_heads 4`, so KV is `4 × 256 × 2(K,V) × 2B = 4096 B` per token
per layer. At 33K context over 16 full-attn layers a decode step must read
**2.16 GB**, and H20's measured achievable read is 3.5 TB/s — a **618 µs**
floor. SGLang's 295 µs would be 7.3 TB/s, **twice the hardware peak**. That
number cannot have been taken at 33K context, so the two windows are not at
the same sequence length and the cross-stack ratio is not defensible.

The marlin row matching to 0.2% does **not** rescue the comparison: marlin
reads weights, so it is context-independent by construction. It proves same
model and same kernel; it says nothing about sequence length. Treating it as a
whole-ledger control was over-reading it.

**The defensible statement is absolute, not relative:** ARLE's FA3 decode
attention runs at **432 GB/s — 12% of achievable — and is 8.1× off its own
roofline.** The target is 618 µs, set by physics, not by another stack.

## Why this took so long to see

Every previous conclusion in this investigation came from a number I had
**projected rather than measured**. `Σ kernel ≈ 14.7 ms` was the T4 trace's
15.886 minus the *expected* gains of T5b and T6. From it I derived a 4.3 ms
"mysterious intra-GPU idle", and from that a 4 µs/launch dispatch cost, and
from that a fusion tranche — built, measured at 0.00 ms, reverted.

The real Σ kernel is **19.184**. The idle never existed. One nsys run on the
champion — the thing I kept deferring because a projection "was close enough" —
refuted the whole chain.

## Not yet root-caused

ARLE's own T4 trace measured this kernel at **0.516 ms/step**. It is now
**5.001**. Nothing between T4 and the champion (T5b lm_head, T6 GDN,
resident-page counter) touches attention, so the change is more likely a
configuration difference between the two captures than a code regression.

**That hypothesis was tested and REFUTED.** It proposed that
`PageMeta::persistent_decode` pins `seqlen_k_capture` to the pool *capacity*
rather than the live length, and that FA3 sizes scheduling from that scalar.
Running with `--qwen35-decode-graph false` takes the `PageMeta::for_slot` path,
where `seqlen_k_capture` is `None` and FA3 sees the real length. If the pin
cost 4.5 ms, that arm would be far faster. Measured `submit`: **20.570 /
20.631 ms against the graph-on champion's 19.030** — graph-off is 1.55 ms
*slower*. The sign is wrong for the hypothesis; the pin is not the cost.

(That arm also reconciles the two launch experiments: inside a captured graph
node count is ~free — 192 fused away bought 0.00 ms — while eager launching
costs ~1.6 µs each, 1.55 ms over ~1000 launches. Both are consistent with T4's
1.84 ms graph win.)

**Superseded hypothesis text follows for the record:** On 2026-08-04 I
verified the pinned scalar does not reach the causal mask (`seqlen.h:52`
resolves to `seqused_k[bidb]` under varlen) and concluded it was safe — but I
only checked **correctness**, never **cost**, even though the code comment I
wrote says "pinning the FA3 scheduling ceiling". The KV pool in this run is
57133 pages (914K tokens) against a 33K live sequence.

## The open question, restated against physics

`arle_fa3_fwd_hd256_bf16_cuda` moves 2.16 GB per decode step at 432 GB/s.
Where the other 88% goes is unmeasured. Structural facts that bear on it:

- **`head_dim = 256`** — twice the common 128, and FA3's hd256 path is a
  separate specialization with its own shim. Per-token KV bytes are doubled by
  the model architecture; that part is not recoverable, but the 618 µs floor
  already accounts for it.
- **`num_key_value_heads = 4`** against 24 query heads — GQA 6:1. A decode
  step is 24 query rows against 4 KV head streams, which is a shape where
  head-parallel schedules can leave most of the machine idle.
- The paged pool is HND `[page, h_kv, page_size, d]`; a 33K sequence is ~2060
  pages of 16 tokens. Whether the kernel reads a page contiguously or strides
  across heads decides whether this is a bandwidth or a latency problem.

Next probe is **`ncu` on this one kernel** — DRAM throughput, achieved
occupancy, and the split/grid it actually launches — not another end-to-end
run. It is the first thing in this investigation with a target set by physics
rather than by a projection or a competitor.

## Rule

**Verifying a shortcut is correct is not verifying it is free.** The pinned
`seqlen_k` passed a rigorous correctness argument and was never asked what it
cost, in a code path whose own comment says it sets a *scheduling* ceiling.
When a value is pinned for capture-stability, measure the kernel at both the
pinned and the natural value before shipping it.

And: **re-measure the composition after every tranche.** A ledger is only
valid for the commit it was taken on; four tranches of projection turned a
+1.45 ms kernel gap into a −0.74 ms idle advantage and I did not notice for a
day.
