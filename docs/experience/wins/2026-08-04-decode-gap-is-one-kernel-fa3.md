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

**313 µs per full-attn layer against SGLang's 18 µs — 17×.** It is 25% of the
decode step on its own, and it is the whole gap: subtract it and ARLE is
*ahead* on every other row except norm.

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

**Leading hypothesis, NOT verified:** `PageMeta::persistent_decode` pins
`seqlen_k_capture` to the pool *capacity* rather than the live sequence
length, and FA3 sizes its scheduling from that scalar. On 2026-08-04 I
verified the pinned scalar does not reach the causal mask (`seqlen.h:52`
resolves to `seqused_k[bidb]` under varlen) and concluded it was safe — but I
only checked **correctness**, never **cost**, even though the code comment I
wrote says "pinning the FA3 scheduling ceiling". The KV pool in this run is
57133 pages (914K tokens) against a 33K live sequence.

Next probe: dump `seqlen_k` and the grid FA3 actually launches, at both
capacities. If confirmed, the fix is to refresh `seqlen_k_capture` to a
bucketed length rather than the pool ceiling — bucketed so it stays
capture-stable.

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
