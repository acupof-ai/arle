# The residual gap vs SGLang is GPU idle, not GPU work — 2026-08-03

> Status: **Measurement, no code change.** Two-sided decode-only ledger,
> ARLE @ T4 vs SGLang 0.5.13, same H20 GPU 6 / same gptq_marlin kernel / same
> int8 weights. Redirects the post-#196 work: at the champion an estimated
> **~0.3 ms of the remaining ~1.9 ms is kernel time; ~1.6 ms is host time
> between steps.**

## Method

Both traces are whole-run captures that include prefill, so a naive
per-kernel sum is contaminated. The ledger instead picks a window from the
last N launches of a kernel that **only** runs during decode
(`conv1d_state_update_kernel` for ARLE, `_causal_conv1d_update_kernel` for
SGLang — 48 per step each, one per linear-attn layer) and attributes every
kernel inside that window.

Two validity checks, both passed:

- **Every per-step count is an exact integer** — 256 marlin, 128 norm, 96
  split, 64 act, 48 linear-layer kernels, 16 full-attn-layer kernels, 1
  lm_head. A window straddling a prefill would not produce integers.
- **Window-size insensitivity** — 200-step and 500-step windows agree to
  0.05% on wall/step and to 0.3% on busy/step.

Step counts also cross-check against a second decode-only kernel on each
side (`store_kvcache` / `decode_prep_paged_hd256_kernel`, 16 per step).

## The ledger — per decode step, c=1, ~33K context

| | ARLE @ T4 | SGLang | Δ |
|---|---:|---:|---:|
| wall | 20.58 ms | 16.12 ms | +4.46 |
| **GPU busy** | 15.89 | 14.44 | **+1.45** |
| **GPU idle** | **4.69** | **1.68** | **+3.01** |
| launches | 1059 | 928 | +131 |
| occupancy | 0.77 | 0.90 | |

**Two-thirds of the gap was already idle, not work.** And the busy side is
dominated by a term that is identical by construction:

| busy term | ARLE | SGLang | Δ |
|---|---:|---:|---:|
| marlin GEMM ×256 | 11.612 | 11.635 | **−0.02 (control)** |
| lm_head ×1 | 1.350 | 0.666 | +0.684 |
| GDN decode ×48 | 0.783 | 0.277 | +0.506 |
| FlashAttn ×16 | 0.516 | 0.295 | +0.221 |
| norm + residual | 0.660 (×304) | 0.504 (×209) | +0.156 |
| paged prep / gate / varlen ×48 | 0.187 | 0.077 | +0.110 |
| conv1d | 0.199 (×96) | 0.125 (×48) | +0.074 |
| split2 / split_qkv | 0.134 (×112) | 0.068 (×48) | +0.066 |
| linear in_proj ×48 | 0.222 | 0.305 | **−0.083 (ARLE faster)** |

The marlin row is the control: same kernel, same weights, 0.2% apart. It
confirms the harness rather than teaching anything, which is exactly its job.

## Projection to the champion — and why it redirects the work

Landed since this trace: T5b (lm_head → cuBLASLt, removes the 0.684),
T6 (GDN decode kernel, removes most of the 0.506), and the resident-page O(1)
counter (−1.21 ms, entirely **host** time). Projecting those onto the ledger:

| | ARLE champion (projected) | SGLang | Δ |
|---|---:|---:|---:|
| GPU busy | ~14.7 | 14.44 | **~+0.3** |
| GPU idle | ~3.5 | 1.68 | **~+1.8** |

Projected total gap ~2.1 ms against a measured 1.91 ms (18.98 vs 17.07) —
consistent within nsys overhead. **These champion rows are projected, not
measured**; a fresh capture should confirm them before anything is built on
them.

The redirect: after #196, kernel work is nearly exhausted as a lever. The
whole-step graph (T4) removed the launch gaps *inside* a step; what remains
is the host round trip *between* steps — the same class of cost as the 1.2 ms
resident-page scan, which no CUDA profile could see and which only a
host-time phase split found.

## Rule

**Occupancy is the first number to read in a two-sided ledger, before any
per-kernel row.** 0.77 vs 0.90 said "the gap is idle" in one comparison, and
it is invariant to every per-kernel attribution question. Reading the kernel
table first led me to draft a norm+residual fusion tranche worth 0.156 ms
against a 1.9 ms gap — real, but an order below the actual lever.

Corollary: keep a term in the ledger that **must** match (here marlin, same
kernel and weights). If the control row disagrees, the harness is wrong and
no other row can be trusted.
