# conv1d decode fusion: two kernels into one, −0.079 ms/step — CUDA, 2026-08-04

> Status: Shipped

## Goal

Close the one row where the
[W8A16 kernel budget](2026-08-04-w8a16-decode-step-kernel-budget.md) put ARLE
behind SGLang: conv1d, 0.202 ms across two kernels against SGLang's single
`_causal_conv1d_update_kernel` at 0.125 ms.

## Hypothesis

The split exists because of a prefill race, documented in `conv1d.cu`: threads
at `t < kernel_size-1` read `conv_state` while the `t == seq_len-1` thread
would write the new ring, with no grid-level ordering. At **`seq_len == 1`**
there is exactly one thread per (slot, channel) and it reads the whole ring
before rewriting it, so the race cannot occur and the two kernels can be one.

The fusion cuts bytes rather than launches: the state-update kernel re-reads
the ring and the new token from global memory, and both are already in the
first kernel's registers. Inside a captured graph a launch is worth 0.084 µs
(same budget entry), so a launch-only fusion would return nothing — this one
should return the second kernel's full 1.618 µs.

## Parameters

- `conv1d_decode_kernel` in `crates/cuda-kernels/csrc/recurrent/conv1d.cu`,
  dispatched from both launchers on `seq_len == 1` / `max_len == 1`
  (`max_len == 1` forces every `row_len` to 1, so the varlen guard is exact)
- H20 GPU 6, Qwen3.6-27B W8A16, 33K prompt, c=1, shipped defaults
- `nsys --cuda-graph-trace=node`, 20 s decode window, 1160 steps

## Results

| kernel | launches/step | ns each | ms/step |
|---|---:|---:|---:|
| before — `conv1d_prefill` | 48 | 2583 | 0.124 |
| before — `conv1d_state_update` | 48 | 1618 | 0.078 |
| **after — `conv1d_decode`** | **48** | **2565** | **0.123** |

**−0.079 ms/step.** The fused kernel costs what the conv kernel alone cost;
the state-update kernel's time is gone entirely, which is the signature of a
fusion that removed traffic. A launch-only fusion would have returned
48 × 0.084 µs = 0.004 ms.

Whole-step figures moved with it but cannot confirm it on their own:
Σ kernel 16.651 → 16.596, `submit` 17.272 → 17.208. Both deltas are under the
0.09–0.115 ms noise floor measured on identical configurations
([entry](2026-08-04-fa3-decode-splits-fill-the-sms.md)). **The per-kernel row
is the measurement here; the step total is only a direction check.**

ARLE's conv1d row is now level with SGLang's 0.125 ms.

## Correctness

`lever_gate.sh` needle ladder 115/300/2000/8000/16000/32000 × 3, `RAW=1
TEMPLATE=qwen3_nonthink NEEDLE_MAX_TOKENS=256`, against the pre-fusion binary's
log at the same flags. Every rung `exact=3 partial=0 miss=0 DET`, envelope
comparison PASS, `GATE_EXIT=0`. The 32000 rung runs 33691 prompt tokens.

Arithmetic is unchanged by construction: the accumulation order over `k` is the
same, and the ring values round-trip bf16 → float → bf16, which is exact.

## Learnings

PASS. The last row where the kernel budget put ARLE behind SGLang is closed.

**Rule: fuse where the second kernel re-reads what the first already had.**
The budget's launch price (0.084 µs) is the floor on what any fusion returns;
anything above that comes from bytes not read. Check the second kernel's loads
against the first kernel's live registers before writing the fused version —
that check is what separates this 0.079 ms from the 0.00 ms of the
[residual-add fusion](../errors/2026-08-04-launch-count-is-not-the-decode-lever.md).

**Rule: a race that forces two kernels may not exist in every shape.** The
split here was correct for prefill and unnecessary for decode, and the decode
path is the one that runs a thousand times a second.

Related: [[feedback_measured_floor_is_not_physical_floor]],
[[feedback_matched_ab_for_small_bench_effects]].
