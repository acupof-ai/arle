# Accumulating slice gradients into the input's existing buffer took 6,210 MiB off the cp=2 backward — 2026-08-20

Commit: `530cbb2c7` (`perf(train): accumulate slice gradients into the input's
existing buffer`), on top of `62b4927b8` (transport chain residency drops).

## Context

Target is global sequence 262,144 on 2 GPUs. `--synthetic-writeback-seq N
--cp-size 2`, ThinkingCap-Qwen3.6-27B-FP8, LoRA r16 α32 attention-qv, 2×H20
(97,508 MiB). cp=2 means local seq = N/2.

Each gated-delta layer slices one packed qkv tensor three ways. Every
`slice_backward` allocated a zero-filled buffer the size of the whole packed
input, and `merge_grad` then summed the three — three full-size allocations
where one accumulated into three times does the same job.

## Result

Matched arms at cp=2 global 131,072 (local 65,536). `floor`, `layer`, and
`modeled` are identical to the digit across the two arms, so the comparison is
against the same weight and optimizer residency:

| local 65,536 | `floor` | `layer` | `modeled` | `actual` | `drift` |
|---|---:|---:|---:|---:|---:|
| before | 39,919 | 21,704 | 61,623 | 75,875 | +14,252 |
| after | 39,919 | 21,704 | 61,623 | **69,665** | **+8,042** |

**−6,210 MiB.** Loss is `3.036179` in both arms, bit-identical.

The same run also serves as the numerics gate for `62b4927b8` (transport chain
residency drops), which until now had no completed run to read a loss from — its
only run had died in the backward at 163,840.

## What worked

`slice_backward_into` takes the fused path only when the destination is provably
safe:

- the sliced input has a producing tape entry (`entry_by_output.contains_key`),
  so this is an intermediate and not a parameter gradient — parameters go
  through `accumulate_grad`, which this bypasses;
- both sides are already resident on device;
- the destination's device buffer has `strong_count == 1` — the same probe
  `merge_grad` uses, because gradients fan out by `Arc` clone and accumulating
  into a shared buffer would corrupt a live sibling.

It accumulates rather than stores. The first attempt reused
`write_slice_device`, which stores; that is wrong twice over — slices of one
tensor can overlap, and the destination may already carry a contribution from a
consumer that is not a slice at all. The CPU tests passed anyway because
`CpuBackend` takes the non-device fallback, so the bug was invisible until read
back off the diff.

## The ceiling did not move

Global 163,840 (local 81,920) still fails, on the same `zeros [1, 81920, 5120]`
(1.6 GB) in the backward, at the same `actual=77,026 MiB`. Identical to the
pre-change measurement to the digit — both arms die at the same allocation, so
the high-water is read at the same point in both. That is not evidence the
change was inert: the 65,536 rung, same binary, moved by 6,210 MiB.

The failing tensor carries the hidden dim (5,120), not the packed qkv dim
(8,960), so it is a different allocation than the one this change targets.

## Where 256K on 2 GPUs stands

Local 131,072 needs roughly `floor` 40,239 + `layer` 43,408 + drift, against
97,508 MiB. The gap is 5–10 GB, not a factor.

The other lever the bf16 audit named — recomputing the conv pair instead of
taping it — was implemented, measured at 0 MiB, and reverted; see
[`errors/2026-08-20-tensor-bytes-are-not-residency.md`](../errors/2026-08-20-tensor-bytes-are-not-residency.md).
The remaining gap is a transient the peak model does not see: `actual` reads
77,026 MiB at local 81,920 while the run dies on a 1.6 GB allocation, so roughly
20 GB lives between checkpoint boundaries and is unaccounted.

## Rule

A gradient buffer can be written in place only when the tape proves no one else
holds it. `strong_count == 1` on the device buffer is that proof; disjointness
of the regions is not.
