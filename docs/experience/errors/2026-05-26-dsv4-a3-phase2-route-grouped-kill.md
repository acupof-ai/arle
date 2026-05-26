# DSv4 A3 Phase 2 route-grouped dispatch KILL

## Context

A3 Phase 2 tried the existing opt-in `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1`
path as the cheapest Class A experiment. The goal was to remove the recv-side
local-count D2H from DeepEP decode without changing greedy output, then decide
whether this could become the default Phase 2 implementation.

Model path and host identifiers are intentionally omitted.

## Root Cause

The first route-grouped run removed the decode-window D2H, but long decode was
not byte-identical. The route-grouped `w2` GEMV multiplied the route weight
inside the GEMV before BF16 output rounding, while the baseline path writes
BF16 expert output first and multiplies the route weight during scatter. That
rounding-order difference stayed invisible on the short math prompt but
changed the 64-token longseq output from `The2 2 ...` to `The2 0 ...`.

Fixing that correctness issue did not make the path a wall-clock PASS. The
route-wise FP4 GEMV kernels and extra NCCL/allocator pressure still erase the
metadata win on long-prefill requests.

## Fix

The opt-in route-grouped path now preserves baseline weighting order:

- `w2` writes unweighted BF16 route output into grouped scratch.
- `dsv4_scale_route_outputs_by_meta_cuda` reads BF16 scratch, multiplies the
  route weight from `recv_meta`, and writes weighted BF16 into `route_out`.
- Non-local padded route slots are explicitly zeroed.

This keeps the old default path unchanged. `ARLE_DSV4_ROUTE_GROUPED_EXPERTS`
remains default-off.

## Results

### Correctness

| Workload | Before fix | After fix |
|---|---|---|
| short decode, `max_tokens=2` | `4062` | `4062` |
| longseq, `max_tokens=64` | `The2 0 0 ...` | `The2 2 2 ...` |
| longseq A/B, `max_tokens=32` | n/a | byte-identical |

### Wall-clock

| Workload | Baseline | Route-grouped fixed | Delta |
|---|---:|---:|---:|
| short decode mean, `max_tokens=2`, 3 measured requests | 0.7891 s | 0.7528 s | -4.60% |
| longseq, `max_tokens=32` | 108.7749 s | 110.2519 s | +1.36% |

The short decode result is below the A3 Phase 2 PASS gate of at least 5%, and
the user-facing longseq decode regresses.

### nsys

Single profile request, filtered to `step_decode_kernel_launch`:

| Metric | DeepEP baseline | Route-grouped fixed |
|---|---:|---:|
| decode wave wall | 252.760 ms | 155.814 ms |
| `cuMemcpyDtoHAsync_v2` runtime calls | 344 | 0 in filtered decode summary |
| D2H memcpy activity | 344 calls / 44,032 B | 0 in filtered decode summary |
| H2D memcpy activity | 696 calls / 12,416 B | 648 calls / 11,552 B |

This is a narrow-window win, not a shippable default: wall-clock framing wins.

## Artifacts

- DeepEP baseline nsys:
  `dsv4-a3-deepep-unsafe-baseline-nsys-20260526-114229`
- Pre-fix route-grouped nsys:
  `dsv4-a3-deepep-unsafe-routegrouped-nsys-20260526-114342`
- Pre-fix longseq A/B:
  `dsv4-a3-deepep-routegrouped-longseq-20260526-034841`
- Fixed longseq `max_tokens=64`:
  `dsv4-a3-routegrouped-scale-fix-long64-20260526-040659`
- Fixed longseq `max_tokens=32` A/B:
  `dsv4-a3-routegrouped-scale-fix-long32-ab-20260526-040949`
- Fixed nsys:
  `dsv4-a3-routegrouped-scale-fix-nsys-20260526-121432`
- Fixed short HTTP A/B:
  `dsv4-a3-deepep-routegrouped-http-20260526-041546`

## Rule

Do not default route-wise expert GEMV just because it deletes D2H in an nsys
decode window. For A3 Phase 2, the next viable path must be a true persistent
grouped GEMM/DeepGEMM-style expert path that preserves baseline rounding and
passes wall-clock on long decode, not more tuning of the current route-wise
GEMV prototype.
