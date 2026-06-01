# DSv4 native DeepEP transport killed by replicated-token TP/EP

## SLO-shape probed? -- N (p2048/o1 trace + p2048/o32 transport A/B only)

## Context

Goal: make the DSv4 serving chain correct before any default-on or SLO claim.
The target stack is FlashMLA attention plus DeepGEMM routed experts on 8xH20.
The open question was why `native-deepep + DeepGEMM` stayed slower than the
all-reduce path after dispatch/combine became reachable.

Remote artifacts:

- `/sgl-workspace/bench-artifacts/dsv4-trace-b13df146-native-deepep/deepgemm-p2048`
- `/sgl-workspace/bench-artifacts/dsv4-trace-b13df146-allreduce/deepgemm-p2048`
- `/sgl-workspace/bench-artifacts/dsv4-native-numrecv-b13df146/deepgemm-p2048`
- `/sgl-workspace/bench-artifacts/dsv4-native-alloc-b13df146/deepgemm-p2048`

## Root Cause

The current DSv4 TP/EP serving path keeps the same token rows on every rank.
That contract matches local expert execution followed by expert all-reduce:
each EP rank computes its local expert contribution for the full token batch,
then the ranks sum the final hidden states.

Native DeepEP dispatch/combine is not a drop-in replacement for that contract.
DeepEP assumes the caller owns a distinct set of input token rows per EP rank.
ARLE instead passed the full prompt token matrix from every TP/EP rank into
`Buffer::dispatch`.

For p2048:

- source tokens per rank: `2047`
- EP ranks: `8`
- routed experts per token: `6`
- observed average `sum(num_recv) / (8 * src_tokens)`: `4.46`
- theoretical distinct-rank fanout for top-6 over 8 ranks:
  `8 * (1 - (7/8)^6) = 4.41`

The observation matches the fanout model. Native DeepEP was moving and reducing
about 4.4x more token rows than the all-reduce contract needs, then paying the
cross-rank queue/barrier cost in combine.

Representative trace:

| Path | p2048/o1 TTFT | Dominant evidence |
| --- | ---: | --- |
| `native-deepep + DeepGEMM` | `7783 ms` | `ffn_native_deepep_combine` dominates |
| `allreduce + DeepGEMM` | `6021 ms` | `ffn_all_reduce` much smaller |

The p2048/o32 non-trace transport run also favored all-reduce (`~5336 ms TTFT`)
over native DeepEP (`~7758 ms TTFT`).

## Secondary Finding

The native path still allocates a fresh zeroed `combined_x` per layer:

```text
infer/src/model/deepseek/mlp.rs:5467
HiddenStates::zeros, 43 calls, 721068032 bytes per rank/request
```

This should be removed by reusing native DeepEP scratch, but it is not the main
root cause. The main root cause is the transport fanout from replicated token
ownership.

## Fix

DSv4 now defaults to the correct transport for the current architecture:

- `ARLE_DSV4_MOE_BACKEND` unset or `allreduce` -> local routed experts plus EP
  all-reduce.
- Expert math remains orthogonal to transport and defaults to DeepGEMM auto
  when available.
- `native-deepep` remains explicit, but the replicated-token TP/EP path refuses
  it unless `ARLE_DSV4_NATIVE_DEEPEP_REPLICATED_TOKENS_UNSAFE=1` is set for
  trace-only experiments.

This is a correctness fix, not an SLO pass. SLO and `>20% vs SGLang` still need
fresh apples-to-apples benchmarks on the required workload.

## Rule

Do not replace expert all-reduce with DeepEP dispatch/combine unless token
ownership changes too. Transport APIs encode a data-distribution contract, not
just a faster collective implementation. For DSv4, native DeepEP needs a
token-sharded EP caller before it can be a correct default.
