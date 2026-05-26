# Qwen3.5 Hybrid Mixed KILL — plan=mixed is not enough

## Context

Follow-up to [`docs/plans/2026-05-25-cuda-perf-codex-collab.md`](../../plans/2026-05-25-cuda-perf-codex-collab.md)
Axis 2. L4 was unavailable, so the CUDA validation ran on V100 in
`/home/chenkailun.c/agent-infer-v100-audit` with cached `Qwen3.5-4B`.

Goal: remove the scheduler Split fallback and make Qwen3.5 hybrid support the
same mixed decode+prefill contract as Qwen3 dense.

The first experiment tried a prefill-backed shortcut and was killed because
decode rows ran through the paged-prefill path. The second experiment kept
decode rows on Qwen3.5's optimized `decode_batch` path and ran prefill rows via
`prefill_forward_paged_batch` inside `forward_mixed_batch`.

## Evidence

Build passed on V100:

- `CUDA_HOME=/usr/local/cuda-12.4`
- `TORCH_CUDA_ARCH_LIST=7.0`
- `cargo build --release -p infer --bin infer --features cuda`

GuideLLM shape for all probes:

- `--concurrencies 16`
- `--max-seconds 60`
- `--warmup 5`
- prompt 4096 tokens, output 256 tokens
- server `--num-slots 16 --max-seq-len 5120`

| experiment | raw artifacts | plan evidence | GuideLLM result |
|---|---|---|---|
| prefill-backed mixed | `bench-output/2026-05-26-axis2-qwen35-mixed-v100-run3/` | `plan=mixed` appeared | c=16 invalid, 0 successful |
| decode-first, cap=512, rows=1 | `bench-output/2026-05-26-axis2-qwen35-hybrid-mixed-v100-cap512/` | mixed=190, split=0 | c=16 invalid, 0 successful |
| decode-first, cap=2048, rows=1 | `bench-output/2026-05-26-axis2-qwen35-hybrid-mixed-v100-cap2048-c16probe/` | mixed active | c=16 invalid, 0 successful |
| decode-first, cap=2048, rows=4 | `bench-output/2026-05-26-axis2-qwen35-hybrid-mixed-v100-rows4-c16probe/` | mixed=18, split=0 | invalid; paged prefill OOM |
| decode-first, cap=2048, rows=2 + continuation queue-front | `bench-output/2026-05-26-axis2-qwen35-hybrid-mixed-v100-frontqueue-c16probe/` | mixed=37, split=0 | invalid, 0 successful, 66 incomplete output tokens |
| decode-first, cap=512, rows=1 + continuation queue-front | `bench-output/2026-05-26-axis2-qwen35-hybrid-mixed-v100-frontqueue-cap512-c16probe/` | mixed=255, split=0 | invalid, 0 successful, 614 incomplete output tokens |

Representative failure logs:

```text
plan=mixed reason=mixed_policy_supported launch_order=single_mixed_launch
decode=phase:10,runnable:10 selected_prefill_rows=2 selected_prefill_tokens=2049
phase_us.decode=912162 phase_us.prefill=1482160 phase_us.total=2396743
```

```text
qwen35 mixed prefill rows=2 total_tokens=4096
caused by: Alloc chunk_state failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

```text
guidellm validation failed:
  - conc16: no successful requests recorded
```

## Root Cause

Qwen3.5 hybrid mixed cannot be made valid by only adding
`supports_mixed_batch()` and sequencing `decode_batch` plus paged prefill inside
one model method.

The decode-first path fixed the semantic bug from the prefill-backed shortcut:
decode rows used the optimized recurrent/full-attention decode kernels. It still
failed the SLO gate because each mixed tick also carried expensive hybrid
prefill work. At c=16, mixed ticks reached hundreds of milliseconds to seconds,
so incomplete requests received some tokens but no request completed 256 output
tokens inside the 60s GuideLLM window.

Two scheduler-side findings are separate:

- Requeueing partial prefill chunks at the tail amplifies TTFT because many
  sessions receive their first prefill chunk before an already-started prompt
  reaches its prefill completion token.
- Moving continuations to the queue front improves first-token order, but does
  not make this mixed implementation acceptable. Decode ITL still collapses
  while hybrid prefill remains inside mixed ticks.

Rows=4 is not viable on V100 because Qwen3.5 paged-prefill scratch OOMs. Rows=2
also hit OOM in mixed prefill after sustained c=16 pressure. Rows=1 with cap=512
avoids the immediate OOM but still produces 0 completed requests.

## Fix

Kill and revert the runtime experiment. Do not mark Qwen3.5 hybrid as
`supports_mixed_batch()` yet, and do not delete Split/default Mixed under the
claim that Qwen3.5 is now covered.

Qwen3.5 needs one of these before mixed can land:

- a real integrated hybrid mixed forward that does not serialize a full decode
  pass plus a full prefill pass per tick;
- or a scheduler/runtime split that lets decode sample/readback complete before
  prefill work occupies the stream;
- plus a SLO-aware mixed prefill budget validated by c=1,4,8,16.

## Rule

`plan=mixed` counters are necessary evidence, not sufficient evidence. A mixed
path only counts if the c-sweep has valid GuideLLM completions and no TTFT/ITL
or throughput regression beyond the gate.

