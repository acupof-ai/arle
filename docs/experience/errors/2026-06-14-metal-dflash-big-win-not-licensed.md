# Metal DFlash big-win license killed

## Context

After the Qwen3.6 Metal DFlash correctness and scheduler-row reachability gate,
the next question was whether the path could already produce a very large
performance win. I added opt-in trace telemetry behind
`INFER_METAL_DFLASH_TRACE=1` and ran same-prompt A/Bs on the canonical Metal
target:

- target: `mlx-community/Qwen3.6-35B-A3B-4bit`
- draft: `z-lab/Qwen3.6-35B-A3B-DFlash`
- KV dtype: `int8`
- DFlash: `block_size=16`, `max_rows=4`, `target_layers=[1, 10, 19, 28, 37]`

The trace records per speculative block: draft, target verify, tape/hidden
drain, rollback, commit, eval, total time, and accepted token count.

## Results

Long-prefill shape, same raw non-thinking prompt:

| path | prompt tokens | output tokens | wall |
|---|---:|---:|---:|
| target-only | 19,827 | 64 | 79.433s |
| DFlash trace | 19,827 | 64 | 78.227s |

DFlash block trace for the 64-token run:

| metric | value |
|---|---:|
| blocks | 35 |
| accepted tokens | 65 |
| mean accepted/block | 1.857 |
| steady draft/block | 26.996 ms |
| steady verify/block | 182.525 ms |
| steady total/block | 221.040 ms |

Decode-heavy shape, same raw non-thinking prompt after a small warmup:

| path | prompt tokens | output tokens | wall |
|---|---:|---:|---:|
| target-only | 41 | 256 | 3.038s |
| DFlash no-trace | 41 | 256 | 10.933s |
| DFlash trace | 41 | 256 | 10.977s |

DFlash block trace for the 256-token run:

| metric | value |
|---|---:|
| blocks | 145 |
| accepted tokens | 255 |
| mean accepted/block | 1.759 |
| accepted histogram | 1:78, 2:43, 3:14, 4:4, 5:3, 6:3 |
| draft/block | 17.909 ms |
| verify/block | 55.313 ms |
| total/block | 74.481 ms |
| trace eval/block | 0.946 ms |

The trace overhead is not the explanation: no-trace and trace wall times were
essentially the same on the decode-heavy shape.

Follow-up draft-cost probe on the Qwen3.5 4B proxy
(`mlx-community/Qwen3.5-4B-MLX-4bit` +
`z-lab/Qwen3.5-4B-DFlash`) split the draft block internally:

| draft segment | mean/block |
|---|---:|
| embed | 0.157 ms |
| draft forward | 10.202 ms |
| slice | 0.051 ms |
| target lm_head | 11.132 ms |
| argmax/materialize | 0.231 ms |
| host token array | 0.001 ms |
| total draft | 21.774 ms |

The cheap knobs are not embed/argmax. The meaningful draft cost is split
between the draft model forward and the target lm_head projection.

Because accepted tokens were low, I also swept the existing
`INFER_METAL_DFLASH_TOKENS` knob on the same Qwen3.5 proxy, no trace, after a
small warmup:

| path | prompt tokens | output tokens | wall |
|---|---:|---:|---:|
| target-only | 41 | 256 | 3.190s |
| DFlash tokens=4 | 41 | 256 | 5.848s |
| DFlash tokens=8 | 41 | 256 | 9.173s |
| DFlash tokens=16 | 41 | 256 | 8.678s |

This proves the default 16-token draft block is not the minimum-cost setting for
low-acceptance prompts. The best cheap setting tested was 4, but it was still
1.83x slower than target-only on the proxy. I attempted the same Qwen3.6
`tokens=4` run, but the local Metal resource guard rejected startup:
`available=28.4GiB`, computed `memory budget 22 GiB`, fixed requirement
`25 GiB`; I did not bypass the guard.

## Root Cause

The current DFlash path has reachability but not the performance shape:

1. Acceptance is too low. Qwen3.6 accepted only 1.76-1.86 tokens per 16-token
   speculative block on the measured prompts.
2. Target verify dominates each block. On the decode-heavy shape, the block
   cost was 74.5 ms, of which 55.3 ms was target verify.
3. Scheduler multi-row support is still serial inside the executor. The Rust
   path loops over rows and calls `qwen35_speculative_block` once per row; it
   has not yet packed row KV/GDR state into the existing C++
   `qwen35_compiled_verify_block_batched_sampled` entrypoint.

Single-row DFlash needs `74.5 / accepted` ms per accepted token. Against the
measured target-only 11.9 ms/token, it needs more than 6.3 accepted tokens per
block just to break even, and more than 12.5 to reach a 2x decode-side win.
Measured acceptance was 1.76.

With true 4-row batched verify but still serial draft, the rough cost model is
`(55.3 + 4 * 17.9) / (4 * accepted) = 31.7 / accepted` ms/token. That would
need more than 2.7 accepted tokens per block to break even and more than 5.3
accepted tokens per block for a 2x decode-side win. Measured acceptance was
still below the break-even threshold.

## Fix

Do not claim a large DFlash win from the current path.

The next licensed implementation step is not rollback/commit
micro-optimization, and it is not embed/argmax cleanup. Rollback plus commit was
below 0.4 ms/block on the decode-heavy trace; proxy draft tracing showed
embed+argmax was also below 0.4 ms/block. The real critical path is:

1. wire true multi-row verify: batch row token blocks, pack row KV/GDR states,
   pass per-row cache positions and RoPE offsets, call
   `qwen35_compiled_verify_block_batched_sampled`, then unpack per-row accepted
   prefixes and rollback each row;
2. re-measure acceptance distribution on the SLO prompts, because batching alone
   cannot create a large win if accepted tokens stay near 1.8/block;
3. only after acceptance and true batched verify clear the break-even thresholds,
   run the same-prompt target-only vs DFlash c-sweep and write a wins entry.

## Rule

DFlash performance is licensed by wall-clock A/B plus accepted-token trace, not
by correctness reachability or row-admission logs. A "large win" requires both
true batched verify and enough accepted tokens per block; either one alone is
not sufficient.
