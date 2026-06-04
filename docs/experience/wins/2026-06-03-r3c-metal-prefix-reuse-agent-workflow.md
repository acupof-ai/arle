# R3c: Metal prefix reuse and chunked prefill through the clean engine

## Goal

Correctness and first workflow evidence for R3c on branch
`arch/ideal-inference-engine`: make `infer_metal::MetalExecutor` handle
multi-chunk prefill and radix-attached prefix pages behind the host-only
`infer-core` seam, then run the multi-turn agent workflow with real MLX tokens.

## What Worked

`MetalExecutor` now drains each scalar Qwen3.5 C++ session into backend-private
MLX K/V and GDR arrays, publishes full page-aligned K/V blocks into a
`MetalPageStore`, and restores both K/V pages and the matching GDR snapshot when
`infer-core` attaches radix page ids to a fresh slot.

Chunked-prefill parity vs the legacy `MetalBackend` stayed bit-identical on
`mlx-community/Qwen3.5-0.8B-MLX-4bit`:

```text
legacy=[814, 20139, 3069, 8978, 45850, 12482, 364, 7072, 61794, 10505, 12636, 13, 814, 20139, 3069, 8978]
new   =[814, 20139, 3069, 8978, 45850, 12482, 364, 7072, 61794, 10505, 12636, 13, 814, 20139, 3069, 8978]
```

Command:

```bash
cargo test --release -p infer --no-default-features --features metal,no-cuda \
  r3c_chunked_prefill_matches_legacy_metal_backend -- --ignored --nocapture --test-threads=1
```

The multi-turn agent workflow now runs instead of failing with the old
"prefix reuse or chunked prefill" guard:

```text
[agent-workflow METAL Qwen3.5-0.8B] turns=3 total_gen=144 total_wall=768.825083ms tok_per_s=187.3 os_impact=OsImpactReport { samples: 3, peak_rss_bytes: 0 }
  turn 0 prompt_len=288 gen=48 ttft_ticks=6 wall=304.817917ms
  turn 1 prompt_len=368 gen=48 ttft_ticks=3 wall=219.417625ms
  turn 2 prompt_len=448 gen=48 ttft_ticks=3 wall=244.558375ms
```

Command:

```bash
cargo test --release -p agent-bench --features metal \
  bench_agent_workflow_metal_qwen35_08b -- --ignored --nocapture --test-threads=1
```

## Rule

For Qwen3.5 hybrid models, a host page-id prefix hit is not sufficient evidence
of a hot Metal prefix. The executor must also restore the matching GDR recurrent
and convolution state for the same page-aligned token length. Treat prefix reuse
as valid only when both K/V pages and the GDR snapshot are present.

## Status

R3c is passing for the single-slot Qwen3.5 clean-engine path. R3d still needs
packed/mixed batches, and the current `MetalPageStore` is conservative: it has no
seam callback for host page release, so it overwrites on page reuse and does not
proactively evict backend-private MLX page storage.
