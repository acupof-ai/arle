# Metal DFlash Parameters

Parameter reference for the Metal DFlash entry points.

Scope:

- `metal_request`
- `metal_bench`
- `metal_serve`

This document is intentionally parameter-first. For support limits, benchmark
results, and usage guidance, see:

- [metal-dflash.md](metal-dflash.md)

## Shared DFlash parameters

These flags exist on all three Metal entry points.

| Parameter | Required | Default | Applies to | Meaning | Notes |
| --- | --- | --- | --- | --- | --- |
| `--dflash-draft-model <PATH_OR_REPO>` | No | disabled | `metal_request`, `metal_bench`, `metal_serve` | Enable Metal DFlash and load the draft checkpoint | Accepts a local path or Hugging Face repo id |
| `--speculative-tokens <N>` | No | draft-config default | `metal_request`, `metal_bench`, `metal_serve` | Override the speculative block size | Leave unset unless benchmark data says otherwise |

Rules:

- If `--dflash-draft-model` is omitted, the backend runs the normal Metal path.
- If `--speculative-tokens` is set lower than the draft checkpoint's default,
  runtime will warn because this can reduce acceptance and throughput.
- `--speculative-tokens 0` is invalid.

## `metal_request`

Purpose:

- one-shot prompt/response run
- smoke test
- quick local validation

Command shape:

```bash
cargo run -p infer --bin metal_request --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt "write a quicksort in python" \
  --raw-prompt
```

| Parameter | Required | Default | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `--model`, `-m <MODEL>` | Yes | none | Target model path or HF repo id | Example: `mlx-community/Qwen3-4B-bf16` |
| `--prompt <PROMPT>` | Conditionally | none | Inline prompt text | Mutually exclusive with `--prompt-file` |
| `--prompt-file <PATH>` | Conditionally | none | Read prompt from file | Mutually exclusive with `--prompt` |
| `--system <TEXT>` | No | none | System prompt for chat formatting | Ignored with `--raw-prompt` |
| `--raw-prompt` | No | `false` | Skip ChatML formatting | Recommended for low-level throughput checks |
| `--stream` | No | `false` | Stream text to stdout | Summary prints to stderr in stream mode |
| `--warmup <N>` | No | `1` | Warmup requests before the timed request | Use `0` for the fastest smoke test |
| `--max-new-tokens <N>` | No | `256` | Maximum generated tokens | Mapped to runtime `max_new_tokens` |
| `--temperature <F>` | No | `0.0` | Sampling temperature | `0.0` means greedy |
| `--top-k <K>` | No | `-1` | Top-k sampling | Metal currently accepts only `-1` or `1` |
| `--ignore-eos` | No | `false` | Keep generating past EOS | Useful for fixed-length measurement |
| `--kv-pool` | No | env fallback | Enable the experimental Metal KV pool | Only affects the Qwen3 fallback path today |
| `--no-kv-pool` | No | env fallback | Force-disable the Metal KV pool | Overrides `AGENT_INFER_METAL_KV_POOL` |
| `--dflash-draft-model <PATH_OR_REPO>` | No | disabled | Enable DFlash | Shared DFlash flag |
| `--speculative-tokens <N>` | No | draft default | Override block size | Shared DFlash flag |

Recommended smoke command:

```bash
cargo run -p infer --bin metal_request --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt "benchmark throughput" \
  --raw-prompt \
  --warmup 0 \
  --max-new-tokens 16
```

## `metal_bench`

Purpose:

- baseline vs. DFlash throughput comparison
- TTFT / prompt TPS / generation TPS / repo-E2E TPS measurement

Command shape:

```bash
cargo run -p infer --bin metal_bench --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt-tokens 20 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 3
```

| Parameter | Required | Default | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `--model`, `-m <MODEL>` | Yes | none | Target model path or HF repo id | Same target meaning as `metal_request` |
| `--prompt-tokens <N>` | No | `20` | Exact prompt token count for the synthetic benchmark prompt | Use a longer prompt to test prefill-heavy cases |
| `--generation-tokens <N>` | No | `256` | Exact output token count | Alias: `--max-tokens` |
| `--warmup <N>` | No | `3` | Warmup runs excluded from stats | Use `1` for quick local comparisons |
| `--runs <N>` | No | `5` | Timed runs | Mean / p50 / p99 are computed across these |
| `--profile` | No | `false` | Print per-run detail | Useful when diagnosing variance |
| `--json` | No | `false` | Emit machine-readable JSON | Good for snapshotting |
| `--save-baseline <PATH>` | No | none | Write current results as a baseline JSON file | Does not compare |
| `--compare-baseline <PATH>` | No | none | Compare against an existing baseline | Fails if metrics regress past thresholds |
| `--update-baseline <PATH>` | No | none | Overwrite a baseline only if thresholds pass | Safe update flow |
| `--baseline-compare` | No | `false` | One-shot baseline-vs-DFlash run with matched params | Mutually exclusive with `--dflash-draft-model`; picks the draft heuristically (see below) |
| `--kv-pool` | No | env fallback | Enable the experimental Metal KV pool | Only affects the Qwen3 fallback path today |
| `--no-kv-pool` | No | env fallback | Force-disable the Metal KV pool | Overrides `AGENT_INFER_METAL_KV_POOL` |
| `--dflash-draft-model <PATH_OR_REPO>` | No | disabled | Enable DFlash | Shared DFlash flag |
| `--speculative-tokens <N>` | No | draft default | Override block size | Shared DFlash flag |

Recommended benchmark pair:

Baseline:

```bash
cargo run -p infer --bin metal_bench --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --prompt-tokens 20 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 3
```

DFlash:

```bash
cargo run -p infer --bin metal_bench --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt-tokens 20 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 3
```

Recommended first workload:

- `--prompt-tokens 20`
- `--generation-tokens 256`

That is the currently validated generation-heavy case.

### `--baseline-compare` one-shot

Purpose:

- Single invocation loads the target twice (baseline Metal, then DFlash) with
  matched warmup / runs / prompt-tokens / generation-tokens and prints a delta
  row on the last stdout line.
- Saves you from maintaining two matched command lines when you just want
  "did DFlash help?"

Defaults for the draft model (heuristic, picks a known-good pair):

| Target model | Draft picked |
| --- | --- |
| `mlx-community/Qwen3.5-4B-*` / any path containing `Qwen3.5-4B` | `z-lab/Qwen3.5-4B-DFlash` |
| `mlx-community/Qwen3-4B-bf16` / any path containing `Qwen3-4B-bf16` | `z-lab/Qwen3-4B-DFlash-b16` |

If the target is not one of the above families, the bench bails out with a
clear error explaining which known pair it expected. Pass
`--dflash-draft-model` yourself for the standard two-command workflow;
`--baseline-compare` and `--dflash-draft-model` are mutually exclusive.

Example:

```bash
cargo run -p infer --bin metal_bench --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --baseline-compare \
  --prompt-tokens 20 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 3
```

Sample final stdout line:

```
compare | baseline TPOT 22.98 ms → DFlash 21.83 ms  (Δ -5.0%)
```

The delta is a percentage change on TPOT (time per output token); negative
means DFlash is faster. The rest of the bench output (per-phase TPS, p50 /
p99, JSON with `--json`) still prints per-run exactly as in the two-command
flow.

## `metal_serve`

Purpose:

- local OpenAI-compatible Metal server
- serial runtime validation

Command shape:

```bash
./target/release/metal_serve \
  --model-path mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --warmup 1 \
  --port 8000
```

| Parameter | Required | Default | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `--model-path <MODEL_PATH>` | Yes | none | Target model path or HF repo id | Server form uses `--model-path`, not `--model` |
| `--port <PORT>` | No | `8000` | HTTP listen port | OpenAI-compatible API |
| `--bind <HOST>` | No | `127.0.0.1` | Host or IP address to bind to | Pass `0.0.0.0` explicitly if you want non-local access |
| `--api-key <TOKEN>` | No | disabled | Require `Authorization: Bearer <TOKEN>` on `/v1/*` endpoints | Falls back to `AGENT_INFER_API_KEY` when the flag is omitted |
| `--max-waiting <N>` | No | `256` | Max queued requests before rejection | Server is still serial, not batched |
| `--kv-pool` | No | env fallback | Enable the experimental Metal KV pool | Only affects the Qwen3 fallback path today |
| `--no-kv-pool` | No | env fallback | Force-disable the Metal KV pool | Overrides `AGENT_INFER_METAL_KV_POOL` |
| `--dflash-draft-model <PATH_OR_REPO>` | No | disabled | Enable DFlash | Shared DFlash flag |
| `--speculative-tokens <N>` | No | draft default | Override block size | Shared DFlash flag |
| `--warmup <N>` | No | `1` | Number of startup warmup requests before serving traffic | Moves cold-start cost ahead of the first real request |
| `--warmup-prompt <TEXT>` | No | built-in short prompt | Prompt used for startup warmup | Keep it non-empty when `--warmup > 0` |
| `--warmup-max-new-tokens <N>` | No | `1` | Generated tokens per startup warmup request | `1` is enough to touch prefill + first decode |

Important server limitation:

- `metal_serve` is still a serial runtime. DFlash improves the single-request
  decode path; it does not add CUDA-style continuous batching.
- Startup warmup only reduces cold-start latency for the first live request.
  It does not change steady-state serial serving throughput.

### `/v1/models` DFlash status

When DFlash is loaded, `GET /v1/models` surfaces a `dflash` sub-object on
each model entry. The field is omitted (not present as `null`) when DFlash
is disabled so OpenAI-compatible clients that only read the legacy fields
keep working.

Shape:

```jsonc
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-4B-bf16",
      "object": "model",
      "created": 1713620000,
      "owned_by": "agent-infer",
      "dflash": {
        "enabled": true,
        "draft": "z-lab/Qwen3-4B-DFlash-b16",
        "speculative_tokens": 5,
        "acceptance_rate": 0.73
      }
    }
  ]
}
```

Field semantics:

- `enabled` — `true` whenever DFlash was successfully loaded at startup.
- `draft` — the draft model id that the backend loaded (the string the
  server resolved; may differ from the flag if the flag was a local path).
- `speculative_tokens` — the block size in effect (after any `--speculative-tokens`
  override or draft default).
- `acceptance_rate` — rolling draft-token acceptance rate over the most
  recent ~1000 DFlash blocks; `null` when no blocks have run yet (e.g.
  warmup-only server).

When DFlash load fails because the draft is incompatible with the target
(e.g. mismatched `hidden_size`), the server logs a `warn!` with the
specific field name and suggested fix, falls back to the standard Metal
path, and `/v1/models` omits the `dflash` sub-object — the server keeps
serving normally.

## Supported combinations

Working today (both default-on since commit `47f958f`, 2026-04-19):

- `Qwen3` (bf16) + Metal DFlash
- `Qwen3.5` (hybrid 4-bit) + Metal DFlash
- backend: `metal`
- build flags: `--no-default-features --features metal,no-cuda`

Rejected:

- CUDA backend (Metal-only today)
- DFlash without a draft model
- Draft / target with mismatched `hidden_size`, `num_attention_heads`,
  `num_key_value_heads`, `head_dim`, or out-of-range `target_layer_ids`.
  These cases previously panicked inside the C++ FFI; they now log a
  `warn!` naming the field + both values + a fix suggestion and fall
  back to the standard Metal path instead of crashing the backend.

## Removed environment variables

These environment variables are no longer the intended user interface:

- `AGENT_INFER_METAL_DFLASH_MODEL`
- `AGENT_INFER_METAL_DFLASH_SPECULATIVE_TOKENS`
- `AGENT_INFER_METAL_KV_POOL`

Use explicit CLI flags instead.

## Practical defaults

Use these defaults unless you are actively benchmarking:

- `--dflash-draft-model`: set it explicitly
- `--speculative-tokens`: leave unset
- `metal_request --warmup`: `0` for smoke, `1` for normal usage
- `metal_bench --warmup`: `1`
- `metal_bench --runs`: `3`

## Troubleshooting by parameter

If DFlash does not activate:

1. Check `--dflash-draft-model`.
2. Check that the target is `Qwen3`.
3. Check that the build uses `--features metal,no-cuda`.

If throughput gets worse:

1. Remove `--speculative-tokens`.
2. Re-run the same command without `--dflash-draft-model`.
3. Compare `generation_tps` on the same workload.
