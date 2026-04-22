# Metal DFlash — single source of truth

Canonical user and engineering guide for DFlash speculative decode on the
Metal backend. Supersedes every prior DFlash resource/plan/readme fragment.

**Status:** default-on for all Metal serving paths (as of commit `47f958f`,
2026-04-19). Qwen3 (bf16), Qwen3.5 (hybrid 4-bit), and Qwen3.6-35B-A3B
(Qwen3.5-MoE target + DFlash draft) are supported on the Metal lane.

Last validated on the binary shipped with commits `3bc8802` (prefill
fast-forward) + `d8cb2f4` (batched `async_eval` defer), 2026-04-20.

## TL;DR — one command

```bash
./scripts/run_dflash.sh                # Qwen3.5-4B-4bit DFlash server, :8000
./scripts/run_dflash.sh bench          # baseline vs DFlash throughput table
./scripts/run_dflash.sh request "hi"   # one-shot chat against the running server
```

The script handles the build flags, default model pair, bind, and DFlash
draft wiring. Run `./scripts/run_dflash.sh help` for the full menu.

## What DFlash is

Draft-assisted speculative decode on Apple Silicon:

1. A small draft model proposes `block_size` candidate tokens
2. The target model runs one forward over the whole block
3. A verify step accepts the longest prefix that matches greedy target output
4. The accepted tokens replace that many sequential decode steps

The target model still produces the final output. DFlash is a throughput
lever, not a model change.

## Supported today

| Dimension | Supported |
|---|---|
| Backend | Metal (Apple Silicon, M-series) |
| Build flags | `--no-default-features --features metal,no-cuda` |
| Target families | `Qwen3` (bf16), `Qwen3.5` (4-bit hybrid GDR + full-attn), `Qwen3.6-35B-A3B` (Qwen3.5-MoE) |
| Entry points | `metal_request`, `metal_bench`, `metal_serve` |
| Default draft pair | `mlx-community/Qwen3.5-4B-MLX-4bit` + `z-lab/Qwen3.5-4B-DFlash`; `mlx-community/Qwen3.6-35B-A3B-4bit` + `z-lab/Qwen3.6-35B-A3B-DFlash` |
| Concurrency | c=1..8 HTTP clients validated; batched path single-forward verify for B≥1 |
| Long prompts | ≥4k tokens OK (scheduler `fast_forward_prefill`, commit `3bc8802`) |

Not supported:

- CUDA scheduler (DFlash is Metal-only today)
- Draft / target with mismatched `hidden_size` (runtime refuses, see `dflash.rs:228-233`)

## Validated model pairs

| Target | Draft | Provenance |
|---|---|---|
| `mlx-community/Qwen3.5-4B-MLX-4bit` | `z-lab/Qwen3.5-4B-DFlash` | [2026-04-19 default-on ship](../experience/wins/2026-04-19-metal-qwen35-concurrent-dflash-default-on.md), [2026-04-20 prefill fast-forward](../experience/wins/2026-04-20-dflash-prefill-fastforward.md), [2026-04-20 batched `async_eval`](../experience/wins/2026-04-20-dflash-batched-async-eval.md) |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | `z-lab/Qwen3.6-35B-A3B-DFlash` | local smoke + single-request DFlash landing in `feat(metal): support qwen36 dflash draft` |
| `mlx-community/Qwen3-4B-bf16` | `z-lab/Qwen3-4B-DFlash-b16` | [2026-04-14 Qwen3 5.9× decode bench](../experience/wins/2026-04-14-metal-dflash-qwen3.md) |

## Runtime map

Minimal architecture diagram, directly matching the implementation:

```text
metal_request / metal_bench / metal_serve
  -> MetalBackend::generate*                    (infer/src/backend/metal.rs)
    -> DFlash requested?
      no  -> Qwen3: generate.rs::metal_generate
          -> Qwen3.5/Qwen3.6: qwen35.rs::metal_generate_qwen35
      yes -> target arch check + draft load     (metal/dflash.rs::load_or_fallback)
          -> Qwen3 target: dflash.rs::metal_generate_dflash_qwen3
          -> Qwen3.5/Qwen3.6 target:
             qwen35.rs::metal_generate_qwen35
               -> metal_generate_qwen35_dflash
                 -> qwen35.rs::with_qwen35_capture_layers
                 -> qwen35_compiled_step prefill for prompt
                 -> capture_qwen35_hidden_from_cpp_outputs
                 -> dflash.rs::qwen35_dflash_speculative_block

metal scheduler runtime:
  metal_serve -> runtime.rs::run_metal_scheduler_runtime
    -> request_state.rs::Qwen35StepDriver::{prefill_token,prefill_tokens,decode_token}
      -> qwen35.rs::with_qwen35_capture_layers
      -> ensure_dflash_target_hidden_for_terminal_prefill
      -> qwen35_dflash_speculative_block
         single-row: sampled full-block verify via
         `verify_block_summary(cache_pos)` + accepted-prefix slice from
         the staged block tokens + GDR rollback on rejection
         + token-only next-block prefetch
      -> qwen35_dflash_speculative_block_batched
         multi-row: packed full-block verify over `[B, block_size]`
      -> fallback to standard decode when target_hidden is still missing
         or the request is on the Rust step path
```

## Trigger, routing, fallback

- Trigger:
  `--dflash-draft-model` enables DFlash. Without it, Metal stays on the plain target path.
- Target routing:
  `MetalBackend::generate_from_token_ids_with_callback` dispatches by loaded weight family.
  `MetalWeights::Qwen3` goes to the Qwen3 draft path; `MetalWeights::Qwen35` covers both dense Qwen3.5 and Qwen3.6-MoE.
- Prefill:
  Qwen3.5/Qwen3.6 DFlash prefill runs through the compiled C++ target model.
  Both the single-request path and the scheduler path now share
  `qwen35.rs::with_qwen35_capture_layers` for the capture-layer setup/reset,
  then `capture_qwen35_hidden_from_cpp_outputs` builds the layer-hidden bundle
  that seeds the first draft block.
- Verify:
  `qwen35_dflash_speculative_block` and
  `qwen35_dflash_speculative_block_batched` now diverge deliberately:
  single-row DFlash uses the native scalar-cache sampled verify entrypoint,
  `CppQwen35Model::verify_block_summary`, which samples inside C++ and returns
  only `(matched_prefix_len, next_token)` plus updated KV/GDR state. Rust then
  slices the accepted prefix directly out of the staged block tokens. That
  keeps the single-row control path simple, avoids the old posterior-block
  readback, and matches the `dflash-mlx` shape more closely: prefetch keeps
  only `seed_token + block_tokens`, while the live draft cache stays in the
  canonical `draft_state` instead of cloning a second cache snapshot.
  Draft sampling, the prefill `staged_first` token, scalar verify summary, and
  packed sampled verify now all suppress the draft runtime's `mask_token_id`
  before argmax/categorical sampling, matching the reference's
  `greedy_tokens_with_mask(..., suppress_token_mask)` contract.
  Batched DFlash still verifies the whole packed block in one forward and
  applies the same rollback rule row-wise. The packed route now also samples the draft
  suffix in one `linear + sample_rows_array` pass over the flattened
  `[B * (block_size - 1), hidden]` slab and threads per-row `cache_pos_arr`
  into C++ as a host int32 slice, avoiding the old per-row draft sampling
  loop and the extra MLX `cache_pos_arr` materialization fence inside
  `full_attn_step`. The packed verifier also no longer reads back the full
  `[B, block_size]` posterior token matrix to CPU just to find the accepted
  prefix: it slices the packed draft/target prefixes once, runs a dedicated
  `[B, T] -> [B]` `prefix_match_len_i32_batched` kernel on GPU, gathers the
  emitted posterior token with one `[B, T] + [B] -> [B]`
  `gather_axis1_i32` pass, and only materializes those two `[B]` vectors back
  to Rust. On the compiled full-attention sublayers, the packed path may use the verify-only
  `batched_sdpa_2pass` kernel when the verify block is mask-free, truly
  batched (`B > 1`), and `block_size == 16`; otherwise it falls back to stock
  MLX SDPA. For single-row `B=1, S=16` verify, the compiled Qwen3.5/Qwen3.6
  target model now threads one `prefer_verify_m16` bit from `ForwardContext`
  down through full-attention, GDR, MLP, MoE, and final logits, so eligible
  verify sublayers reshape `[1, 16, H] -> [16, H]` once and stay on one
  canonical `quantized_matmul` / `qwen35_moe_block_forward_cpp` path instead
  of bouncing through per-layer special cases. Both paths return the same
  accepted-token contract and updated target hidden state.
- Scheduler fallback:
  `Qwen35StepDriver::decode_token` keeps one canonical escape hatch: if
  terminal prefill did not seed `target_hidden` yet, or the request is on the
  Rust step path instead of the compiled C++ path, Metal falls through to the
  standard target decode path for that token instead of aborting the request.
- Fallback:
  `metal/dflash.rs::check_compatibility` refuses only true shape mismatches now. For Qwen3.6, rebucketed draft heads are accepted when `q_proj_width` and `kv_proj_width` match the target. Any compatibility failure logs a warning and falls back to standard Metal generation instead of aborting startup.

## Build

```bash
cargo build --release --no-default-features --features metal,no-cuda
```

## Running

### Server (`metal_serve`)

```bash
./target/release/metal_serve \
  --model-path mlx-community/Qwen3.5-4B-MLX-4bit \
  --dflash-draft-model z-lab/Qwen3.5-4B-DFlash \
  --port 8000
```

OpenAI-compatible endpoints live under `/v1/*`. Point any OpenAI-compatible
client at `http://127.0.0.1:8000/v1`.

### One-shot request (`metal_request`)

```bash
./target/release/metal_request \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --dflash-draft-model z-lab/Qwen3.5-4B-DFlash \
  --prompt "write a quicksort in python" \
  --raw-prompt \
  --max-new-tokens 128
```

### Bench (`metal_bench`)

Baseline — DFlash off:

```bash
./target/release/metal_bench \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --prompt-tokens 32 --generation-tokens 256 --warmup 1 --runs 3
```

DFlash on:

```bash
./target/release/metal_bench \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --dflash-draft-model z-lab/Qwen3.5-4B-DFlash \
  --prompt-tokens 32 --generation-tokens 256 --warmup 1 --runs 3
```

## Parameters

See [`metal-dflash-params.md`](metal-dflash-params.md) for the full
parameter table. The common flags:

| Flag | Default | Meaning |
|---|---|---|
| `--dflash-draft-model <PATH_OR_REPO>` | unset (off) | Enable DFlash; local path or HF repo id |
| `--speculative-tokens <N>` | draft-config default | Override block size (rarely needed) |

Rule: unset `--speculative-tokens` unless bench data says otherwise. The
draft checkpoint ships a trained default; smaller values can reduce
acceptance and throughput.

## Known performance ceilings (2026-04-20)

- **Qwen3.5-4B-4bit single-stream**: step time ≈ `4.4 + 6.3·B` ms on M4 Max
  (40 cores) — the GDR recurrent kernel is the limiter, measured at 6.1 ms/row.
  DFlash is either neutral or a small win depending on workload; concurrency
  ≥ 2 scales linearly through B=8.
- **Full-concurrency ceiling**: ~145 tok/s aggregate at c=8 (Qwen3.5-4B-4bit).
  Further gains require GDR kernel work (profile via Xcode Metal capture —
  see [`metal-gdr-kernel-xcode-capture.md`](../plans/metal-gdr-kernel-xcode-capture.md)).
- **Qwen3-4B bf16**: 5.9× decode speedup (25.9 → 152.0 tok/s) on M4 Pro at
  `prompt=20, generation=256`.

## Debug env vars

All are off unless explicitly set. See `infer/src/backend/metal/dflash.rs`
for definitions; listed here so operators know what's supported:

| Var | Location | Meaning |
|---|---|---|
| `DFLASH_DRAFT_MASK=causal` | `dflash.rs:271` | Force causal draft attention (diagnostic only) |
| `DFLASH_DRAFT_CPP=<path>` | `dflash.rs:534` | Override compiled draft kernel cache path |
| `QWEN35_DFLASH_PROFILE=1` | `dflash.rs:1870` | Emit per-block profiling to stderr |
| `INFER_CAPTURE_STEP=N` + `MTL_CAPTURE_ENABLED=1` | GPU capture hook | Xcode trace one step; see [GDR capture runbook](../plans/metal-gdr-kernel-xcode-capture.md) |

## Troubleshooting

**"draft hidden size mismatch" at load time**
Target and draft must share `hidden_size`. The validator at
`dflash.rs:228-233` enforces this.

**`WrongPhase` errors on long prompts (>512 tokens)**
Fixed on 2026-04-20 by `fast_forward_prefill` (commit `3bc8802`). If you
see this on a build older than `3bc8802`, rebuild.

**Throughput worse than plain decode**
1. Re-run without `--dflash-draft-model` for a fair baseline on the same
   binary.
2. Remove `--speculative-tokens` if set.
3. Compare `generation_tps`, not total wall time.
4. For small effects (<10%): run matched-A/B in two separate sessions per
   [`feedback_matched_ab_for_small_bench_effects.md`](../../memory/feedback_matched_ab_for_small_bench_effects.md)
   before concluding either way.

**No tokenizer in draft repo**
Expected. The target tokenizer is the source of truth; the draft checkpoint
needs only config + weights.

## Related docs

- [`metal-dflash-params.md`](metal-dflash-params.md) — full parameter table
- [`../plans/metal-gdr-kernel-xcode-capture.md`](../plans/metal-gdr-kernel-xcode-capture.md) — GPU profiling runbook
- [`../experience/wins/2026-04-19-metal-qwen35-final-state.md`](../experience/wins/2026-04-19-metal-qwen35-final-state.md) — terminal state of the Qwen3.5 DFlash arc
- [`../experience/errors/2026-04-19-dflash-long-prompt-prefill-chunking-desync.md`](../experience/errors/2026-04-19-dflash-long-prompt-prefill-chunking-desync.md) — root cause for the prefill fast-forward fix
- Code: `infer/src/backend/metal/dflash.rs`, `scheduler.rs::fast_forward_prefill`, `crates/mlx-sys/src/mlx_dflash_draft_model.cpp`
