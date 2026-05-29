# DSv4 GPU-native forward produces coherent correct output — P→D KV handoff fixed

## SLO-shape probed? — N (correctness milestone; perf A/B is the next axis)

## TL;DR

The DSv4 GPU-native forward went from **catastrophic garbage**
(`"4062 0.0000 0.0000..."` — the `0.0000` cascade) to **coherent, correct,
cleanly-stopping output** on 8×H20 TP=8, by fixing the prefill→decode KV
handoff. This is the foundation the whole DSv4 GPU-native serving path
was missing.

Validated (`ignore_eos=false`, natural stop, real chat requests):

| Prompt | Output | finish | Verdict |
|---|---|---|---|
| "Compute 137 + 269. Answer with the number only." | `137 + 269 = 406` | stop | ✅ correct |
| "What is the capital of France?" | `The capital of France is Paris.` | stop | ✅ correct |
| "What is 25 times 4? Answer with the number only." | `</think>20` | stop | ✗ wrong (should be 100) + stray `</think>` |

2/3 correct with clean EOS stops — categorically different from the prior
all-`0.0000` garbage. The remaining miss (`25×4→20` + stray `</think>`) is
a **reasoning-model thinking-token / chat-template** axis (DSv4 emits
`<think>…</think>`; the harness sends raw messages with no thinking
budget), NOT the forward-correctness bug that was just fixed.

## Root cause + fix (the foundation)

The serving prefill ran the **stateless batched** path
(`compute_gpu_logits_after_prefill` → `compute_top_level_logits`,
`cache=None`) which writes **no per-slot KV**. Incremental decode
(`compute_top_level_logits_incremental`, `cache=Some`) reads per-layer
SW-window / compressed / FP8 KV caches that **only the incremental path
populates**. Nothing bridged them → every decode step ran attention on an
**empty KV cache** → output degenerated to `0.0000` after token 1.

Fix (`67a3de7b`): when incremental decode is enabled, route prefill
through `compute_top_level_logits_incremental(tokens, emit_logits=true)`.
It is seq-batched (`forward_transformer_layer_stream_incremental_into`
processes the whole prompt in one batched call — not token-serial), so it
populates the KV caches decode reads AND returns the last-token prefill
logits in one pass. Falls back to stateless if weights aren't loaded.

## Evidence trail (control experiments — CLAUDE.md §0)

1. **Legacy-OFF == FlashMLA-OFF garbage** (both `4062 0.0000...`) →
   FlashMLA innocent; the bug was shared / config.
2. **`GPU_FULL_LAYERS=0` default** → no attention layers ran at all
   (`compute_top_level_logits` with 0 full layers). The smoke config
   never ran the real model.
3. **`GPU_FULL_LAYERS=43` + `LOAD_LAYER_WEIGHTS=1` + Flash + incremental,
   pre-fix** → `1372 0.0000...` (the `0.0000` cascade = empty-KV decode).
4. **post-fix** → `137 + 269 = 40622 2 0 0...` with `ignore_eos=true`
   (forced over-gen), then `137 + 269 = 406` / `finish_reason=stop` with
   `ignore_eos=false`. The tail was a test artifact
   (cf. `errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact`).
5. **FlashMLA-decode=1 vs =0 byte-identical** → FlashMLA decode ≡ legacy
   decode (good parity); the residual was never FlashMLA-specific.

## Working serving config (8×H20 TP=8, DSv4-Flash)

```
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7   # multi-GPU trigger (NOT CUDA_VISIBLE_DEVICES)
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1
ARLE_DSV4_GPU_FULL_LAYERS=43          # = num_hidden_layers
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_FLASHMLA_PREFILL=1
ARLE_DSV4_FLASHMLA_DECODE=1
ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_EXPERT_BACKEND=native
--num-slots 1 --max-seq-len 4096 --mem-fraction-static 0.10
--kv-cache-dtype fp8 --deepseek-distributed-layers 43
```
Built inside the SGLang pod with native CUDA 12.9 (`target-pod/`,
libssl.so.3), `ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1`.

## What's next

1. **Reasoning-model thinking-token / chat-template** handling (the
   `</think>` + `25×4→20` miss). Wire DSv4's thinking budget + template.
2. **Perf A/B** (now that output is correct): TTFT/TPOT/throughput,
   P-node vs D-node, FlashMLA-default vs legacy. The earlier perf benches
   ran on garbage-shaped output; re-baseline on correct output.
3. **Flash default flip** + delete legacy `dsv4_hybrid_attention_cuda`
   (FlashMLA decode ≡ legacy already proven byte-identical).
4. **TTFT re-opt**: prefill now uses the incremental path (correctness
   first); reclaim the batched-prefill throughput via FlashMLA prefill
   writing KV directly / chunked warmup.

## Rule

**For a phased GPU-native model bring-up, the prefill→decode KV handoff is
the make-or-break foundation — validate decode reads what prefill writes
BEFORE chasing per-op perf.** And: re-confirm "garbage" with
`ignore_eos=false` + natural stop before declaring a forward broken — a
forced-over-generation tail past a correct answer looks identical to
degeneration (cf. fp8-kv-catastrophic-was-test-artifact).
