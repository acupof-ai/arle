# Metal M4 Pro single-user ladder — 0.8B→35B fresh decode/TTFT/TPOT (front-door numbers)

## Goal
The README front door cited two perf numbers with no clean wins-entry source —
"197 tok/s on L4 (Qwen3.5-4B)" and "85.6 tok/s on M4 Pro (Qwen3.6)". An Explore
audit found the 197 had no c=16 backing in `wins/` and the 85.6 came from an
ad-hoc 2026-05-18 bench (not a recorded entry; a 2026-06-04 note even logged a
regression to ~48 tok/s). ckl: *"你可以直接本地测一下 … 本地有的模型都记录下 吞吐和
ttft tpot"*. This entry is the SOLID re-measurement of every locally-cached MLX
model on the Metal backend, so the front door shows measured truth, not folklore.

## Hypothesis
Qwen3.6-35B-A3B (MoE, ~3B active) decodes at roughly the 4B-dense rate despite
35B total params — the MoE value prop. Dense models scale ~1/params on the M4
Pro's ~273 GB/s unified memory.

## Params / Env
- **Hardware:** Apple M4 Pro, 48 GB unified memory (macOS 25.3.0).
- **Binary:** `target/release/arle` built `--no-default-features --features
  metal,no-cuda`, tree `4ea77e11`.
- **Serve:** `arle serve --backend metal --model-path <hf-id> --port 8000
  --max-prompt-tokens 6144 --max-total-tokens 8192` (auto wired-limit on).
- **Workload:** 512-in / 128-out, **c=1**, temp=0, `ignore_eos` (exact length),
  median of 6 (after warmup). Single user is the documented Metal local focus
  ([[feedback_metal_focus_c1_local]]); strictly serial — one model loaded at a
  time ([[feedback_no_concurrent_metal_loads_ssd_thrash]]).
- **Harness:** `scripts/bench_local_metal.py` (+ `bench_local_metal_all.sh`).
  The rewrite Metal serve defers `stream=true` (R5 tranche 2), so guidellm's
  streaming TTFT/ITL path can't run against it. Numbers come from a two-point
  non-streaming decomposition instead:
  `lat(max_tokens=1) ≈ TTFT`; `lat(max_tokens=128) = TTFT + 127·TPOT`;
  `decode tok/s = 1000/TPOT`. **Every call uses a unique nonce-prefixed prompt**
  so the RadixCache never hits — both points pay the same uncached prefill, which
  then cancels cleanly in the subtraction (a cache hit on one point but not the
  other corrupts TPOT; observed first-cut: a stray 21 ms cached TTFT dragged
  decode to a false 126 tok/s before the nonce fix — §0 confounder).

## Results (M4 Pro, 512-in/128-out, c=1, median of 6)

| Model (Metal, 4-bit) | Active | Decode tok/s | TPOT ms | TTFT ms | E2E tok/s |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B  | 0.8B | **317.8** | 3.15  | 168.5  | 225.3 |
| Qwen3.5-4B    | 4B   | 84.1      | 11.89 | 820.3  | 54.9 |
| Qwen3.5-9B    | 9B   | 50.0      | 20.01 | 1448.8 | 32.1 |
| Qwen3.6-35B-A3B · MoE | ~3B | **85.3** | 11.73 | 1231.0 | 47.1 |

- **Hypothesis confirmed:** the 35B-A3B MoE decodes at **85.3 tok/s — essentially
  the 4B-dense rate (84.1) and 1.71× the 9B dense (50.0)**, despite 4.4×/8.75× the
  weights, because only ~3B params activate per token. TTFT (1231 ms) sits between
  4B and 9B — prefill touches the router + active experts, not all 35B.
- **Dense ladder is ~roofline-shaped:** 0.8B→4B→9B decode 318→84→50 tok/s tracks
  the param ratio (a 4-bit weight-read-bound decode on ~273 GB/s).
- The old 85.6 Metal/Qwen3.6 claim was *coincidentally* close (measured 85.3);
  the 197 L4 number is **not** reproduced here (L4 is CUDA, not local) — dropped
  from the front door rather than carried unverified.

## Correctness
All four served coherent completions (`/v1/completions`, exact 128 tokens via
`ignore_eos`); served ids match the model dirs; no resource-guard rejections
after a transient 37 GiB working-set settled to ~34 GiB free pre-run.

## Supplement — every other locally-cached checkpoint (2026-06-14)
`scripts/bench_local_metal_supplement.sh` attempted the remaining local models so
coverage is measured, not assumed. One more **serves**: DiffusionGemma-26B-A4B-it-4bit
(block-diffusion fast path, `ARLE_DIFFUSION_MAX_DENOISING_STEPS=4`) at **55.7 tok/s
end-to-end** (≈ the support-matrix 60 tok/s claim; the AR two-point decode/TPOT are
not physically meaningful for diffusion). The rest **fail closed** at validation:
Qwen3.6-35B-A3B-**MTP**-4bit (`could not detect Qwen3.5 text weight prefix` — the MTP
head moves the layout); z-lab Qwen3.5-4B / Qwen3.6-35B-A3B **DFlash** (draft-only,
no tokenizer); Qwen2.5-0.5B/1.5B-bf16, Llama-3.2-1B-bf16, Qwen3-0.6B (`R3a Metal
executor requires Qwen3.5 layer_types`). **Coverage boundary: Metal serve = Qwen3.5/3.6
family + DiffusionGemma.** Recorded in the [snapshot](../../../benchmarks/snapshots/2026-06-14-metal-m4pro-ladder.json).

## Rule
- **Front-door numbers get a wins entry or they don't ship.** Two cited headlines
  had no clean source; one was stale by a known regression. Audit-then-measure
  beats carrying folklore ([[feedback_docs_are_not_truth]],
  [[feedback_bench_delta_vs_baseline_not_raw]]).
- **Identical-prompt latency probes silently measure the prefix cache.** For a
  true uncached TTFT / a clean two-point TPOT, make every request prompt unique;
  otherwise one cached point corrupts the subtraction.
- **MoE is the laptop story:** 35B at 4B-dense speed is the number that belongs on
  the front door, not the raw param count.
