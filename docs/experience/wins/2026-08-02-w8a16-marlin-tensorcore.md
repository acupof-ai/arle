# W8A16 Marlin tensor-core GEMM — bf16-class decode at half the weight VRAM — CUDA, 2026-08-02

> Status: **Shipped.** Vendored SGLang gptq_marlin (kU8B128), wired behind the
> W8A16 dispatch, SM+shape gated. Compiles sm_80→sm_120 as one binary; parity
> 18/18; c=1 decode 1.73× bf16; weights 53→30 GB. Pod-verified on sm_90 H20.

## Context

The 2026-07-31 wiring lit up the W8A16 path but on a **scalar** warp-per-row
GEMV — compute-bound on FMA cores (ncu SM 69%), 27B decode c=8 ≈ 55 tok/s vs
bf16 130. ARLE is infra: INT8 must reach its ceiling alongside FP8. The fix is a
tensor-core GEMM.

**Copy target = SGLang gptq_marlin, not Machete.** Machete is Hopper-only
(hardcoded `Sm90` + wgmma) — it would abandon the sm_120 G4 box, half the CUDA
fleet. Marlin runs sm_80→sm_120 as one binary (`__CUDA_ARCH__>=800`,
`mma.sync.m16n8k16`, no wgmma/TMA), is SGLang's actual w8a16 backend (doctrine:
kernels align to SGLang), and is zero-torch (TVM-FFI, not at::Tensor).

## Hypothesis

W8A16 upcasts int8→bf16 in registers and runs `mma` at bf16 rate = ½ fp8 rate,
so it **cannot** beat fp8 in compute-bound prefill. But decode is
latency/bandwidth-bound: time/step scales with weight bytes moved through HBM.
int8 moves 1 byte/param + a bf16 group scale — at g=128 that's 1.016 B/param vs
bf16's 2, so the weight-byte ratio is **1.94×**, not a clean 2×. Predicted:
bf16-class decode at fp8-class VRAM, better-than-fp8 accuracy. (Measured: the
1.73× win lands, but the kernel is occupancy-bound at 51 % HBM, not at the
bandwidth wall — see Learnings; the ratio holds because both lanes sit under the
same latency limit.) The win is decode latency + VRAM, not prefill throughput.

## Parameters

```bash
python3 scripts/gen_bench_prompts.py bench-agent-32k-64.jsonl 64 32768 256
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:<port> --model <27B-W8A16 | 27B-bf16> \
  --prompts-jsonl bench-agent-32k-64.jsonl \
  --concurrency-grid 1,4,8 --requests-per-concurrency 16 \
  --max-tokens 256 --seed 20260416 --timeout-seconds 900 \
  --output bench-output/<arm>/bench
```

- Binary: `arle @ 435c399f` (matched across all arms), `--qwen35-decode-graph` ON.
- Arms: W8A16-marlin, W8A16-scalar (`ARLE_W8A16_DISABLE_MARLIN=1`, bench-only
  toggle since removed), bf16 (`/host/nvme0/models/Qwen3.6-27B`).
- Deviation (stated per §3.3): 32k×256 slots doesn't fit one H20 (Qwen3.6 hybrid
  reserves 146 MB/slot recurrent state) → `--max-running-requests 8`
  `--max-total-tokens 40000`; the full 1/4/8 grid fits 8 slots, no concurrency
  dropped.
- Prompt tokens: mean 33128 (+1.0% of 32768 target). Completion: ~3980–4093/arm
  (greedy divergence across quant levels, stated).
- Trials: bf16 ran ×2 (reproduce <4%).

## Environment

- Host / GPU: H20, sm_90 (cc 9.0)
- Model: ISO-merged Qwen3.6-27B (`iso-tc-huihui`), W8A16 (gs=128, symmetric,
  scale=amax/127) vs BF16 source
- Slots 8, `mem_fraction_static=0.85`, decode graph ON (ARMED, captured — fast
  path, not eager)

## Results — decode latency (ITL p50, the clean metric)

| concurrency | marlin | scalar | bf16 | marlin vs bf16 | marlin vs scalar |
|---:|---:|---:|---:|---:|---:|
| **1** | **26.9** | 39.6 | 46.5 | **1.73× faster** | 1.47× |
| 4 | 47.1 | 88.1 | 166.4 | (confounded) | 1.87× |
| 8 | 47.4 | 134.7 | 51.8 | (confounded) | 2.84× |

**c=1 is the only clean point** — pure decode, no concurrent prefill. 26.9 vs
bf16 46.5 ms = 1.73×. Both lanes are latency-bound at c=1 (ncu: 51 % HBM, 12.5 %
occupancy — see Learnings), so the ratio tracks the ~1.94× weight-byte ratio,
not a bandwidth ceiling. c=4/8 mix decode ITL with in-flight 32k-prefill stalls
(bf16 c=4 is a non-monotone pothole, 166 ms > its own c=8 51.8 ms — a scheduler
artifact, not a GEMM property), so their ratios are not a decode measurement.

## Results — VRAM (P1: free the int8 source after repack)

Repack builds `marlin_packed`; Marlin never reads the int8 `qweight`/`qscales`
again, so they were dead resident — 27B W8A16 was ~53 GB, double intended. After
freeing them at repack (invariant: qweight freed ⟺ marlin_packed present ⟺
marlin dispatch hits; fallbacks keep qweight only when marlin_packed is None):

| | weights+context | vs bf16 | KV pool (max_total_tokens) |
|---|---:|---:|---:|
| W8A16 post-fix | **29.9 GB** | 0.58× | 829484 |
| W8A16 pre-fix | 53.2 GB | 1.03× | — |
| bf16 27B | 51.9 GB | 1.00× | 476716 |

Weights halved (int8 1 B/param vs bf16 2; the +8% over ½ is the tile layout +
bf16 group-scales). KV capacity 1.74× on the same GPU — the downstream payoff.

## Problems

- First bench used short synthetic prompts + eager decode → an inflated c=8=270
  tok/s that I first misread as "physically impossible" (wrong: decode is
  bandwidth-bound, not mma-rate-bound). The sanctioned 32k graph-on rerun
  resolved it — the win is real, cleanest at c=1. See
  [[feedback_co_evolving_oracle_cannot_catch_the_bug]].
- c=4/8 prefill-contended on a single H20 at 8 slots — c=1 is the citable point.
- P2 gate fallback (gs∉{32,64,128}) not exercised — no such checkpoint on the
  pod; it's a Rust `matches!` branch, Mac-typechecked.

## Learnings

**PASS at c=1: W8A16 Marlin decode is 1.73× bf16 (26.9 vs 46.5 ms ITL) at 0.58×
the weight VRAM, sm_90, graph-captured fast path.** The 1.73× is the weight-byte
ratio realized in decode — both lanes move weight bytes through HBM under the
same latency limit, so halving the bytes ≈ halves the step. The value prop
holds: bf16-class decode at fp8-class VRAM with better-than-fp8 accuracy. Not a
prefill/large-batch win (bf16-rate mma) — do not bench there for a win.

**We are NOT at the memory wall at c=1 — ncu roofline (H20, app clocks, m=1,
2026-08-02, `/host/marlin-bench/ncu/marlin_roofline_noclk.csv`).** The Marlin
decode GEMM achieves only **~51–53 % of peak HBM** on the large FFN shapes
(ffn_gate_up 50.8 %, ffn_down 52.9 % — ~2.1 of the H20's ~4.0 TB/s), SM ~45 %.
It is **occupancy/latency-bound**, not bandwidth- or compute-bound: warp
occupancy pinned at **12.5 %** (1 of 8 slots), a single 78-block wave (1
block/SM) that can't hide HBM latency. So an earlier draft's "at the memory wall,
matmul cannot be beaten" was **wrong** — there is ~40 points of HBM headroom,
gated by occupancy, that a higher-occupancy / split-K tiling could in principle
recover. Two facts that DO hold: (1) **byte-optimal** — DRAM traffic is 92.4 MB
for the 17408×5120 int8 weight (89.1 MB weights + scales), read exactly once, no
redundant traffic; the headroom is latency-hiding, not wasted bytes. (2) it's
SGLang/vLLM's own `gptq_marlin` (`kU8B128`) — the SOTA kernel **by provenance**,
same source those stacks ship.

**SOTA framing — what's proven vs not.** Provenance (same kernel) + byte-
optimality (weight read once) + the 1.73× recorded win are real. What is NOT
proven: (a) a physical-ceiling claim (we're at 51 % HBM, not the wall); (b) an
end-to-end win over SGLang (1.73× is vs ARLE's own bf16). A rigorous SOTA claim
needs either closing the occupancy gap toward the HBM wall, or a matched A/B vs
SGLang W8A16 (same GPU/model/weights/ctx, c=1, decode tok/s + p50/p99 ITL).

**Why W8A16 and not W4A16 (a stronger decode target).** Same Marlin kernel
family runs W4A16 at a ~3.87× byte ratio (0.5 B/param) — roughly 2× this path's
1.94×. We chose int8 for **accuracy**: symmetric per-group int8 is near-lossless
vs the int4 quality hit, and pairs with bf16 activations for a
better-than-fp8-accuracy decode. If a future workload prioritizes decode speed
over accuracy, W4A16 on the same kernel is the more aggressive lever.

Next lever (evidence-backed by the ncu run, in priority order): recover the c=1
occupancy gap (12.5 % → multi-wave / split-K, the `c_tmp` fp32-reduce buffer is
already allocated for it) — this is the real SOTA gap and needs its own measured
before/after; then matched A/B vs SGLang W8A16; sm_120 fleet-coverage run.
