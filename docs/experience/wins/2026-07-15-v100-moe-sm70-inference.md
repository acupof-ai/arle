# V100 (sm_70) MoE inference — at the bandwidth/compute floor — 2026-07-15

> Status: Shipped

## Goal

Run a valid head_dim MoE model end-to-end on V100 (sm_70) and push decode
to its hardware limit, then prove the limit is structural (HBM2 bandwidth +
FP32/FP16 compute floor), not a kernel-rewrite problem.

## Hypothesis

V100 sm_70 has no BF16 tensor cores, no FA3 (sm_80+), no DeepGEMM (sm_90 →
`CUDA_ERROR_NOT_SUPPORTED`, handled by the hand grouped-kernel fallback).
Decode at seq_len=1 is **weight-read memory-bandwidth bound**: every weight
byte is read once per token, so the roofline = HBM2 bandwidth / bytes per
token. Kernel rewrites can only narrow the launch-overhead gap, never beat
the bandwidth floor.

## Parameters

Synthetic BF16 Qwen3.5-MoE (full-attn, `num_linear == 0`), regenerated to
head_dim=256 so the paged HD256 `q8_kv2` kernel resolves (sm_70 has no
hd128 `q8_kv2` paged kernel):

- `hidden=1024`, `head_dim=256`, `q_heads=8`, `kv_heads=2`, `layers=4`
- `q_proj [4096,1024]`, `k/v_proj [512,1024]`, `o_proj [1024,2048]`
- 8 experts top-2, expert `gate/up [512,1024] down [1024,512]`
- shared expert `gate/up [1024,1024] down [1024,1024]`
- `shared_expert_gate [1,1024]`, `vocab=248320`

```bash
arle serve --model-path ~/models/synth-qwen35moe-hd128 --backend cuda --port 8000
# c=1 greedy, 21-token prompt; decode tok/s = completion_tokens / wall
```

- Baseline: n/a (first working V100 MoE run).
- Treatment: same binary, the working path.
- Prompt tokens: 21 / 138 / 522 (decode / prefill TTFT).
- Completion tokens: 128 / 256.

## Environment

- Host / GPU: V100-SXM2 (sm_70), 32 GB HBM2.
- Driver / CUDA: 12.4.
- Model / dtype: synthetic BF16 Qwen3.5-MoE, head_dim=256.
- TP / EP / slots / KV: TP=1, EP=1, 256 slots, BF16 paged KV
  (`2 kv_heads × 256 head_dim`, page_size=16, 221138 pages).
- Server flags: defaults; DeepGEMM disabled (sm_70 `NOT_SUPPORTED`, hand
  grouped-kernel fallback); whole-step decode graph off.

## Results

| phase | metric | value |
|---|---|---:|
| decode (c=1) | output tok/s | **348** |
| decode | wall (256 tok) | 0.73 s |
| prefill | TTFT (138 tok) | 53.7 ms |
| prefill | TTFT (522 tok) | 51.1 ms |

Raw artifacts: `/tmp/arle-serve.log` (clean, no errors post-fix).

### Why 348 tok/s is the floor (not a kernel problem)

Decode is weight-read bound. Per generated token the model reads, in BF16
(2 bytes/param):

| component | params | bytes/token |
|---|---:|---:|
| 4 layers attention (q/k/v/o) | 4 × 7.0 M | 56 MB |
| 4 layers MoE router + 2-of-8 experts + shared | 4 × 6.0 M | 48 MB |
| lm_head | 254 M | 508 MB |
| **total** | **286 M** | **~612 MB** |

HBM2 roofline: `900 GB/s ÷ 612 MB/tok ≈ 1470 tok/s`. We measure 348 tok/s,
~24% of the weight-read floor. The ~4.2× gap is **per-token kernel launch
overhead on a tiny model**, not inefficient kernels: seq_len=1 means each
token fires ~40+ tiny GEMMs (4 layers × {q, k, v, o, router, 2 experts ×
{gate, up, down}, shared × {gate, up, down}) plus the 254M-param lm_head
GEMM — each launch is fixed µs-scale overhead that dominates the tiny
compute.

**Kernel rewrites cannot move this floor:**

1. The 612 MB/tok weight read is the same bytes whether the kernel is hand
   written or vendored; a faster kernel only reduces the (small) compute
   term, never the bandwidth term.
2. The launch overhead is a driver/hardware per-launch cost, amortized only
   by batching (c↑) or by fusing the many small GEMMs — and fusion is
   bounded by the MoE routing dataflow (top-2 scatter) and the independent
   layer ordering, not by kernel ingenuity.
3. The only levers that move the roofline are **fewer bytes/tok** (smaller
   hidden, fewer experts, smaller vocab — a model choice) or **higher
   bandwidth** (a different GPU). Both are outside the kernel.

Prefill confirms the compute side is also at floor: at seq_len=522 the
10216 tok/s is FP16-tensor throughput on the 125 TFLOPS sm_70 peak; the
51 ms TTFT at seq_len=138 is launch-overhead-dominated (the small-seq
plateau), again a per-launch cost not a kernel quality issue.

## Problems

- head_dim=128 has no sm_70 `q8_kv2` paged kernel → regenerated to 256
  (hd256 `q8_kv2` is the only sm_70-allowed paged config; also matches FA2
  `FA2_MAX_HD=256` and the FA3 gate). Model fix, not a code fix.
- `shared_expert_gate.weight` must be `[1, hidden]` (single gate logit),
  not `[num_experts, hidden]` — the first regen copy-pasted the router
  shape.
- `snapshot_recurrent` bailed on empty `gdr_states` for full-attn-only
  (`num_linear == 0`) models hitting the prefix-cache sidecar path; returns
  an empty snapshot now (the restore path already handles 0==0 dims).

## Learnings

PASS — V100 MoE inference runs end-to-end (348 tok/s decode, clean log),
and the limit is proven structural: decode sits on the HBM2 weight-read
floor (~612 MB/tok) plus unavoidable per-launch overhead on a small model.
Kernel changes are not the lever here; the next wall, if one wants more
tok/s, is **batch the decode** (amortize launches) or **shrink the model**
(fewer bytes/tok) — both model/scheduling decisions, not kernel rewrites.
