# B2 CP decode head-sharding — CUDA, 2026-08-17

> Status: Shipped

## Goal

Recover the cp=2 decode regression on a 128K prompt at world=2: without B2,
cp=2 cannibalizes attn_tp (attn_tp 2→1) and decode drops to ~43 tok/s; the
target is the cp=1 (attn_tp=2) rate of ~60 tok/s.

## Hypothesis

Decode is weight-bandwidth-bound (marlin W8A16 GEMMs ≈52% of the 27B decode
step; the FA3 chain <4%). Under CP decode, treating the cp group as additional
attn_tp ranks — each rank computes 1/(attn_tp×cp) of the attention heads with
a load-time weight subset, then all-reduces the partial hidden over the global
comm — is mathematically identical to attn_tp=world decode and recovers the
regression by halving the attention weight read, not the compute.

## Parameters

- Build: `bash scripts/pod.sh build b2decode --release --features cuda,nccl --bin arle` (jobs=32, 6m37s, BUILD_EXIT=0). Source head `039618336` (B2 commit `807e6c0b4`).
- Model: `/data00/ThinkingCap-Qwen3.6-27B-FP8`, bf16 KV pool.
- world=2: `INFER_TP_SIZE=2`. Baseline `INFER_ATTN_CP_SIZE=1` (attn_tp=2); B2 `INFER_ATTN_CP_SIZE=2` (attn_tp=1, attn_cp=2).
- Correctness: `scripts/lever_gate.sh` needle ladder, `TEMPLATE=qwen3_nonthick RAW=1`, RUNS=3.
  - Gate A (non-B2 wash): `LENGTHS=115,300,446,2000,8000` — all < 8192, B2 must not engage.
  - Gate B (B2 engaged): `LENGTHS=8192,16384` — kv+1 ≥ 8192 on the first decode step.
- Perf: `scripts/decode_rate_probe.py`, 128 generated tokens, steady decode rate.
  - Gate C: prompt 130289 tokens (~128K).
  - Gate D: prompt 224324 tokens (max feasible, see Problems).

## Environment

- Host / GPU: shared H20 pod, 8×H20 (sm_90), 97 GB/GPU.
- Driver / CUDA: pod CUDA 12.8.
- Model / dtype: ThinkingCap-Qwen3.6-27B-FP8, W8A16/Marlin attention weights, bf16 KV.
- TP / CP: world=2; baseline attn_tp=2; B2 attn_tp=1 × attn_cp=2.
- Topology confirmed in serve log: `attn_tp_rank=0 attn_cp_rank=0/1 (cp comm GLOBAL)`.

## Results

### Correctness (needle ladder ×3)

| Gate | Lengths | arm | exact | DET | garbage | verdict |
|---|---|---|---:|---|---:|---|
| A wash | 115–8000 | cp=1 baseline | 3/3 | yes | 0 | — |
| A wash | 115–8000 | cp=2 (B2 off) | 3/3 | yes | 0 | within envelope |
| B engaged | 8192, 16384 | cp=1 baseline | 3/3 | yes | 0 | — |
| B engaged | 8192, 16384 | cp=2 B2 | 3/3 | yes | 0 | byte-identical to baseline |

Gate B is the critical gate: B2 engages on the first decode step (pt=8324 /
16733 → kv+1 ≥ 8192) and the cp=2 B2 outputs are byte-identical to the cp=1
baseline at both lengths, zero garbage-class outputs.

### Decode perf

| Gate | Prompt | arm | decode tok/s | cold TTFT (s) |
|---|---:|---|---:|---:|
| C | 130289 | cp=1 (attn_tp=2) | 57.72 | 63.8 |
| C | 130289 | cp=2 B2 | 59.24 | 37.0 |
| D | 224324 | cp=1 | 50.42 | 132.7 |
| D | 224324 | cp=2 B2 | 50.54 | 86.2 |

B2 recovered the 128K decode rate to the cp=1 baseline (59.24 vs 57.72; the
pre-B2 cp=2 regression was ~43 tok/s). CP also halves cold-prefill TTFT
(128K: 63.8→37.0 s; 224K: 132.7→86.2 s). At 224K the decode rate degrades
with KV length equally on both arms; B2 holds parity.

Raw artifacts (pod): `/host/arle-gates/needle_gate_{baseline_cp1,b2_cp2_wash,baseline_cp1_long,b2_cp2_engaged}.log`, `gate{C,D}_{cp1,cp2}_probe.log`, `serve_*.log`.

## Problems

- **True 256K is unreachable without a source change.** The RoPE cache is capped at `max_position_embeddings=262144`, and `max_prompt_tokens` is clamped to `max_total_tokens × 7/8` (`crates/infer-api/src/loaded.rs:478`), a hard prompt ceiling of 229376 at the 262144 RoPE limit. `--max-prompt-tokens` cannot raise it (same clamp). Gate D ran at the maximum feasible length (224324). Raising the 256K ceiling is a separate change (RoPE cap + clamp), not a B2 defect.
- `pod.sh build` rejects the `--` separator before cargo argv; dropped it.
- `decode_rate_probe.py` crashed on an empty-choices SSE chunk; fixed in the probe script (new file, not the runtime).
- The first 128K probe prompt (141237 tok) exceeded the 122500 prompt cap; recalibrated target/serving flags.

## Learnings

PASS. B2 CP decode head-shipping recovered the 128K decode regression (43→59.24
tok/s at world=2, matching the cp=1 attn_tp=2 baseline) with byte-identical
correctness on the B2-engaged needle ladder. The load-time weight subset
(W8A16/Marlin preserved, zero per-step slicing) and the full-head pool at the
natural head offset are the load-bearing choices: a per-step slice would have
cost 1.5× the weight traffic it saved, and compact-at-offset-0 would have read
rank 0's KV heads on every cp_rank≠0.

Next walls:
- T3.2 KV ownership sharding (capacity past 512K, quant-KV at 256K) — the 256K ceiling above is the first blocker.
- T3.4 CP×spec / CP×recall / CP×quant-KV combination debts.
- The `decode_recurrent_live` tail-prefill guard is defense-in-depth: ARLE has no Decoding→Prefilling in-place transition today, so the B2-live state-pollution paths the adversarial review flagged are unreachable; the sidecar restore refuses 1/cp snapshots to keep it that way.
