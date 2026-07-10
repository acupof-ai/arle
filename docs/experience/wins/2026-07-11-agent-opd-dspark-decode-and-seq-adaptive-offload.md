# Agent-OPD round: DSpark cuts serial decode −29% + seq-adaptive writeback-offload cuts backward −36% → −30% round wall (H20 GPU1 A/B)

> Status: Shipped

## Context

Three-arm A/B on H20 GPU1 attacking the profiled round breakdown
([ms/% breakdown](2026-07-11-agent-opd-round-profile-ms-breakdown.md): rollout
40% / writeback 33% / eval 23%, all serial on the engine lock). Student
`/host/Qwen3.6-27B-FP8`, corpus `staged-run1`, ROUNDS=1, task-limit 4 × 2 samples,
eval-n 4, max-turns 5, max-tokens 512, writeback-cap 4, LoRA r16 α32 qv, rollout
temp 0.7 seed 1234, eval greedy. Same binary all arms (foreign quant_linear edit
cancels in the delta). eval pass_rate=1.0000 4/4 across A/B/C — correctness intact.

- **A** = dspark OFF
- **B** = dspark ON (`--dspark-draft-model /host/Qwen3.6-27B-DFlash`), offload default ON
- **C** = dspark ON + `--writeback-offload false`

## Per-stage (total ms/round)

| stage | A: dspark OFF | B: dspark ON | C: ON+offload OFF | ΔB | ΔC |
|---|---|---|---|---|---|
| rollout_decode (8) | 73523.7 | 52134.5 | 51333.9 | **−29.1%** | **−30.2%** |
| eval (1) | 38552.8 | 26843.6 | 24717.3 | **−30.4%** | **−35.9%** |
| writeback (4) | 56890.3 | 60117.3 | 40220.4 | +5.7% (noise) | **−29.3%** |
| **round wall** | 174960.8 | 145300.7 | 122378.2 | **−17.0%** | **−30.1%** |

Writeback backward phase (avg s/call): A 11.13 · B 11.79 · **C 7.51** (−36% vs B).
forward 3.0→2.4, ce/opt negligible. Writeback +5.7% B-vs-A is run-to-run noise
(dspark is decode-only; control holds).

## What Worked

### L1 — DSpark on serial B=1 agent-OPD decode: LICENSED
- **Engagement proven** (the trainer path never calls `spec_decode_stats()`, so
  no in-process accept_rate; proven two other ways): 78 `[dspark-draft]`
  block-forward lines under `ARLE_DSPARK_PHASE`, AND a *net* decode acceleration
  (−29% rollout / −30% eval) — impossible at zero accepts (zero-accept dspark is
  strictly slower, paying draft-forward for nothing). So drafts land.
- Effective **1.41× rollout / 1.44× eval** — real but below the licensed serial
  ~1.9×; the gap is the workload accept rate on short synthetic agentic decodes.
- Decode-only: writeback (non-decode) unchanged, exactly as required.
- Already default-ON in `agent_opd_curve.sh` (agent-OPD is serial B=1,
  `agent_opd.rs:473`); no flip needed.

### L2 — eval downfreq: already default `eval_every=2`, amortizes 23%→~11.5%.

### L3 — seq-adaptive writeback grad-checkpoint offload: LICENSED, code landed
- `--writeback-offload false` cut backward **11.8→7.5s (−36%)** → writeback
  **−33% vs B**, **no OOM** at seq≈1276. Confirms the gdb hypothesis
  (errors/2026-06-28): default offload=ON pays a serialized `cuMemcpyHtoDAsync`
  grad-checkpoint re-upload that starves the GPU on short trajectories.
- **Not a blanket flag flip** — errors/2026-06-28 measured offload=OFF OOMs the
  allocator at seq≥~9600 (long forward fragments the pool). So the flag is now
  **seq-adaptive**: `runtime_flags::writeback_offload_for_seq(seq_len)` =
  `writeback_offload() && seq_len >= 4096`. Short (common) trajectories get the
  fast on-device path by default; long ones self-protect; `--writeback-offload
  false` still forces off. Threshold 4096 conservative (2.3× margin below the OOM
  anchor; nested-SDPA checkpointing `0b7a1d89` bounds inner O(seq²)). Applied at
  all three writeback variants (`opd.rs` masked / frozen-prompt-kv / GKD).

**Stacked: C = −30.1% round wall** (dspark decode + seq-adaptive offload),
quality-neutral, two levers, ~2 lines of code (the rest is default/flag).

## Rule

- **A decode-only spec win is proven by a *net* speedup, not an accept counter,
  when the in-process engine exposes no stats** — zero-accept dspark is strictly
  slower, so any net acceleration proves drafts land. Cross-check with the
  draft-forward log lines.
- **A regime-specific perf flag flips seq-adaptively in code, not as a script
  arg** — `--writeback-offload false` wins at short seq but OOMs long; a script
  default is a footgun the moment max-tokens grows. Gate on the measured quantity
  (seq_len) at the call site so every caller is correct; keep the flag as an
  explicit override.

Logs: pod `/host/aopd_dspark_ab/arm_{off,on,on_nooffload,proof}.log`.
