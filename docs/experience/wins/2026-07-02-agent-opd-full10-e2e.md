# Agent-OPD full 10-round e2e: 438s → 90.5s per round (4.8×), numerics intact

## Context

End-to-end validation of the day's optimization series (FP8 GEMM → cuBLAS
dequant; fused prefill SDPA + wrapper deletion; partial-rotary RoPE device
kernels) on the same 10-round toy agent-OPD shape as the original baseline
run (27B Qwen3.6-FP8, share-frozen-base, 1 task, temperature 0, LoRA
attention-qv r16). Profile probes OFF — clean wall clock.

## What Worked

run-full10-opt (GPU 1, RUN_EXIT=0): **15min05s total = 90.5s/round** vs the
pre-optimization ~438s/round (**4.8×**). Loss trajectory preserved:
0.2863 → 0.3255 → 0.3293 → 0.2563 → 0.2522 → 0.2379 → 0.2409 → 0.2740 →
0.2543 → 0.2465 — same descending shape as the original (0.2888→0.2402).
VRAM stable: after-writeback 35 951→35 953 MiB across rounds (no leak);
live peak 37.7 GB / 97.9 GB.

Round anatomy (live phase trace, per round):
- writeback: **12.3s** (forward 2.2s + backward 10.0s + ce/opt ~0.1s)
- rollout turn: ~11s (single turn at temp 0; GPU util 0% for most of the
  round — decode is NOT the bottleneck)
- **round-end silent block: 60s on some rounds, 11s on others** — sits
  between the after-writeback vram print and the next round's first log
  line, runs after the FINAL round too. Code window:
  `sync_lora_from_store` into the rollout engine + loop tail
  (train_cli.rs:2401-2445; eval and adapter-save are disabled in this
  config). Not yet attributed — next probe is a timing print around
  sync_lora + one short rerun.

## Rule

- e2e wall-clock per round is the ground-truth framing for OPD optimization
  wins — kernel-level layer walls (13.5s → 3.2s etc.) undercount everything
  that happens between writebacks.
- GPU util sampling (15×2s) is the cheapest rollout-vs-sandbox attribution:
  0% util during a "rollout-dominated" phase kills the decode-bound
  hypothesis instantly.

## Addendum — full10-v2 after the LoRA-promotion fix (same day)

run-full10-v2 (16a95fe0 + e584863d): **4min38s total = 27.8s/round** —
**15.8× vs the original 438s/round**, 3.3× vs the morning's 90.5s. Loss
trajectory improved beyond every prior run: 0.2829 → 0.2737 → 0.2665 →
0.2531 → 0.2405 → 0.3220 → 0.2244 → 0.2012 → 0.2011 → **0.1740** (old code
never went below ~0.24) — consistent with the frozen-base drift bug the
promotion incidentally fixed: the student now trains against an honest
frozen base and the per-round LoRA sync compounds correctly.

Round anatomy now: rollout ~11s + writeback ~12.3s (backward 10.0s, of
which LinearAttention 4.2s = the next wall) + sync_lora 0.02s + misc ~4s.

## Addendum 2 — adaptive checkpointing (f6d11206)

run-adaptckpt-toy1r: full tape fits at seq~1010 (est ~24GB vs 60GB free) so
backward skips the recompute entirely: **backward 10.0s → 5.06s (−49%)**,
`calls=2` count 0, VRAM peak 80-85GB trims back to 36-40GB per round, loss
0.2795 in band. LinearAttention backward (4.2s) is now 83% of backward.
