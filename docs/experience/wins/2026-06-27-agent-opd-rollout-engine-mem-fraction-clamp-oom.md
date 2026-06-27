# agent-OPD full-loop OOM at HEAD is a real VRAM over-reservation (mem_fraction_static clamped to 0.5 min), NOT ELKEID — fix: lower the clamp floor so the co-resident rollout engine honors its 0.2 request

`pending-remote` (tn/Kerberos tunnel to the H20 box dropped before the
post-fix re-run; ticket renewal needs interactive corp SSO). Root cause is
**measured** (RUST_LOG=info + ARLE_OPD_VRAM_TRACE on the pod); the fix is a
1-file seam-constant change, unit-tested green locally.

## Context

Mainline task: run the agent-OPD full loop at HEAD (`0eab9345`: borrow fix
`0ca638a7` −5.54 GB + held-out eval `7dff5484` + loop-close + fast CE + capstone)
on a free H20 (GPU 4), with the 3-task held-out eval, and observe whether it
survives the suspected ELKEID HIDS kill. Synced HEAD's 8 changed source files
(tarball over the pod's `65a46817` content) → built clean (`BUILD_EXIT=0`, 2m41s
incremental) → binary verified to carry the `agent-opd` eval flags.

## Root Cause (measured — overturns the prior "node-governance SIGKILL" hypothesis)

At HEAD the run does **NOT** die by a silent SIGKILL. It exits **cleanly with
`RUN_EXIT=1` and an explicit `CUDA_ERROR_OUT_OF_MEMORY`** on the student's first
big bf16 tensor (`embed_tokens` `[248320, 5120]` = 2.54 GB), GPU freed to 3 MiB
after. 3 identical reproductions. This is a deterministic, attributable OOM — the
earlier `errors/2026-06-27-...node-governance...` entry's structural conclusion
was the case-as-fact trap (an aggregate "it's governance" that the decoded HEAD
run falsifies).

VRAM ledger at the OOM point (`ARLE_OPD_VRAM_TRACE=1`, `RUST_LOG=info`):
- post-weights, pre-pool free: **68791 MB** (27B FP8 weights ≈ 28 GB resident).
- full-attn KV pool profiled: **`mem_fraction_static 0.2` → 318246 tokens
  (19890 pages) ≈ 20.8 GB**, leaving only **2229 MB free** → student OOMs.

The bug: `infer_seam::clamp_mem_fraction_static` clamped the requested fraction to
a **0.5 minimum** (`MEM_FRACTION_STATIC_MIN`). The OPD rollout engine asks for
`0.2` (train_cli.rs:2089) to stay small alongside the co-resident trainable
student, but it was silently raised to `0.5`, so the pool reserved
`free − 0.5×total` ≈ 20 GB regardless. The `--share-frozen-base` borrow (−5.54 GB,
working: "borrowing 400 resident FP8 base projections") was not enough headroom
against the 0.5-floored pool.

## What Worked

`crates/infer-seam/src/resource.rs`: lower `MEM_FRACTION_STATIC_MIN` `0.5 → 0.05`.
`PROFILE_KV_TOKENS_FLOOR` (4096 tokens) already guarantees the pool never collapses
to zero, so the floor — not a 0.5 clamp — protects admission at the bottom. With
`0.2` now honored: reserve = `0.8×97.5 GB` = 78 GB > free → pool floors to 4096
tokens ≈ 0.27 GB → ~20 GB freed → free rises 2.2 GB → ~22 GB, ample for the
student's ~6-8 GB bf16 materialization (embed + lm_head + norms + LoRA). Only
sub-0.5 caller in the tree is OPD (intended); every serve path passes 0.9/default,
unaffected. Two unit tests updated (`clamp_mem_fraction_static_band`,
`profile_kv_pool_tokens_clamps_fraction`); `cargo test -p infer-seam` 18/18 green.

## Pending-remote

Re-run on the H20 once the tunnel is back:
`INFER_CUDA_DEVICE=4 ARLE_OPD_VRAM_TRACE=1 arle train agent-opd
--student-model /host/Qwen3.6-27B-FP8 --dataset /root/agent_opd_task.jsonl
--staged-root /root/staged --eval-dataset /root/agent_opd_eval.jsonl
--eval-staged-root /root/eval_staged --eval-every 1 --rounds 2
--samples-per-prompt 2 --max-turns 24 --max-tokens 1024 --lora-layer-start 32
--rollout-num-slots 1 --save-lora-adapters /root/agentopd_value --pythonpath lib:test`.
Expect: student load completes (free ~22 GB), round-0 baseline + per-round held-out
pass-rate logged. Report the pass-rate trend (baseline → round-N, ±Δ). If a
genuine ELKEID kill then appears (silent, no RUN_EXIT), the setsid-session probe
at `crates/train/src/sandbox.rs:60` (new session vs the current `.process_group(0)`)
is the next step — but the HEAD blocker is this OOM, not ELKEID.

## Rule

A "killed" OPD loop is a case to decode at the exit-signal level before
generalizing: a clean `RUN_EXIT` + `CUDA_ERROR_OUT_OF_MEMORY` is a VRAM-budget bug
(attributable, fixable), NOT the silent external SIGKILL that "node governance /
ELKEID" implies. Decode the actual exit before trusting a prior structural verdict.
And a `mem_fraction_static` request that silently clamps to a higher floor is a
self-deception: verify the *effective* fraction in the profiling log, not the
requested one.
