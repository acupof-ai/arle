# Pending-remote GPU queue

Date: 2026-09-12. Owner of the next free card starts here instead of
rediscovering the run list. Entries are derived from `docs/agenda.jsonl`,
the registry (`operators/registry.toml`), the verify scripts, and the
wins/errors entries named per row — not from memory.

This document records **stable** facts only: the exact command, the device
and why, the decision rule, the prereg row it opens, and dependencies. It
does not record which card is currently free or which run is in progress;
that state rots within the hour. Device inventory at the bottom is dated
for that reason and is the only non-stable section.

Every command assumes a pod/V100 shell with the tree on the SHA in the row.
H20 commands run in the pod tree; the release build is mandatory for any
bench-labelled run (`pod-remote-run.sh` rejects other profiles). GPU claims
go through `scripts/pick-gpu.sh` (or `ARLE_PARITY_GPU=<i>` to pin one),
never by picking a free index by hand.

## Order at a glance

1. **One sm90 card frees → parity batch (P1).** It is self-contained, runs in
   ~tens of minutes, and re-validates every gate fix and the three audit
   lanes (#380/#381/#385/#389) at once. Its per-gate logs are also the
   measured clean-run margin the Phase-2 tightenings (P11) need, so P11
   cannot start until P1 has run.
2. **Cheap single-card diagnostics, any card, short holds:** P4 fp4 M=1
   A/B, P5 DSpark c=8 branch decider, P8 decode-graph flag check. Run P5
   before interpreting any DSpark acceptance result from P3.
3. **#334 must merge, then P3 (FlashQLA/DSpark verify grid).** This is the
   multi-card consumer and the gate for P2 (the #300 varlen deletion).
   Run tp=1 first (one card, earliest signal), then tp=2/4/8 as cards free.
4. **V100 (P9) is a different host and runs concurrently with everything
   H20.** Already built to one command; it needs the V100 free and an
   explicit go decision.
5. Independent single-card serve studies — P6 splits-64 perf, P7 step1b
   gate, P10 NVML A/B, P14 thinking-needle channel (folds into P7's serve)
   — slot onto any card not running the P3 grid.
6. Multi-rank DSv4 e2e (P12) and OPD cp=2 (P13) need separate setup (model
   weights / training lane) and are scheduled on their own.

## The queue

### P1 — Registry parity batch (all gates, one sm90 card)

- **Command (pod tree):**
  `scripts/parity_gpu_batch.sh <output-dir>`
  It builds every `crates/infer-cuda/examples/*_parity.rs` listed in
  `operators/registry.toml` in one release cargo call, records
  `--kernel-build-id`, claims one free sm90 card via `pick-gpu.sh`, runs
  each gate positive and with `--negative-control`, and writes
  `results.tsv` / `results.md` plus a never-executed section. Override the
  claim with `ARLE_PARITY_GPU=<i>`.
- **Device:** exactly one sm90 (H20). Several gates hard-require sm90
  (deepgemm_grouped_prefill, both flashmla decode gates, fa3 shim); the
  batch targets sm90 as a whole.
- **Decides:** every gate prints `ALL PASS` positive and
  `NEGATIVE CONTROL OK` negative, exit 0. This run is the first GPU
  confirmation of: the five first-run gate fixes (#354 argmax/elementwise/
  marlin×2, #355 sampler, #357 dsa bf16 fixture, #359 deepgemm one-hot
  bands, #361 moe per-token bias), the negative-control teeth (#380 fa3/
  paged per-row + never-executed summary, #381 moe m_indices + sibling
  specificity, #382 flashmla pack oracle, #385 prefill index-builder
  tooth), and the new marlin_w8a16 absolute cap (#389).
- **Expected SKIP (named, not failure):** `fa2_sm70` (needs sm_70, runs on
  the V100 — P9), `dsv4_parity` (needs the multi-rank model launcher —
  P12).
- **Prereg:** the script opens/closes one `parity-gpu-batch-<ts>` row.
- **Depends on:** nothing (run first).
- **Unblocks:** P11 (its per-gate metric logs are the measured
  clean-run margins); contributes the `gdr_varlen_parity` four-shard PASS
  that P2 needs.

### P2 — Delete varlen GDR (#300, draft)

- **State:** draft PR #300, branch `lane/gdr-varlen-removal`
  (`d0a375a9d`), premise "one chunked FlashQLA path at every TP", net
  -727 plus deleting the varlen half of `gdr_varlen_parity.rs` (keeps the
  FQ-vs-f64 anchor). Not merged.
- **Command (after P3's script is on main):**
  `GDR_CHUNKED=0 scripts/dspark_flashqla_verify.sh <pre-#300-sha> <#300-head> <out-dir>`
  The flag appends `--qwen35-gdr-chunked false` identically to both arms
  (there is no env var; the old `ARLE_QWEN35_GDR_CHUNKED` is deleted), and
  the probe asserts the `linear/gdr_fq` op count is 0 — a misspelled switch
  fails rather than passing as a no-op.
- **Device:** sm90 cards per the tp grid (1/2/4/8).
- **Decides (merge gate, from agenda dspark-accept-collapse):**
  flashqla_chunk_parity at (16,48)/(8,24)/(4,12)/(2,6); TP2/4/8 needle ×3
  + lever; DSpark c=8 verify ITL and acceptance base-vs-treatment. If the
  A/B attributes the c=8 collapse to varlen, deletion is justified on
  correctness; if paths are numerically equal, #300 stands on simplicity.
  `gdr_varlen_parity` must PASS on the batch (P1) first.
- **Prereg:** per arm, opened by the verify script.
- **Depends on:** P3's script merged (#334), P3 green, P1 gdr_varlen PASS.

### P3 — FlashQLA / DSpark runtime verify across attn_tp 1–8 (#334)

- **State:** PR #334 OPEN, branch `lane/dspark-verify-batch`; the script
  `scripts/dspark_flashqla_verify.sh` is **not on main yet** — merge is a
  prerequisite.
- **Command:** `scripts/dspark_flashqla_verify.sh <base-sha> <treatment-sha> <out-dir>`
  Builds both SHAs `--release -p cli` in detached worktrees; per arm ×
  attn_tp boots Qwen3.8-27B-DSpark and runs (a) `lever_gate.sh`
  (`GATE_PROFILE=generic`, needle ladder ×3 + concurrent arm; base seeds
  the envelope with `LEVER_GATE_ALLOW_NO_BASELINE=1`), (b) matched TTFT/
  prefill via `bench_throughput.py` c=1 on 8K prompts, (c) DSpark
  acceptance c=1 and c=8 via `bench_dspark_accept.py --concurrency 8` (one
  global stats pair per c), plus a launch row and a separate
  `ARLE_CUDA_PROFILE=1` probe reading the `linear/gdr_fq` vs
  `linear/gdr_recurrent` op.
- **Device:** sm90, claimed through `pick-gpu.sh reserve-set`; the grid is
  attn_tp 1/2/4/8 and an attn_tp with too few free cards is recorded SKIP.
  tp=8 consumes all eight cards.
- **Decides:** #327/#329/#331 correctness (needle/launch/GDR-path are hard
  gates, exit non-zero on failure); TTFT and c=1/c=8 acceptance are
  recorded measurements, not gates.
- **Prereg:** `dspark-flashqla-verify-<ts>-{base,treatment}` per arm.
- **Depends on:** #334 merge. Run P5 first so a scheduling-driven 0% is
  distinguished from a verify-side defect before reading the acceptance
  arm. **Unblocks:** P2.
- **Concurrency:** the only multi-card job; tp=8 is whole-box. tp=1/2
  leave cards for other single-card jobs but every card runs a 27B serve,
  so keep one serve per card.

### P4 — fp4 M=1 GEMV vs Marlin matched A/B (w3 / marlin-m1-gemv)

- **Commands (pod):**
  `bash scripts/pod.sh build kab --release --features cuda --bin arle`
  then `bash scripts/kernel_ab_fp4.sh kab`
  Direct form: `arle kernel {fp4-gemv,marlin-fp4-gemm} --shape 1,34816,5120 --ref cpu --iters 100`.
- **Device:** one sm90 card, short hold (~100 iters + CPU ref).
- **Decides:** both kernel lines print `time_ms` + `max_rel` and PASS
  (`PASS_MAX_REL=1e-2`); the trailing verdict compares time@M=1 to settle
  whether the zero-production-caller dense fp4 GEMV beats tensor-core
  Marlin at decode shape, which closes `marlin-m1-gemv` and w3.
- **Prereg:** one `fp4-m1-ab-<ts>` row (command + keep/remove decision).
- **Depends on:** nothing. Filler job for a briefly free card.

### P5 — DSpark c=8 acceptance branch decider (#386)

- **Command (self-contained, spawns its own serve):**
  `python3 scripts/dspark_c8_accept_branch.py --port 8000 --concurrency 8`
  Snapshots `/v1/stats` `spec_decode` counters across c=1 and c=8 windows.
- **Device:** one sm90 card for the Qwen3.8-27B-DSpark serve.
- **Decides exactly one branch, no judgement call:** chains=0 →
  NOT-SEEDING (DSpark not engaged, e.g. routed Plain on quant KV);
  chains>0 drafted=0 → BUDGET-ZERO-KEEPS (the confidence goodput bar
  rises 7.86× c=1→c=8, 0.002506→0.019699, truncating every chain to its
  anchor); drafted>0 accepted=0 → PROPOSED-AND-REJECTED (scheduling
  exonerated, verify-side hypotheses H1/H2/H3 survive). Garbage counters
  are a hard error, never a branch.
- **Prereg:** the script self-opens `dspark-c8-accept-branch-<ts>`.
- **Depends on:** nothing; **run before P3's acceptance arm is
  interpreted.** This is the cheapest discriminator for
  `dspark-accept-collapse`.

### P6 — Quantized-decode split ceiling 16→64 perf license

- **State:** code complete (`kMaxSplits` 64 pinned at the three sites),
  host test green; the wall-clock license is unmeasured.
- **Command:** against an fp8 serve (`--kv-cache-dtype fp8`, otherwise
  default):
  `python3 scripts/bench_throughput.py --url http://127.0.0.1:8000 --model Qwen3.8-27B --prompts-jsonl bench-agent-32k-16x8.jsonl --concurrency-grid 1,8 --requests-per-concurrency 16 --max-tokens 214 --seed 20260416 --timeout-seconds 900 --output bench-output/quant-splits-64/bench`
  Baseline = archived pre-change binary, treatment = current, ≥3 trials
  per arm, matched/simultaneous. Unit gate on a CUDA host:
  `cargo test -p infer-cuda --release --features cuda quant_decode_num_splits`.
- **Device:** one sm90 card, long hold (decode A/B).
- **Decides:** whether the 16→~20 selected splits at B=1 convert to a
  measured decode gain over the baseline envelope; the tileRL direction
  (+1.7%/+1.1%) is a prior, not the license. No gain → no perf license to
  raise the ceiling.
- **Prereg:** `quant-splits-64-ab-<ts>`.
- **Depends on:** nothing; independent single-card study.

### P7 — Host KV-pool single-writer (Step 1b) remote gate

- **State:** #276 merged (`bfe18035b`); device gate never run.
- **Commands (against a spec-capable CUDA serve):**
  1. `scripts/needle_gate.py` ×3 same-config, spec arm vs baseline envelope;
  2. `scripts/spec_parity.py --model <cuda-model> --draft-model <draft>` —
     token-id exact equality of speculative vs greedy arms (zero mismatch);
  3. one MTP c-sweep confirming accept counts and decode ms/token match the
     2026-09-10 batched-MTP acceptance entry.
- **Device:** one sm90 card with the MTP/DSpark draft weights loaded.
- **Decides:** host pool length stays equal to device truth across spec
  (a divergence fails needle and token parity loud).
- **Prereg:** `step1b-remote-gate-<ts>`.
- **Depends on:** nothing beyond model weights; naturally shares a serve
  window with P3/P5.

### P8 — Decode-graph DeepGEMM min-routes flag validation (#273)

- **State:** #273 merged.
- **Command:** serve an MoE model twice and read graph-capture behaviour:
  once `--qwen35-deepgemm-min-routes 64` (below the route threshold, forces
  a non-graphable dispatch) and once at default 1024; confirm the decode
  gate reads the same flag and graph capture reacts as documented.
- **Device:** one sm90 card; observation during a normal serve, cheap.
- **Decides:** the gate guarding a flag-tunable dispatch actually reads
  that flag's runtime value at both settings.
- **Prereg:** `deepgemm-minroutes-flag-<ts>` (or fold into a serve
  prereg). Filler observation; pair with any MoE serve window.

### P9 — FA2 sm70 parity on V100

- **State:** binary built to one command on `ssh v100`
  (`~/fa2sm70-scratch`, tree at main `b7a82035f`, release sm_70,
  build-id `bundle:3414c4aa…`). Not executed. Env: CUDA 12.4 +
  gcc-11 host + `INFER_TILELANG_PYTHON=/usr/bin/python3` (tilelang 0.1.13)
  in `~/fa2sm70-scratch/env.sh`.
- **Command:** `bash ~/fa2sm70-scratch/run.sh` (occupancy-guarded; sets
  `INFER_CUDA_DEVICE=0`). Negative: the same binary with
  `--negative-control`.
- **Device:** the V100 (Tesla V100-SXM2-32GB, cc 7.0) — a different host
  from the H20 pod; runs concurrently with H20 work.
- **Decides:** first-ever execution on any device (H20 SKIPs forever):
  clean `FA2 sm70 ALL PASS` rc0 within the existing bound
  (rel 4e-2/slope 6e-2/floor 2e-2/viol 1e-3), negative
  `NEGATIVE CONTROL OK (all)` rc0. A failing rel_l2 is data for the
  Phase-2 tolerance work, **not** a reason to widen the bound; bounds must
  not be edited to force a pass.
- **Prereg:** `fa2-sm70-firstrun-<ts>`.
- **Depends on:** the card being free and an explicit go decision.

### P10 — In-process NVML sampler on/off A/B (#388)

- **State:** #388 merged; sampler default-off (`ARLE_OBSERVE_GPU=1`).
- **Command:** matched A/B of DSpark decode c>=2 on the pod, sampler off
  vs on, simultaneous or interleaved matched pairs; primary metric decode
  p99 inter-token latency.
  `scripts/prereg.py start --name nvml-observe-ab --cmd "DSpark decode c>=2 matched A/B, ARLE_OBSERVE_GPU off vs on, 8xH20" --hypothesis "10s in-process NVML sampling moves decode p99 ITL by less than the ~10% matched-A/B noise envelope; the bbd422973 stall does not recur"`
- **Device:** one sm90 card running the DSpark c>=2 serve (the same serve
  shape P3/P5 use).
- **Decides:** on-arm p99 delta inside the ~10% matched-noise envelope →
  default-on; a regression keeps it default-off.
- **Prereg:** `nvml-observe-ab`.
- **Depends on:** nothing; prefer a dedicated matched pair rather than
  wrapping the profiler-heavy P3 serves.

### P11 — Phase-2 parity tolerance tightenings

- **State:** derivations landed (#389); three bounds are flagged loose
  enough to admit a plausible defect and tighten only where a derivation
  AND a measured clean-run margin agree.
- **Bounds:** fa3_hd256_shim `TOL_FP8_ANCHOR` rel 0.22;
  dsv4_tp_oproj `DG_SLOPE` 0.14 (above even the two-operand 0.125
  worst-bin sum); dspark_sampler `PROB_REL_L2_MAX` 0.02 (~20× its floor).
- **Command/process:** read the gate's clean-run metric from the P1
  `results.tsv`/per-run logs, set each bound between the measured margin
  and the derived supremum, re-run that one gate positive + negative on a
  card, and write the derivation + measured margin next to the constant.
  Separate lane/PR per bound (or one PR if all three re-runs are green).
- **Device:** one sm90 card, short per-gate reruns.
- **Decides:** a tightened bound the clean run still passes and the
  negative arm still trips; if the clean margin already sits at the
  supremum, leave the bound and record why.
- **Depends on:** P1 (measured margins).

### P12 — DSv4 multi-rank e2e parity (the never-run registry gate)

- **Command:** set `INFER_DSV4_MODEL_PATH` plus
  `INFER_CUDA_DEVICES` / `INFER_TP_RANK` / `INFER_TP_SIZE` and the
  `INFER_NCCL_ID_FILE` bootstrap documented in
  `crates/infer-cuda/examples/dsv4_parity.rs:23-37`; every rank runs the
  identical single-row DSv4 forward and the launcher compares rank 0's
  `clean_tokens` against the fixed echo oracle.
- **Device:** N sm90 cards (the TP world) and the DSv4 FP8 weights; not
  part of the one-card parity batch (it SKIPs there).
- **Decides:** the model-launcher path echoes the fixed sequence across
  ranks. Needs its own one-command launcher prep (same shape as the V100
  prep) before it can run.
- **Prereg:** `dsv4-multirank-parity-<ts>`.
- **Depends on:** launcher/weights setup; independent of P1–P11 logic.

### P13 — Agent-OPD cp=2 matched rollout A/B

- **State:** agenda `opd-cp2-rollout-ab` (active); closes the pending item
  in `wins/2026-08-07-agent-opd-rollout-fleet` and
  `errors/2026-08-07-agent-opd-cp2-rollout-divergence-deadlock`. This is
  the OPD/training lane, not inference serving.
- **Command:** same binary/subset16 manifest, cp=2 on two H20
  (ThinkingCap-Qwen3.6-27B-FP8), one config change per arm, 3 rounds;
  correctness = `RUN_EXIT=0`, both serve pids in the shared dump, loss
  values in family with the fulltrain trajectories. Run through the agent
  harness (the deterministic `--synthetic-writeback-seq` path is the
  control; stochastic rollouts must not be compared across ranks).
- **Device:** two sm90 cards reserved for the training lane.
- **Decides:** the rank-0-keeps-the-lane fleet form matches single-GPU
  rollout at cp=2 without the divergence/deadlock.
- **Prereg:** `opd-cp2-rollout-ab-<ts>`.
- **Depends on:** the OPD lane owner and a two-card reservation; outside
  the inference parity queue.

### P14 — Thinking-model needle channel: budget artifact vs serving bug vs widen

- **State:** diagnostic mechanism merged, criterion deliberately unchanged;
  decision pending-remote per
  `docs/plans/2026-09-12-reasoning-content-criterion.md`.
- **Commands (one ThinkingCap serve with `--max-thinking-tokens 512`):**
  1. `NEEDLE_MAX_TOKENS=16 python3 scripts/needle_gate.py 115,241,446 3 0.0`
     — reproduce `NEEDLE_REASONING_ONLY` at the current gate budget;
  2. `NEEDLE_MAX_TOKENS=512 ... needle_gate.py 115,241,446 3 0.0` — does a
     completed answer reach `content`;
  3. `RAW=1 TEMPLATE=qwen3_nonthink ...` same ladder — retrieval control;
  4. archive the raw JSON of one A and one B response (`content`,
     `reasoning_content`, `finish_reason`, `completion_tokens`).
- **Device:** one sm90 card, one ThinkingCap serve window; folds into P7
  (same model class, same ladder shape).
- **Decides:** leading hypothesis is the 16-token gate budget, not the
  criterion: B puts the needle in `content` with `finish_reason=stop` →
  budget guidance, gate unchanged. Empty content with needle in reasoning
  under adequate budget + `finish_reason=stop` → serving-splitter bug (fix
  `split_reasoning`/SSE lockstep) before ever widening; only if a reference
  serve also returns empty content and real callers read reasoning does the
  greedy arm widen. The full decision table is the criterion plan's table.
- **Prereg:** `needle-reasoning-channel-<ts>`.
- **Depends on:** nothing; shares a serve window with P7.

## Not in this queue (different blocking resource or stale)

- **ttft-35b (blocked):** needs a Mac without swap pressure, not a GPU.
  Stays out of the card queue.
- **metal-test-sigsegv (active):** macOS autograd/train signal 11 on the
  Metal toolchain; debugged on Mac, no card.
- **Older July pending-remote entries** (cuda-canonical-tp L4 match,
  opd-engine-knobs, opd-gkd-anchor): no open agenda row and no active
  owner; treated as stale. Revive explicitly if the owner surfaces one.
- **quant-linear storage validation** (`ops/quant_linear.rs`
  `storage_states`): the table test is host-only and runs on Mac; the
  device half (`validate_storage` at load, `loader.rs:2853`) is exercised
  by the first FP8 model load on the pod, not by a dedicated command —
  observe it during P3/P6/P7, do not schedule a standalone run.

## Concurrency map (if more than one card frees)

| Job | Cards | Hold | Runs alongside |
|-----|-------|------|----------------|
| P1 parity batch | 1 sm90 | medium | anything on other cards |
| P4 fp4 A/B | 1 sm90 | short | other 1-card jobs |
| P5 c=8 branch | 1 sm90 | one serve window | other cards |
| P6 splits A/B | 1 sm90 | long | other cards |
| P7 step1b gate | 1 sm90 | one serve window | other cards |
| P8 flag check | 1 sm90 | folds into a serve | n/a |
| P14 thinking channel | 1 sm90 | folds into the P7 serve | n/a |
| P10 NVML A/B | 1 sm90 | matched pair | other cards (not the P3 serves) |
| P3 verify tp=1/2/4 | 1/2/4 sm90 | long | 1-card jobs on leftover cards, one serve per card |
| P3 verify tp=8 / P2 | 8 sm90 | whole box | nothing else |
| P12 dsv4 multirank | N sm90 | multi-rank | its own reservation |
| P13 OPD cp2 | 2 sm90 | training reservation | its own lane |
| P9 fa2 sm70 | 1 V100 (separate host) | short | **all H20 jobs concurrently** |

Rule: never put two model serves on one card (each pins a 27B FP8 working
set). Parallelize across cards; serialize serve jobs on a card. P3 at
tp=8 and P2 are the only whole-box jobs and must be the sole claimant.
