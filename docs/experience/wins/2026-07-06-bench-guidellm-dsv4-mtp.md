# DSv4 TP=8/EP=8 MTP speculative-decode bench — guidellm sweep, BLOCKED by GPU1 contention, 2026-07-06

> Status: **BLOCKED — no bench executed.** Written per the "no silent skip" bench
> discipline: code/build/flag verification is complete and SOLID; the actual
> `scripts/bench_guidellm.sh` sweep could not run this session because DSv4
> TP=8/EP=8 requires all 8 physical H20s and GPU1 was held by a different,
> long-lived concurrent session the entire session. No numbers below are
> fabricated — every unmeasured cell says so explicitly.

## SLO-shape probed?  N

Not run — see Blocker.

## Roofline check

Not run — see Blocker. Deferred, not a KILL: no measurement was attempted, so
there is no roofline number to be below-threshold on.

## Goal

Measure TTFT / TPOT(ITL) / output tok/s for DeepSeek-V4-Flash-FP8 served at
TP=8/EP=8 on the 8×H20 pod with MTP (checkpoint-native NextN speculative
decode, `--spec-type mtp`) enabled, via the canonical
`scripts/bench_guidellm.sh` sweep, to quantify the MTP win over the
non-speculative baseline.

## Hypothesis

MTP draft-verify (default depth `DEFAULT_MTP_DRAFT_TOKENS=2`, topk
`DEFAULT_MTP_DRAFT_TOPK=1`, exact-greedy acceptance) should lower per-token
decode latency (ITL) and raise output tok/s whenever the draft head's greedy
token matches the trunk's verify token, at the cost of the extra draft-head
forward pass on a miss. Directionally a win on DSv4 given the same NextN
head/verify design already measured correct on Qwen3.6
(2026-06-06 EAGLE/MTP phase2 wins).

## Command (verified to construct correctly; not executed against a live server)

```bash
# Serve (TP=8/EP=8, MTP on, all 8 physical GPUs):
CUDA_HOME=/usr/local/cuda \
  INFER_TP_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 INFER_DSV4_MAX_SEQ_LEN=16384 \
  /host/arle-build/target/release/arle serve --backend cuda \
    --model-path /host/DeepSeek-V4-Flash-FP8 --bind 0.0.0.0 --port 18195 \
    --spec-type mtp

# Bench (canonical, locked params — docs/plans/guidellm-integration.md §3):
scripts/bench_guidellm.sh dsv4-mtp-full \
  --target http://localhost:18195 \
  --model DeepSeek-V4-Flash-FP8 \
  --processor /host/DeepSeek-V4-Flash-FP8
```

## Environment

- **Backend:** CUDA, H20 ×8 (97871 MiB/card), CUDA 12.9 (per prior pod sessions).
- **Model:** DeepSeek-V4-Flash-FP8, `/host/DeepSeek-V4-Flash-FP8`, checkpoint
  `num_nextn_predict_layers=1`, `num_hidden_layers=43` (confirmed via
  `config.json` on the pod — the checkpoint does ship an MTP draft head).
- **Commit:** `f22ad1ff0` (`fix(scheduler): poll log-file content instead of
  just its presence`), which sits on top of the 7-commit push this task was
  commissioned to verify (`5fd6a8984..7f6e87fca`, InFlightGuard cancellation
  propagation + `Engine::cancel_request`). Pod tree was re-synced to
  `f22ad1ff0` mid-session after a second commit landed concurrently — see
  Notes.
- **Feature set:** `cargo build --release --features cuda,nccl,deepep --bin arle`.
  Native DeepEp sidecar **not** built this session (`ARLE_DEEPEP_DIR` unset →
  build warns `skipping arle_deepep_sidecar build`); MoE EP dispatch would run
  the allreduce/naive fallback, not the native DeepEP backend. Perf-only
  caveat, not a correctness gap.
- **Non-default flags:** `--spec-type mtp` (defaults: `mtp_draft_tokens=2`,
  `mtp_draft_topk=1`); `INFER_TP_SIZE=8`; `INFER_DSV4_MAX_SEQ_LEN=16384`.
- **Profiling state:** N/A — no run.
- **Server launch:** not started (see Blocker).

## Canonical params (locked, unused this session)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256` (+ stdev/min/max clamps)
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| — | **NOT RUN** | **NOT RUN** | **NOT RUN** | **NOT RUN** | **NOT RUN** | **NOT RUN** |

## Results — service-side KV / scheduler metrics

Not collected — no run.

## Results — request accounting

Not collected — no run.

## Problems

- **Root blocker: GPU1 held by a different concurrent session for the entire
  window of this task.** `nvidia-smi` showed GPU1 at a steady 50281 MiB the
  whole session (GPUs 0,2–7 idle at ~0 MiB). Host-side (`tn exec ps`, correct
  namespace per `reference_h20_pod_pid_namespace_gpu_trap.md`) identified the
  occupant: PID 2094648, `./target/release-fast/arle serve --model-path
  /host/Qwen3.6-27B-FP8 --bind 0.0.0.0 --port 18200 --max-running-requests 4
  --dump-messages-dir /host/tb_dumps`, **started Sun Jul 5 23:00:29 2026**,
  alive 12h40m+ at last check — a standing serve backing a
  `terminal-bench-core` harness (`tb run -d terminal-bench-core==0.1.1 -a
  claude-code -m anthropic/Qwen3.6-27B-FP8 -t hello-world ...`) from an
  unrelated, concurrent Claude/Codex session on this shared box. It actively
  fielded a harness task during this session (started ~03:19 UTC, exited
  ~03:31 UTC) and the serve process stayed up afterward — a genuinely
  standing eval backend, not a stale zombie.
- **DSv4 TP=8/EP=8 has zero tolerance for a busy GPU in the 0–7 range.** EP
  mirrors TP (`ep_size = world_size`); `ExpertSplit::new`
  (`crates/infer-cuda/src/moe_config.rs:59-68`) hard-errors unless
  `num_experts (256) % ep_size == 0`. Among ≤8 workers only 1/2/4/8 divide 256
  evenly, so there is no 7-GPU subset (excluding GPU1) that still satisfies
  "TP=8/EP=8" — this isn't a policy choice, it's enforced by an `ensure!` at
  load time.
- **Did not kill the occupant.** Per project rule (kill only your own
  processes; never blind-kill a foreign, actively-used PID), PID 2094648 was
  left alone. A prior wins entry
  ([2026-07-05 P1/P2/P4 needle gate](2026-07-05-dsv4-p1-p2-p4-needle-gate.md))
  independently excluded GPU1 for the same reason ("foreign process holding
  VRAM") — this is now a **≥2-day recurring contention specifically on GPU1**,
  worth flagging to the fleet rather than re-discovering per session.

## Learnings

- **"Pin measurement runs to GPU1" (generic box fact) does not apply to a
  full-pod TP=8/EP=8 job** — that guidance is for isolated single-GPU runs.
  A TP=8 job has an all-or-nothing footprint across every physical GPU; any
  one busy GPU blocks the whole run, with no partial-exclusion fallback once
  the expert-count divisibility constraint is factored in.
- **GPU1 contention is now a recurring, multi-day pattern**, not a one-off —
  two independent sessions (2026-07-05 and 2026-07-06) both found it occupied
  by a standing service. Worth a fleet-level note (e.g., "tear down
  `tb_serve`-style standing servers between sessions" or "reserve GPU1 for
  terminal-bench, route all-8-GPU jobs to a different window") rather than
  re-discovering the same blocker.
- **What IS verified and SOLID this session** (not blocked by the GPU
  contention): (1) pod tree resynced from a corrupted orphan-commit state
  (`445330f`, a disconnected single-commit snapshot with no parent — likely a
  stray pod-side `git commit` from a previous session) to the correct
  `f22ad1ff0` HEAD via a full-history git bundle + `git reset --hard`; (2) the
  DSv4 MTP flag chain is real and load-bearing, not guessed:
  `--spec-type mtp` (`crates/cli/src/args.rs:676`) → `ServeSpecType::Mtp`
  (`crates/cli/src/serve.rs:363-366`) → `engine_config.mtp_draft_tokens`
  (default `DEFAULT_MTP_DRAFT_TOKENS=2`, `crates/infer-api/src/serve.rs:171`)
  → DSv4 constructor `mtp_draft_tokens` param
  (`crates/infer-cuda/src/dsv4.rs:1297-1462`), gated CUDA-only
  (`crates/cli/src/serve.rs:289-296`); (3) the release binary built clean
  twice (`BUILD_EXIT=0`, full 1m52s / incremental 42s after the concurrent
  HEAD advance) and contains the round-6 cancellation-propagation symbols
  (`cancel req#`, `RelayEnvelope::CancelRequest`) confirming it was actually
  built from the commissioned commit range, not a stale tree.

## Δ vs baseline

- **Baseline:** none — first attempt at a DSv4 TP=8 MTP guidellm sweep;
  no prior snapshot to diff against.

## Artefacts

- None produced — no bench executed.

## Notes

- What changed in code since the commissioning push: nothing DSv4/MTP-related;
  one unrelated concurrent commit landed mid-session
  (`f22ad1ff0 fix(scheduler): poll log-file content instead of just its
  presence`, `crates/infer-util/src/logging.rs`), pod tree + build were
  re-synced to it per the "never benchmark a stale tree" rule.
- Suspected cause of the GPU1 occupant not being ours: cmdline/log inspection
  (`/root/tb_serve.log`) shows it is a `Qwen3.6-27B-FP8` single-process
  (`world_size=1`) serve started independently of this task, backing a
  `terminal-bench-core==0.1.1` harness run — unrelated model, unrelated
  purpose, different session.
- Follow-ups: retry this bench once GPU1 is free (poll
  `nvidia-smi --query-gpu=index,memory.used --format=csv` for GPU1 < 1000 MiB,
  or `tn exec ps -p 2094648` to confirm the standing serve is gone), or once
  the fleet gives explicit authorization to tear down PID 2094648 on the
  H20 pod. No further code changes needed to run this bench — it is purely
  GPU-access-blocked.
