# W4 Canonical Gate Miss - agent-load benchmark, agent-w4-tool-resume, arle-cuda, 2026-05-04

## Goal

Validate the shipped W4 H4 pressure-eviction fix at `23f97c09` against the
canonical `agent-w4-tool-resume` trace on the L4 CUDA box.

Goal type: regression / diagnosis.

## Hypothesis

The H4 fix should turn canonical W4 pressure into bounded inactive
`SessionSlot` eviction before host pinned leaf eviction exhausts headroom:

- `session slot pressure eviction` should be non-zero.
- host pinned leaf-headroom failures should drop below `6,272` (10x below the
  `62,725` before-snapshot).
- resume admissions should show at least 30 long-prefix matches (`>=8000`
  tokens), driving avoided prefill above 50% with mission success above 90%.

## Command

Build:

```bash
source /tmp/arle-env.sh
CUDA_HOME=/usr/local/cuda cargo build --release -p infer
CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda --bin infer
```

Note: the first command is the briefed command and completed, but the workspace
has empty default features and only produced `libinfer`. The `infer` server
binary has `required-features = ["cuda"]`, so the explicit CUDA binary build
was required to run the bench.

Server:

```bash
source /tmp/arle-env.sh
./target/release/infer --model-path models/default --port 8000 \
  --num-slots 8 --max-seq-len 12288 --kv-cache-dtype fp8 \
  --mem-fraction-static 0.85 \
  --t1-host-pinned-high-water 0.98 --t1-host-pinned-low-water 0.95 \
  --t1-host-pinned-capacity-mb 32768 --t1-host-pinned-min-prompt-tokens 4096 \
  2>&1 | tee bench-output/2026-05-04-w4-canonical/server.log
```

Client:

```bash
source /tmp/arle-env.sh
PYTHONUNBUFFERED=1 python3 scripts/bench_agent_trace.py \
  --workload agent-w4-tool-resume \
  --server http://127.0.0.1:8000 \
  --label w4-canonical-23f97c09 \
  --out bench-output/2026-05-04-w4-canonical/results.json \
  --trace-out bench-output/2026-05-04-w4-canonical/trace.jsonl \
  2>&1 | tee bench-output/2026-05-04-w4-canonical/client.log
```

Stats:

```bash
curl -s http://127.0.0.1:8000/v1/stats \
  > bench-output/2026-05-04-w4-canonical/stats_after.json
```

## Environment

- **Workload:** `agent-w4-tool-resume`.
- **Backend / engine:** `arle-cuda`.
- **Model:** `models/default -> Qwen3-4B`.
- **Hardware:** NVIDIA L4, 23,034 MiB VRAM.
- **Driver / CUDA:** NVIDIA driver `580.82.07`, `nvcc` CUDA `12.8`.
- **Commit:** `23f97c09c20e99dd95f7fc0d57fd77242d11f7c4`.
- **Feature set:** `cargo build --release -p infer --features cuda --bin infer`.
- **KV dtype / cache mode:** paged KV `FP8E4M3`, contiguous cache `BF16`.
- **Session / prefix flags:** prefix cache on, `--num-slots 8`,
  `--max-seq-len 12288`, T1 host pinned capacity `32768 MiB`, high/low water
  `0.98 / 0.95`, min T1 prompt tokens `4096`.
- **Scheduler capacity:** `137,328` max paged tokens, `8,583` pages, page size
  `16`, host pool `34,359.7 MB`.

## Results

The run was stopped after an irreversible gate miss. At request pressure around
`83` in `/v1/stats` and request `92` in the final server log, the host
leaf-headroom failure count had already exceeded the acceptance threshold:
`10,516 > 6,272`. Continuing could not make Gate 2 pass, so final stats were
captured and the server was stopped.

### Acceptance Gates

| gate | target | actual | status |
|---|---:|---:|---|
| 1. Session slot pressure eviction | `>0` and hard counter visible | `75` log lines; `session_slot_pressure_evictions_hard=0`; all observed lines were `mode=Soft` | **MISS / partial** |
| 2. Host leaf-headroom failures | `<6,272` | `10,516` at abort | **MISS** |
| 3. Long matched-prefix admissions | `>=30` entries with `matched >=8000` | `0`; attach histogram was `radix_gpu_attach=32 x85`; generated trace has no admission rows | **MISS** |
| 4. `/v1/stats` available | final stats fetch succeeds | `stats_after.json` captured | PASS |
| 5. Avoided prefill and mission success | avoided-prefill `>=50%`, mission success `>=90%` | stats snapshot: `matched_prefix_tokens=32`, `resume_prefill_tokens=8180`, avoided-prefill `0.39%`; full mission result unavailable because run was stopped after Gate 2 became impossible | **MISS** |

### Service Snapshot

`stats_after.json` was captured before shutdown:

| metric | value |
|---|---:|
| requests | 83 |
| active / waiting / scheduled | 8 / 0 / 8 |
| kv_util | 100.0% |
| prefix_hit_rate | 91.2% |
| prefix_skip_rate | 0.4% |
| prefix_request_hit_rate | 100.0% |
| prefix_request_skip_rate | 0.4% |
| session_affinity_hit / miss | 83 / 8 |
| `session_slot_pressure_evictions_hard` | 0 |
| `matched_prefix_tokens` | 32 |
| `resume_prefill_tokens` | 8180 |

Final server-log counters:

| metric | value |
|---|---:|
| `session slot pressure eviction` lines | 75 |
| `host pinned tier has no leaf eviction headroom` lines | 10,516 |
| `prefix cache pressure fallback` lines | 32 |
| completed request lines | 87 |
| zero-token completions | 2 |
| `radix_gpu_attach=32` admissions | 85 |
| `radix_gpu_attach>=8000` admissions | 0 |

## Delta Vs Before-Snapshot

- **Before-snapshot:** [`2026-05-03-bench-agent-load-session-slot-eviction-w4-gate-miss.md`](2026-05-03-bench-agent-load-session-slot-eviction-w4-gate-miss.md).

| metric | 2026-05-03 before | 2026-05-04 W4 canonical | delta |
|---|---:|---:|---:|
| session slot pressure eviction lines | 0 | 75 | new signal |
| hard pressure eviction counter | unavailable | 0 | still not firing |
| host leaf-headroom failures | 62,725 | 10,516 | -83.2% |
| target host failures | `<6,272` | 10,516 | miss by 4,244 |
| max matched prefix | 32 | 32 | flat |
| `>=8000` matched-prefix admissions | 0 | 0 | flat |
| avoided-prefill ratio | 0.3517% | 0.39% snapshot | +0.04 pp |
| final `/v1/stats` | unavailable | available | fixed |

The 83.2% drop in host leaf-headroom failures is real but not enough. The
acceptance gate required a 90% drop, and the run had not reached the resume
phase when it crossed the miss threshold.

## Problems

- HardPressure did not fire. The stats surface reported
  `session_slot_pressure_evictions_hard=0`, and every observed pressure log was
  `mode=Soft`.
- Soft pressure eviction released inactive slots one at a time, but once T1 hit
  leaf-headroom exhaustion, host demotion still entered a long warning loop.
- The scheduler eventually fell back to dropping GPU blocks:
  `prefix cache pressure fallback` occurred `32` times.
- Admissions still attached only the short shared prefix (`32` tokens), not
  session-keyed long prefixes.
- Two warmup requests completed with zero generated tokens before shutdown.
- SIGINT was delivered after `stats_after.json` was captured, but the scheduler
  thread did not exit cleanly; SIGTERM was required to stop the process.

## Hypothesis For Why HardPressure Did Not Help

The new pressure path proves that inactive slot release is wired, but the T1
leaf-headroom exhaustion signal is not escalating into hard session-slot
eviction in time. The runtime keeps releasing soft-eligible slots and then
attempts host demotion even though the host tier has no leaf eviction headroom.
Because the hard counter remains `0`, the W4 path still reaches the old failure
shape: repeated demotion failures, fallback GPU drops, and only 32-token
prefix attaches.

This is not an obvious one-line follow-up from the bench alone. The next code
change should target the scheduler pressure trigger/eligibility boundary so T1
leaf-headroom exhaustion forces hard inactive-slot release before the demotion
loop, and should preserve a hard-counter assertion in the canonical W4 run.

## Fix

No code follow-up was committed in this validation tick. The root cause is
localized to pressure-mode escalation/reclaim behavior, but the exact minimal
patch is not obvious without another scheduler pass. Per the brief, this entry
records the gate miss and stops.

## Rule

For W4 canonical, `session slot pressure eviction` being non-zero is not enough.
The hard-pressure counter must be non-zero under T1 leaf-headroom exhaustion,
and host leaf-headroom warning volume must stay below the 10x-drop threshold
before the run can be treated as a pressure-eviction fix.

## Artefacts

- Client log: `bench-output/2026-05-04-w4-canonical/client.log`
- Generated trace: `bench-output/2026-05-04-w4-canonical/trace.jsonl`
- Server log: `bench-output/2026-05-04-w4-canonical/server.log`
- Server launch: `bench-output/2026-05-04-w4-canonical/server_launch.txt`
- Stats before: `bench-output/2026-05-04-w4-canonical/stats_before.json`
- Stats after: `bench-output/2026-05-04-w4-canonical/stats_after.json`
- GPU env: `bench-output/2026-05-04-w4-canonical/gpu_env.txt`
