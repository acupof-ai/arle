# W4 H5 Gate Miss - agent-load benchmark, agent-w4-tool-resume, arle-cuda, 2026-05-05

## Goal

Validate the W4 H5 host leaf pressure eviction fix at `bae845e0` against the
canonical `agent-w4-tool-resume` trace on the L4 CUDA box.

Goal type: regression / diagnosis.

## Hypothesis

The H5 fix should escalate T1 host leaf-headroom exhaustion into hard inactive
`SessionSlot` eviction before host demotion enters the old warning loop:

- `session_slot_pressure_evictions_hard` should be non-zero.
- host pinned leaf-headroom failures should stay below `6,272`.
- resume admissions should show at least 30 long-prefix matches (`>=8000`
  tokens), driving avoided prefill above 50% with mission success above 90%.

## Command

Build:

```bash
source /tmp/arle-env.sh
CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda --bin infer
```

Server:

```bash
source /tmp/arle-env.sh
./target/release/infer --model-path models/default --port 8000 \
  --num-slots 8 --max-seq-len 12288 --kv-cache-dtype fp8 \
  --mem-fraction-static 0.85 \
  --t1-host-pinned-high-water 0.98 --t1-host-pinned-low-water 0.95 \
  --t1-host-pinned-capacity-mb 32768 --t1-host-pinned-min-prompt-tokens 4096 \
  2>&1 | tee bench-output/2026-05-05-w4-h5/server.log
```

Client:

```bash
source /tmp/arle-env.sh
PYTHONUNBUFFERED=1 python3 scripts/bench_agent_trace.py \
  --workload agent-w4-tool-resume \
  --server http://127.0.0.1:8000 \
  --label w4-h5-efccbc0d \
  --out bench-output/2026-05-05-w4-h5/results.json \
  --trace-out bench-output/2026-05-05-w4-h5/trace.jsonl \
  2>&1 | tee bench-output/2026-05-05-w4-h5/client.log
```

Stats:

```bash
curl -s http://127.0.0.1:8000/v1/stats \
  > bench-output/2026-05-05-w4-h5/stats_after.json
```

## Environment

- **Workload:** `agent-w4-tool-resume`.
- **Backend / engine:** `arle-cuda`.
- **Model:** `models/default -> Qwen3-4B`.
- **Hardware:** NVIDIA L4, 23,034 MiB VRAM.
- **Driver / CUDA:** NVIDIA driver `580.82.07`, `nvcc` CUDA `12.8`.
- **Scheduler fix commit:** `bae845e0`.
- **Bench head / label:** `efccbc0d`, `w4-h5-efccbc0d`.
- **Feature set:** `cargo build --release -p infer --features cuda --bin infer`.
- **KV dtype / cache mode:** paged KV `FP8E4M3`, contiguous cache `BF16`.
- **Session / prefix flags:** prefix cache on, `--num-slots 8`,
  `--max-seq-len 12288`, T1 host pinned capacity `32768 MiB`, high/low water
  `0.98 / 0.95`, min T1 prompt tokens `4096`.
- **Scheduler capacity:** `137,328` max paged tokens, `8,583` pages, page size
  `16`, host pool `34,359.7 MB`.

## Results

The run completed all `256 / 256` turns with all `128` scored resume turns OK.
H5 fixed the hard-pressure counter and removed the prior host leaf-headroom
warning train, but the long-prefix admission and avoided-prefill gates still
missed.

Client summary:

| metric | value |
|---|---:|
| turns OK | 256 / 256 |
| scored turns OK | 128 |
| tokens total | 32,286 |
| wall total | 4,918.59 s |
| TTFT p50 / p99 | 4,938.6 / 22,873.9 ms |
| ITL p50 / p99 | 73.4 / 86.8 ms |
| W4 resume TTFT p50 / p99 | 4,938.6 / 22,873.9 ms |
| W4 resume E2E p50 / p99 | 37,282.4 / 55,182.5 ms |

### Acceptance Gates

| gate | target | actual | status |
|---|---:|---:|---|
| 1. Session slot pressure eviction | hard counter `>0` | `session_slot_pressure_evictions_hard=196`; `196` hard log lines | PASS |
| 2. Host leaf-headroom failures | `<6,272` | `0` exact prior warning lines | PASS |
| 3. Long matched-prefix admissions | `>=30` entries with `matched >=8000` | `0`; attach histogram still includes `radix_gpu_attach=32 x232` | **MISS** |
| 4. `/v1/stats` available | final stats fetch succeeds | `stats_after.json` captured after scheduler settled to `active=0` | PASS |
| 5. Avoided prefill and mission success | avoided-prefill `>=50%`, mission success `>=90%` | mission `100%`; avoided-prefill `32 / 8,557 = 0.37%` | **MISS / partial** |

### Service Snapshot

`stats_after.json` was captured after the final cleanup pass:

| metric | value |
|---|---:|
| requests | 256 |
| active / waiting / scheduled | 0 / 0 / 0 |
| kv_util | 86.8% |
| prefix_hit_rate | 96.9% |
| prefix_skip_rate | 0.4% |
| prefix_request_hit_rate | 100.0% |
| prefix_request_skip_rate | 0.4% |
| session_affinity_hit / miss | 248 / 8 |
| `session_slot_pressure_evictions_hard` | 196 |
| `matched_prefix_tokens` | 32 |
| `resume_prefill_tokens` | 8,557 |
| `tokens_out` | 40,451 |
| `step_p50` | 75.0 ms |
| `active_mem` / `peak_mem` | 19,939.1 / 21,571.1 MB |

Final server-log counters:

| metric | value |
|---|---:|
| `session slot pressure eviction` lines | 251 |
| `session slot pressure eviction: mode=Hard` lines | 196 |
| `host pinned tier has no leaf eviction headroom` lines | 0 |
| `prefix cache pressure fallback` lines | 314 |
| `host tier full` lines | 313 |
| completed request lines | 256 |
| `radix_gpu_attach=32` admissions | 232 |
| `radix_gpu_attach>=8000` admissions | 0 |

## Delta Vs Before-Snapshot

- **Before-snapshot:** [`2026-05-04-bench-agent-load-w4-canonical-gate-miss.md`](2026-05-04-bench-agent-load-w4-canonical-gate-miss.md).

| metric | 2026-05-04 W4 canonical | 2026-05-05 W4 H5 | delta |
|---|---:|---:|---:|
| hard pressure eviction counter | 0 | 196 | fixed |
| host leaf-headroom failures | 10,516 | 0 | -100.0% |
| target host failures | `<6,272` | 0 | pass |
| max matched prefix | 32 | 32 | flat |
| `>=8000` matched-prefix admissions | 0 | 0 | flat |
| avoided-prefill ratio | 0.39% snapshot | 0.37% final | -0.02 pp |
| final `/v1/stats` | available | available | unchanged |
| mission success | unavailable; run stopped after gate miss | 256 / 256 turns OK | fixed run completion |

The H5 pressure fix is effective for the specific hard-escalation failure from
the before-snapshot. It does not fix the separate W4 admission failure: resumed
requests still attach only the short shared prefix, so avoided prefill remains
effectively unchanged.

## Problems

- Long-prefix session admissions still do not attach the retained session
  prompt. The run produced `0` admissions with `radix_gpu_attach>=8000` and
  final stats still reported `matched_prefix_tokens=32`.
- Avoided prefill remains far below the `>=50%` gate at `0.37%`.
- The scheduler still falls back to dropping GPU blocks under pressure:
  `prefix cache pressure fallback` occurred `314` times and `host tier full`
  occurred `313` times.
- The client-side `/v1/stats` sample in `client.log` showed `active=1` because
  it raced the final scheduler cleanup. A separate final `stats_after.json`
  fetch showed the settled state at `active=0`.

## Hypothesis For The Remaining Miss

H5 successfully converts the host leaf-headroom signal into hard inactive-slot
eviction, so the previous pressure-mode boundary is no longer the limiting
fault. The remaining miss is likely in the session-keyed long-prefix
publish/admission path: retained resumes are still matching only the
32-token shared system prefix instead of the per-session warmup prompt.

The next code change should target long-prefix publication, keying, or
admission lookup. More hard-pressure tuning is unlikely to move the W4 avoided
prefill gate by itself.

## Fix

No code follow-up was committed after this validation run. Per the brief, this
entry records the gate miss and stops this tranche with the H5 pressure fix
shipped but the W4 long-prefix gate still open.

## Rule

For W4 H5, hard pressure eviction and leaf-headroom warning elimination are
necessary but not sufficient. The run is not a win until it also shows at least
30 `>=8000`-token prefix admissions and avoided prefill above 50%.

## Artefacts

- Client log: `bench-output/2026-05-05-w4-h5/client.log`
- Generated trace: `bench-output/2026-05-05-w4-h5/trace.jsonl`
- Server log: `bench-output/2026-05-05-w4-h5/server.log`
- Results JSON: `bench-output/2026-05-05-w4-h5/results.json`
- Stats after: `bench-output/2026-05-05-w4-h5/stats_after.json`
