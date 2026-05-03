# W4 Session Slot Eviction Gate Miss

## Goal

Validate bounded inactive `SessionSlot` eviction for W4 canonical
`agent-w4-tool-resume`: canonical 128-session replay must reach resume phase and
show avoided-prefill > 50% as substrate proof. Mission remains >= 90%.

## Hypothesis

Session-keyed lookup was correct for one session, but canonical W4 filled T1
because inactive slots kept block-level refs. Releasing inactive slot refs under
T0/T1 pressure should let host eviction reclaim warmup KV and allow later resume
requests to attach long session prefixes.

## Params

- Commits under test:
  - `d7bca14e` `feat(scheduler): evict inactive session slots under pressure`
  - `85f7a965` `fix(scheduler): treat unset slot block location as gpu`
- Server:
  - `./target/release/infer --model-path models/default --port 8000 --num-slots 8 --max-seq-len 12288 --kv-cache-dtype fp8 --mem-fraction-static 0.85 --t1-host-pinned-high-water 0.98 --t1-host-pinned-low-water 0.95 --t1-host-pinned-capacity-mb 32768 --t1-host-pinned-min-prompt-tokens 4096`
- Trace:
  - `scripts/bench_agent_trace.py --workload agent-w4-tool-resume`
- Env:
  - NVIDIA L4 24GB
  - `CUDA_HOME=/usr/local/cuda`
  - `CARGO_HOME=/tmp/cargo-home-local`
  - `PEGAINFER_CUDA_SM=89`
  - `LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64`
  - `ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig`
  - `INFER_TILELANG_PYTHON=/usr/bin/python3`

## Verification

- `cargo fmt -p infer --check`: pass.
- `cargo test -p infer --no-default-features --features no-cuda --lib`: pass
  (`535 passed; 9 ignored`) before `d7bca14e`, and pass (`535 passed; 9
  ignored`) before `85f7a965`.
- `cargo check -p infer --no-default-features --features cuda,no-cuda`: pass
  with pre-existing `infer/src/speculative/cuda.rs` unused-import warnings.
- W4 smoke after `85f7a965`: pass.
  - warmup TTFT 1713.7 ms.
  - resume TTFT 150.6 ms.
  - `matched_prefix_tokens=8256`.
  - `resume_prefill_tokens=323`.
  - `prefix_request_skip_rate=96.2%`.

## Results

Canonical W4 reached resume phase but missed the substrate gate.

- Turns OK: `185 / 256`.
- Scored resume turns OK: `57 / 128`.
- Resume TTFT p50/p99 on scored OK turns: `10119.1 / 22636.1 ms`.
- Server `/v1/stats after`: unavailable because the server exited before the
  final stats fetch.
- Resume admissions observed in server log: `64`.
- Resume matched-prefix aggregate from admission logs:
  - matched tokens: `1,920`.
  - prompt tokens: `545,997`.
  - avoided-prefill ratio: `0.3517%`.
  - max matched prefix: `32` tokens.
  - histogram: `16 tokens x 8`, `32 tokens x 56`.

Delta:

- vs rewritten session-keyed lookup baseline `ed56cf2c`: improved from not
  reliably reaching resume to reaching resume, but still no long-prefix attach.
- vs historical W4 baselines (`e577d670` 0.35%, A3 `03253745` 0.38%, A6
  `fbb39407` 0.38%, chat-lib `c5f960a8` 0.377%): effectively unchanged
  avoided-prefill.
- vs gate: `0.3517%` actual, `>50%` required.

## Problems

- The smoke run proves the SessionSlot side index still works when only one
  session needs to survive.
- Canonical pressure did not trigger the intended slot-release path:
  `session slot pressure eviction` log count was `0`.
- Host reclaim instead spun through block demotion failures:
  `failed to demote block ... host pinned tier has no leaf eviction headroom`
  occurred `62,725` times.
- Resume admissions fell back to token-walk/common-prefix behavior (`16` or
  `32` tokens), so the same W4 chat-template divergence still dominated.
- The run eventually lost the server before final `/v1/stats`, producing 71
  resume errors after the first 57 scored resumes.

## Root Cause

The bounded eviction policy exists, but it is not on the effective pressure path
for canonical W4. Slot-held block refs still protect the warmup KV while host
demotion is already out of leaf headroom, so reclamation scans and fails instead
of releasing inactive slot refs early enough. The small `metadata.location=None`
fix was necessary for GPU-tier filtering, but not sufficient; canonical behavior
shows the slot lifecycle/pressure hook remains miswired.

## Rule

A one-session W4 smoke is necessary but not sufficient for session-keyed lookup
work. Any future slot-retention change must prove, during canonical warmup
pressure, that inactive slot eviction actually fires before resume:

- `session slot pressure eviction` count > 0 under pressure.
- resume `matched_prefix_tokens` distribution has long-prefix entries
  (8k-scale), not only 16/32.
- canonical `stats_after` remains available.

