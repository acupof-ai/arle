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
  - `758ea29e` `feat(scheduler): evict inactive session slots under pressure`
  - `d505e06e` `fix(scheduler): treat unset slot block location as gpu`
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
  (`535 passed; 9 ignored`) before `758ea29e`, and pass (`535 passed; 9
  ignored`) before `d505e06e`.
- `cargo check -p infer --no-default-features --features cuda,no-cuda`: pass
  with pre-existing `infer/src/speculative/cuda.rs` unused-import warnings.
- W4 smoke after `d505e06e`: pass.
  - warmup TTFT 1713.7 ms.
  - resume TTFT 150.6 ms.
  - `matched_prefix_tokens=8256`.
  - `resume_prefill_tokens=323`.
  - `prefix_request_skip_rate=96.2%`.
- W4 root-cause debug replay after canonical miss:
  - Trace:
    `bench-output/2026-05-03-session-slot-evict-w4-smoke/one-session.jsonl`.
  - Output:
    `bench-output/2026-05-03-w4-root-cause-debug-1session-r2/results.json`.
  - Server log:
    `bench-output/2026-05-03-w4-root-cause-debug-1session-r2/server.log`.
  - Temporary `W4DBG` instrumentation was used for this replay only and was
    removed before committing this entry.
  - Result: resume TTFT `146.3 ms`, `matched_prefix_tokens=8256`,
    `resume_prefill_tokens=323`, `prefix_request_skip_rate=96.2%`.

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
- The debug replay confirms the SessionSlot functional path is internally
  consistent when the slot survives:
  - publish stored `committed_len=8256`, `blocks=516`, `block_size=16`.
  - lookup requested `8256` tokens and returned `matched_len=8256`,
    `lookup_blocks=516`, `recompute=false`.
  - `lookup_session_blocks` recorded `516` metadata-present blocks and no
    `has_metadata=false` miss.
  - admission selected the session slot and planned direct GPU attach:
    `session_slot=true`, `lookup_matched=8256`, `gpu_ready_tokens=8256`,
    `direct_gpu_attach=true`.
  - metrics recorded the slot value, not the token-walk value:
    `matched_prefix_tokens=8256`, `prompt_tokens=8579`,
    `resume_prefill_tokens=323`.
- Canonical pressure did not trigger the intended slot-release path:
  `session slot pressure eviction` log count was `0`.
- Host reclaim instead spun through block demotion failures:
  `failed to demote block ... host pinned tier has no leaf eviction headroom`
  occurred `62,725` times.
- Resume admissions fell back to token-walk/common-prefix behavior (`16` or
  `32` tokens), so the same W4 chat-template divergence still dominated.
- The run eventually lost the server before final `/v1/stats`, producing 71
  resume errors after the first 57 scored resumes.

## Hypothesis Check

H1: `publish_session_slot` receives only the sealed radix prefix.

- Status: no for the 1-session control.
- Evidence: warmup finish logged `committed_len=8256`, `blocks=516`,
  `block_size=16`; this is the full warmup prefix rounded to full 16-token
  blocks, not a `~32/2` partial prefix.
- Next implication: the publish call site is not the immediate functional bug.

H2: `lookup_session_blocks` stops at the first metadata miss, e.g. around block
3.

- Status: no for the 1-session control; not proven for canonical pressure
  without pressure-time instrumentation.
- Evidence: the debug replay logged `516` `has_metadata=true` blocks, no
  `has_metadata=false` line, and returned `matched_len=8256`.
- Canonical clue: canonical resumes still attached only `16` or `32` tokens, so
  a pressure-only slot disappearance/metadata-loss path remains possible, but
  the small debug replay rules out a basic lookup walker bug.

H3: slot lookup returns 8k, but `matched_prefix_tokens` is reported from the
regular token-walk path.

- Status: no.
- Evidence: admission logged `session_slot=true`, `lookup_matched=8256`,
  `direct_gpu_attach=true`, and prefill metrics logged
  `matched_prefix_tokens=8256`, matching `/v1/stats` after the replay.
- Next implication: the A2 stats path is wired to the session-slot attach path
  for successful slot hits.

H4: canonical pressure never makes inactive slots eligible for release before
T0/T1 reclamation exhausts headroom.

- Status: confirmed as the canonical failure mode to fix next.
- Evidence: canonical W4 had `session slot pressure eviction` count `0`, while
  host demotion failed `62,725` times with `host pinned tier has no leaf
  eviction headroom`; resume admissions attached only `16/32` tokens
  (`Request 128-135` attached `16`, later resumes `32`).
- Code-path evidence: `evict_inactive_session_slots_for_pressure` only releases
  slots that pass all three gates: `ref_count == 0`,
  `last_access_tick + prefix_cache_keepalive_ticks <= now`, and tier match
  (`Gpu`/`HostPinned`). Under canonical pressure, no slot passed that filter,
  so `session_protected_blocks()` continued excluding slot blocks from radix
  eviction and reclamation fell back to repeated demotion failures.
- Next implication: the next scheduler change should be a pressure-specific
  inactive-slot release path that can drop LRU inactive slots under T0/T1
  headroom pressure before host demotion loops, with logging that proves it
  fires during canonical warmup pressure.

## Root Cause

The bounded eviction policy exists, but its eligibility gate is too narrow for
canonical W4 pressure. One-session replay proves publish, slot lookup,
admission, and metrics are correct when the slot survives. Canonical pressure
instead reaches T0/T1 reclaim with all inactive session slots still protected:
no inactive slot eviction fires, host demotion has no T1 leaf headroom, and
resume requests fall back to radix/token-walk matches of `16/32` tokens. The
small `metadata.location=None` fix was necessary for GPU-tier filtering, but
not sufficient; the remaining bug is the pressure-time slot eviction trigger and
eligibility policy, not the SessionSlot lookup substrate itself.

## Rule

A one-session W4 smoke is necessary but not sufficient for session-keyed lookup
work. Any future slot-retention change must prove, during canonical warmup
pressure, that inactive slot eviction actually fires before resume:

- `session slot pressure eviction` count > 0 under pressure.
- resume `matched_prefix_tokens` distribution has long-prefix entries
  (8k-scale), not only 16/32.
- canonical `stats_after` remains available.
