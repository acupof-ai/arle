# KV/multiproc series final verify: park round-trip live, livelock dead, crash fail-fast in 0.15s

## Context

Round 6 on the 8×H20 pod, build `5cafb308` — the first with ALL fixes:
Qwen3.6 chunked whole-slot tier (`1edaac9f`), park-or-nothing
oversubscription + checkpoint-scaled ready barrier (`5cafb308`),
crashed-worker ack-window escape (`1edaac9f`). Closes the series opened by
the kv-ssd multiproc bug.

## What Worked

- **Qwen3.5-122B TP=4 park round-trip (the round-5 livelock input)**: A+B
  simultaneous pairs now complete in ~5s (A 4.57s / B 2.57s; run 2: 4.18 /
  2.17), `demoted_slots 6→10, promoted_slots 5→8`, **zero "KV tier insert
  refused" warns** (round 5: 8,236), both texts coherent. The tp_min
  consensus that held through round-5's refusal storm now carries real
  demote/promote traffic.
- **#85 sidecar prefix reuse (was pending-remote)**: same 2614-token prompt
  twice → **6.19s cold → 1.58s warm (−74%)**, `prefix_cache.hits +1,
  hit_tokens +2608`, coherent.
- **Crashed-worker fail-fast**: `kill -9` one rank mid-decode → in-flight
  HTTP terminal 500 in **0.15s** (pre-fix: infinite hang on rate-limited
  warns); post-crash submits fail in 5ms with a clear message.
- **Checkpoint-scaled barrier**: coordinator logs `engine-ready barrier:
  3404s (checkpoint-scaled)` for the 274 GB DSv4 dir (old fixed 600s killed
  every cold boot). Warm boot 27s.
- **Lane-A empty-output NOT reproducible** (3 stagings, ids decoded): the
  only special token generated is terminal EOS — no special-token loop; the
  round-4 artifact was payload-specific. New decoded case for follow-up: the
  single-user chat render produced a spurious safety refusal on innocuous
  filler text.
- 14/14 unit tests on real CUDA (incl. the livelock regression); DSv4
  regression completion clean; pre-warm found the checkpoint fully
  page-cached (274 GB at ~55 GB/s).

## Problems

- **One graceful promote failure per rotation pair** (`swap-in requires an
  empty slot 0`): promote fires a tick before the finishing occupant
  vacates the executor pool slot; recovers via recompute — filed **#134**.
- **NCCL survivors spin after fail_all**: the hang is gone but the group
  doesn't self-exit; operator kills by PGID — filed **#135**.
- The coordinator logged its own startup truth at a dropped level —
  fixed same day (`48ceb565`, serve defaults to info).
- GPU 1 foreign-claim cost two DSv4 boots (OOM at weight alloc) — the
  pick-idle-GPUs-at-boot rule stays mandatory on this box.

## Rule

A fix series isn't closed until the exact failing input from the previous
round is re-run on the fixed build: round-5's livelocking pair became the
round-6 headline PASS, and its refusal storm count (8,236 → 0) is the
cleanest possible delta. And kill-tests belong in the protocol: "stops
hanging" (0.15s to error) and "cleans up after itself" (#135) are separate
claims — verify them separately.
