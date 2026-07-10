# DSv4 R4 fixes pod acceptance + Wave B' coverage matrix

## Context

Pod verification of the codex-R4 series (`4ad32362e`: chain-protected pool
eviction, f7891c3f0 revert; binary sha256 `41eccec9…`, TP=4 H20,
DeepSeek-V4-Flash-FP8, allreduce/incremental-KV/deepgemm, mtt=2048) plus the
Wave B' coverage matrix queued from the KV-reuse refactor. GPUs 0-3 (4/5 held
by a foreign serve); W1-W3 ran against a same-binary TP=8 mtt=16384 serve
(sha256-verified via /proc/pid/exe, request windows counter-validated clean).

## What Worked

| Lane | Result |
|---|---|
| prefix_state unit tests (incl. new `eviction_protects_publishing_chain`) | 4/4 pass |
| A warm cross-request reuse ×10 salts | 10/10 exact; per-pair stats `hits +1, hit_tokens +384 = floor(462/128)·128`; warm 0.86-1.41s vs cold 1.56-2.55s |
| B solo cold ×15 | 14/15 (meets baseline); miss = `738292` digit substitution |
| C n=4 ×5 (20 req) | 20/20 exact |
| W2 shared-preamble floor ×3 shapes | matched deltas 256/640/1920 = exact `floor(shared/128)·128` for pt 354/754/1955 |
| W3 restore TTFT pt=1749 ×5 | median cold 3.98s → resend 0.51s (7.9×); no bulk-rebuild spike |
| W4 `--kv-dram 64MB` pressure boot | no panic/wedge, resend exact via clean recompute, pool_refused=0 |
| W5 16K boot (mtt=16384), len 2000+8000 ×5 each | 10/10 exact — the pre-existing 16K truncation did NOT reproduce on this binary |

## Findings (carried forward)

- **W1 multi-turn: generated pages never hit.** Turn-2 (P+R+question) matched
  exactly `floor(pt1/128)·128` — prompt-only floor, `crosses_into_R=false`
  5/5. Only prefill pages publish+confirm; finish-time seal of the generated
  region is missing or unconfirmed → multi-turn extension reuse is
  prompt-bounded. Filed as an issue.
- **Substitution class is solo-reachable**: two `738292` events (Lane B rep 3,
  W2 N=700 tail), both temp=0 COLD SOLO at pt≥462 — batch-size dependence is
  not a prerequisite for the #150 signature (~1/15-1/30 solo rate here).
  Posted to #150.
- **W4 under-stressed**: 64MB is below one chain (`published_pages=0`), so
  chain protection under live L2 pressure ran only as the unit test; a
  chain-sized-budget lane needs a supported mid-range value.
- One non-reproducing lockstep wedge (`ack wait exceeded 120s at tick #4`) on
  one boot, confounded by a foreign TP=8 relaunch on overlapping GPUs; 3
  clean boots of the same binary+config served 100+ requests. Watch-list.
- pt=461-shape salts answer `…is **738**.` truncated by max_tokens=16
  identically cold and warm — harness shape artifact, reconfirmed; use
  pt=462-shape salts for gates.

## Rule

- Verification lanes on a shared box must pin per-lane provenance (binary
  sha256 via `/proc/<pid>/exe`, GPU set, serve config) — two lanes here ran
  on a foreign-booted serve and stayed valid only because the binary hash and
  request-window counters were checked.
- A pressure lane needs its budget sized to actually cross the mechanism
  under test (64MB < one chain ⇒ the eviction path never fired).
