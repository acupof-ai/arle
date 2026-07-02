# KV tier close-out: leak attributed (bounded by design), admission window live, bandwidth measured

## Context

Rounds 3–4 on the 8×H20 pod (DSv4-Flash TP=4, GPUs re-picked at boot) closing
the series: tick-ack lockstep window (`27162919`), Qwen3.6 TP consensus
(`0fdbf711`), disk-before-recall (`5009dc24`), chunked-blob API + capture
logging + counter truth (`365796d2`), oversize fail-closed (`326fab5d`),
rank-prefix init-order fix (`e855141c`, from a round-4 finding).

## What Worked

- **Round-3 gate caught a process-abort bug before any serve**: the disk
  level's `write_slot` asserts `len <= slot_bytes` while the host level
  accepted any size — an oversize payload aborted the process on cold-spill
  (immediately under `--kv-dram 0`). One size contract now lives at
  `insert()` (refuse + warn → recompute). 11/11 unit tests green on CUDA in
  round 4, including the failed disk variant and the new fail-closed test.
- **"Leak" attributed — NOT a leak** (decoded case, same 3770-token prompt
  ×4, T=0): run 1 stores the prompt blob (5 pages); runs 2–4 restore the
  FULL prompt (no prefill ⇒ no re-capture of the prompt blob) and the
  finish-sidecar captures (prompt+generated, identical at T=0) supersede in
  place — capture logs show `key=N superseded key=N-1` on all 4 ranks,
  `disk_pages` stable at 10, no key ever leaks. Distinct generated suffixes
  accumulate one blob per distinct vector by design, bounded by the per-rank
  cap via coldest-prefix eviction (P3: +4 blobs for 4 new vectors, then
  flat). Latency: 19.7s cold → 0.44–0.49s restored; answers identical.
- **Admission window (tick-ack, window=4)**: B fired 2s into A's decode —
  the timing that could NEVER admit before — completes in **5.24s**
  (mid-A, park fires: demoted/promoted 11/11, failures 0). `/v1/stats`
  mid-decode: **0.138–0.307s** (was a 12.3s stall answered after
  completion). Counter-truth fixes verified: `kv_tier.available=true` from
  the first stored blob; `reuse_hit_resident` stays 0 on whole-slot reuse
  while `prefix_cache.hits`/`hit_tokens` climb.
- **Bandwidth (打满)**: device baseline (virtio `/dev/vda2`, dd QD1 direct)
  197/195 MB/s write/read. Store path (2 GiB of production-shaped chunked
  blobs, `--kv-dram 0` shape): **write 1.20 GiB/s (6.5× device QD1 — page
  cache absorbing, nothing forces synchronous writeback), warm read
  2.22 GiB/s**. Serve-level park churn (11 A+B pairs / 65s, cap=1): 4.6
  parks/min, zero promote failures, total tier traffic ≈ 0.19 GB/s — ~6×
  under the store-path ceiling; storage is NOT the bottleneck for
  park/prefix at these shapes. Churn aggregate 22.6 tok/s vs 17.3 solo
  (solo included a cold prefill — no clean warm control; at minimum no net
  aggregate loss).
- `tn hold` (persistent connection daemon, tune `a3635aa`) carried both
  rounds: round-3's 41 tool calls in 6.3 min vs round-1's 109 in 31 min.

## Problems

- Round-2's `disk_pages` 5→15→25→35 was a mixed-phase observation read as a
  leak; the controlled round-4 series bounded it. Rule reinforced: decode
  the per-event sequence before naming something a leak.
- Worker `[rankN]` log prefixes were silently dropped: the generic logger
  init ran before `worker_entry` and `init` is call_once — fixed by
  short-circuiting workers first (`e855141c`); re-verify prefixes next pod
  session.
- Hygiene for next rounds: `cargo test --features cuda` clobbers
  `target/release/arle` (feature unification) — build the serve bin after
  tests or use a separate target dir; 16 stranded day-old `arle` processes
  on the pod predate these sessions (cleanup next hold); GPU 1 keeps being
  claimed by a foreign process — pick idle GPUs at boot, every time.
- Open (filed for follow-up, not KV): "Lane A." style prompts steer DSv4
  into an unclosed reasoning segment at T=0 → empty visible content with
  correct token accounting (template/steering artifact).

## Rule

A monotonic gauge is not a leak until the per-event sequence says so: log
insert/supersede/evict with store counts at the source, then read the events
— round 2's "leak" dissolved into two capture sites doing their jobs.
And any two-level store needs ONE size contract at the insert boundary;
letting levels disagree turns a cache miss into a process abort.
