# Unified KV tier (prefix-in-store + new flags) verified: NVMe prefix restore 26.6s → 0.47s

## Context

Series `17aeebcc` (DSv4 prefix cache rides THE tier store, env knob deleted,
2-collective lockstep contract) + `8b72083a`/`1564a342` (unified flags:
`--kv-dram` / `--kv-disk` / `--kv-disk-limit` / `--kv-oversubscription`,
deployment-total ÷ world budgets, fail-loud gates, resolved-tier log line).
Verified on the 8×H20 pod (DSv4-Flash-FP8, TP=4, GPUs 1/3/4/6) and locally on
Metal (M4 Pro).

## What Worked

Pod, serve flags `--kv-disk /host/... --kv-disk-limit 64GiB --kv-dram 0
--kv-oversubscription --max-running-requests 1`:

- **Unit tests on real CUDA**: `PrefixIndex` ×4 + namespace disjointness +
  manifest round-trip — 6/6 green (`cargo test -p infer-cuda --features cuda`).
- **Resolved-tier line identical on all 4 ranks**: `KV tiers: dtype=bf16 | L1
  mem_fraction_static=0.9 | L2 0B/rank (deployment Off, world 4) | L3
  root=... cap 17179869184B/rank | features: prefix,park` — 64 GiB ÷ 4 =
  16 GiB/rank as designed.
- **Prefix reuse through NVMe** (3375-token prompt, T=0, `--kv-dram 0` forces
  blobs straight to disk): cold prefill **26.62 s** → restored **0.470 s** →
  **0.470 s** (runs 2–3), identical answers, `hit_rate=1.0`,
  `hit_tokens=6750`, prefix blobs persist at rest (~64 MB real mmap blocks per
  rank; idle `disk_pages=5`). Zero restore/peer-rank errors in the session log.
  (Run 1 includes first-request warmup; 0.47 s total round-trip vs a 3.4K-token
  prefill is unambiguous regardless.)
- **Park round-trip**: simultaneous A(256)+B(64) → `demoted_slots=11,
  promoted_slots=11, slot_promote_failures=0`; mid-run stats observable
  (`active=1 queue=1 demoted=8 promoted=7` at ~4s). Coherent outputs.
- **Flag surface**: all 7 deleted flags rejected by clap (with did-you-mean
  pointing at the new names); `--kv-disk-limit` without `--kv-disk` fails
  pre-model-load.
- **Metal local** (Qwen3.5-0.8B-MLX-4bit; canonical 35B skipped — resource
  guard correctly rejected 19 GiB weights on a box with 17 GiB available, which
  itself exercises the guard): in-builder T2 attach (per-PID namespace,
  4 GiB / 42366 pages), write-through wrote 19 MB after two requests,
  `prefix_match_full_blocks=1` on the second identical prompt,
  `--kv-recall --kv-cache-dtype bf16 --kv-disk` combo boots and serves, and the
  no-flag default path logs zero tier lines (clean default).

## Problems

- **Prefix blob accounting leak (under investigation)**: identical-prompt
  re-publishes accumulate `disk_pages` (5→15→25→35) — superseded snapshots
  appear retained. Note the engine captures at TWO sites (prefill-complete
  with `prompt_tokens`, `finish_slot` with prompt+generated), so distinct-key
  blobs per request are expected; whether the supersede-removal path also
  fails needs a decoded case. Fix path: sink the chunked-blob API into
  `CudaKvTierStore` (host-testable) + a double-capture accounting unit test.
- `kv_tier.available` gauge reads `false` until the first demote even with the
  store attached; prefix restores bucket under `reuse_hit_resident` (never
  `reuse_hit_disk`) because `record_attached_prefix_metrics` keys on page-tier
  capacity, not blob placement. Counter-truth fixes pending.

## Rule

A restore that answers in 0.47 s what prefill answers in 26 s is licensed by
`hit_tokens`/`disk_pages`/mmap-blocks agreeing — never by wall-clock alone.
And any best-effort cache write path needs an accounting test that inserts the
same key twice and asserts the store returns to baseline: supersede-without-
remove is invisible in functional tests and only shows up as monotonic
`disk_pages`.
