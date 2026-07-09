# DSv4 content-keyed prefix-state pool (Phase 2a) — pod evidence gate PASSED

> Status: **Shipped** — fix commit `0b5bd3d55` on top of the code series
> (`5dc5ef79b`, `754648421`, `14b264ea7`). 4×H20 (GPUs 4-7), TP4,
> DeepSeek-V4-Flash-FP8, `--max-total-tokens 2048`, binary sha256
> `7b76c44d1c20…` @ tree `0b5bd3d5`.

## Context

Reland of DSv4 cross-request prefix reuse: host-resident `Dsv4PrefixStatePool`,
content-keyed by host page id, written once per completed page from the
executor choke point, restored on radix prefix match, spilled to L3 mmap under
the `--kv-dram` share. Plan: `docs/plans/2026-07-09-dsv4-kv-reuse-seam-refactor.md`
Phase 2. The first pod round FAILED E2 (warm resends 40-60% corrupt,
`'738'` early-EOS + `291→292` digit subs, cache-off control clean).

## E2 root cause — byte-proven, then fixed

The FP8 compressed band is packed ONLY on the decode lane
(`flashmla_pack_compressed_delta`; every FP8 pool pack site is decode-only).
Publish captures at prefill chunk ends — an env-gated fingerprint probe showed
`band=0/196224` (all zeros, the `zero_slot_band` fill) on every cold-published
page, while staging/dsa/ring sections were valid and restore read back
byte-identical hashes across resends (no lifecycle pollution). Restore then
wrote those zeros over the band AND set `fp8_kv_comp_packed_rows=matched/ratio`,
disabling the decode-lane bulk self-heal — every warm decode ran CSA sparse
attention over 96 zeroed FP8 keys. A constant perturbation that flips
near-tied tokens: `'738'`+EOS, `291→292`.

Fix (`0b5bd3d55`): never capture/restore the band (derived state = staging
quantized); restore leaves `fp8_kv_comp_packed_rows=0` and the first
post-restore decode's existing bulk pack rebuilds it from the restored bf16
staging. Paired same-binary A/B on the failing trial (`job2a-e2fx`, resend ×10,
req 1 cold + 9 warm):

| Arm | Warm exact | Signature |
|---|---|---|
| pre-fix, warm | 0/9 | `'738'` early-EOS ×9 |
| post-fix, warm | 6/9 | residual = hedging (`**738**. (Wait…`) |
| post-fix, cache OFF (same prompt) | 2/9 (3/10 incl. cold) | same hedging/truncation |

The residual is the documented pre-existing near-tied-digit solo miss floor
(signature-matched to `errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md`
Part B) — this salt is a hard instance (cache-off misses 7/10); warm restore
now performs ≥ cold recompute on the same prompt.

## Evidence lanes (committed binary, one serve per lane, `job2b_*.log`)

| Lane | Config | Result |
|---|---|---|
| E1a solo ×15, unique salts, cache ON | n=1, len 500 | **15/15 exact** |
| E1b same, cache OFF | n=1, len 500 | **15/15 exact** |
| E2 fixed prompt ×10 (clean salt) | resend, len 500 | **10/10 exact — all 9 warm**; hit_tokens=384/req, clamped=6 blocks, warm TTFT med 0.184s vs cold 0.768s (**4.19×**) |
| E2h fixed prompt ×10 (hard salt) | resend, len 500 | warm 5/10 ≥ cache-off 3/10; pre-fix 1/10 |
| E4 tiny DRAM + L3 | `--kv-dram 256MiB --kv-disk … --kv-disk-limit 8GiB` | **4/4 exact**; cold publish `kv_system_disk_pages +5` (L3 spill), warm restores read through L3 exact |
| E5a 2 identical simultaneous ×10 | concsame n=2, len 500 | 19/20 (1 miss = pre-existing `7382.` truncation) |
| E5b n=4 unique ×10 | len 2000 | 26/40 vs cache-OFF control **24/40** — pre-existing concurrent bug (n≥3, content-dependent), NOT prefix-path; same salts, same signature both arms |
| E6 n=4 unique ×15 wall | len 2000 | **60/60 exact**; wall mean **9.104s** vs same-day same-binary cache-OFF **9.136s** (−0.35%) — publish cost within noise |

E6 note: the earlier "+4.8% vs 8.68-8.72s" was a cross-day baseline — the
same-day cache-off control puts today's floor at 9.136s, so the pool is free
at this shape (and the fix also deleted ~200 KB/page of band D2H). The
one-sync-per-tick publish batching (B4) landed regardless.

## Lifecycle hardening (same commit, codex R2 findings)

- Confirm only `newly_cached` pages (deduped page ids recycle; confirming
  them left stale confirmed entries under reusable ids).
- Drop provisional pool entries on slot free/abort/preempt (new seam hook
  `release_provisional_prefix_pages`, engine snapshots the page list before
  `kv.free_slot`).
- L3 read-on-miss promote removes the superseded disk record (was
  double-resident); guarded so a disk-routed re-insert is never removed.
- `split_dram_share` floors park at one 16 MiB chunk; `--kv-dram` help
  documents the shared split.
- Publish loop documents why an MTP commit crossing a boundary must skip
  boundary capture (ring advances every token — unrecoverable).

## Rule

- Derived device state (FP8 band = quantized staging) must never be
  captured/restored — restore resets its progress counter and lets the
  existing self-heal rebuild from the source-of-truth section. Capturing it
  froze a lane-dependent pack schedule into the entry.
- A restore-corruption verdict needs a SAME-PROMPT cache-off control: the
  original E2 "control clean 8/8" ran a different salt; the failing salt's
  own cache-off floor was 3/10.
