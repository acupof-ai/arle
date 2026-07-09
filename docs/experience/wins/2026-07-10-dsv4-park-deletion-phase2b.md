# DSv4 whole-slot park deleted (Phase 2b) — preemption rides the prefix-state pool, pod gate PASSED

> Status: **Shipped** — commits `61c06b187` (deletion, −869 LOC) + `fc850c7c6`
> (preempt-requeue counter). 4×H20 (GPUs 4-7), TP4, DeepSeek-V4-Flash-FP8,
> `--max-total-tokens 2048`, binary sha256 `971dacd6602a…` @ tree `fc850c7c`,
> logs `job2c_*.log` on the pod build tree.

## Context

Plan: `docs/plans/2026-07-09-dsv4-kv-reuse-seam-refactor.md` Phase 2b. The
whole-slot park (`demote_slot`/`promote_slot` via `Dsv4SlotSnapshot`/
`Dsv4LayerImage` 16 MiB blobs) was the last Route-B serialization consumer —
a second position-exact capture path parallel to the 2a content-keyed
prefix-state pool. Deleted: `Dsv4LayerImage`/`Dsv4CompressorImage`/
`Dsv4FlashMlaImage`/`Dsv4DsaOfficialImage` + capture/restore, slot/layer
`swap_{out,in}_image`, `mirror_restore_pages`/`mirror_slot_pages`,
`flashmla_set_band_cursor`, the DSv4 `slot_tier` store, and 2a's B6 50/50
`split_dram_share` — the prefix pool now takes the whole `--kv-dram` share
(and the whole `--kv-disk` cap). Qwen3.6's park is untouched.

## Planner-fallback trace (engine-side, code-verified)

- `kv_slot_tier_enabled()` → false gates BOTH oversubscription surfaces:
  the engine P5 loop (`lib.rs:1094`) never runs, and `--kv-oversubscription`
  fails loud at boot (`loaded.rs:1923`). `try_park_for_oversubscription`'s
  PARK-OR-NOTHING contract is preserved: a refused demote leaves the victim
  running — no dead-end, no 2026-07-05-class livelock.
- KV-overflow retract (`requeue_preempted_decode_with_bias`) never requires
  demote success: it falls to `reset_for_recompute()` — generated tokens are
  CLEARED and regenerated after re-admission through the standard
  prefix-attach path (2a pool restores the prompt prefix; publish already
  happened at prefill-seal, `lib.rs:888`, with its one in-call sync — so
  preempt-time serialization is zero, publish-is-the-demotion). Confirmed:
  the requeue does NOT carry generated tokens as a prompt extension.
- **Pre-3b, capacity preemption is structurally unreachable for DSv4**:
  fixed bands (`fixed_pages_per_slot`) make `append_pages_needed()` = 0 for
  occupied slots, so `retract_decode_to_fit` never fires. Measured: 0
  preempt-requeue events across every lane (`fallback_recompute` counter +
  log line added in `fc850c7c6` for 3b's gate, where demand paging makes
  the path reachable).

## Evidence lanes (one serve each, ports 18220-18224 after a foreign-serve
## collision on 18211 — port preflight added to the harness)

| Lane | Config | Result |
|---|---|---|
| L0 flag reject | `--kv-oversubscription` | serve exits rc=1 with "no whole-slot tier" — **PASS** |
| E1 solo ×15, unique salts, cache ON | n=1, len 500 | **15/15 exact** (wall 0.84–1.01 s) — matches 2a E1a 15/15 |
| E2 fixed prompt ×10 | resend, len 500 | **10/10 exact**; warm TTFT med 0.182 s vs cold 0.761 s = **4.18×** (2a: 4.19×), hit_tokens 384/req — pool-takes-whole-share budget change is regression-free |
| QP wave1 | `--max-running-requests 4`, N=12, len 1200 | **12/12 exact** — queue + slot turnover clean |
| QP wave2 | same, N=24, len 500 | 17/24 vs same-salt same-binary cache-OFF control **15/24** — misses are the pre-existing concurrent-decode bug (`'738.'` truncation + `291→292` digit subs, BOTH arms; `errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md`), NOT the 2b path |

Server survived every lane; zero preempt events (expected); no `slot_tier`
lines in any serve log.

## Rule

- A "fallback path" claim needs the caller-by-caller trace: DSv4's park
  fallback splits into an INERT surface (oversubscription — gated off, fail
  loud at boot) and a LIVE one (KV-overflow requeue) that only 3b's demand
  paging can actually reach. Gate the reachable one when it becomes
  reachable; measure the unreachable one at 0.
- Foreign serves share the box's port space: every lane harness must
  preflight its port (`curl /v1/models` answering before launch = FATAL) —
  needle "hits" from another session's Qwen3.6 looked plausible for 4 reps.
