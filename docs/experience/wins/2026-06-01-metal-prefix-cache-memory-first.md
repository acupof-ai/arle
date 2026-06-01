# Metal prefix cache — memory-first + budget-gated async SSD persist (kill the replay stall) — 2026-06-01

## Context — proven root cause (instrumented earlier, not re-derived)

Single-machine c=1 long-context Metal first-response was ~4× slower than
mlx-lm. Earlier instrumentation (NOT source inference) pinned it: the stall
is between token1 and token2, in a **synchronous SSD prefix-cache publish on
the scheduler thread**.

`publish_prompt_prefix` (runtime.rs) called `export_qwen35_disk_prompt_prefixes`
→ `stream_prefix_snapshots_at_lengths`, which built a **fresh replay
`Qwen35StepDriver` and re-prefilled the WHOLE prompt in 16-token blocks** just
to mint block-aligned SSD snapshots. Measured `export_us` on the serving thread:

```
512 tok → 1.9s   2k tok → 7.0s   8k tok → 30.5s
```

This is the token1→token2 stall. M_e.13 SSD cache is default-on, so every
first-seen long prompt paid it once.

## What changed — Option A (memory-first; runtime.rs + request_state.rs only)

1. **Memory-first, zero replay on the serving path.** `publish_prompt_prefix`
   now only takes the cheap full-length in-memory snapshot
   (`export_qwen35_live_prefix_snapshot` → `export_drained_prefix_snapshot`,
   a direct clone of the already-resident KV+GDR — no replay) and inserts it
   into the host-RAM LRU. The synchronous replay disk-publish is deleted.

2. **Budget gate for SSD.** `worth_persist = prefill_cost_us > READBACK_US_PER_TOKEN × tokens × SAFETY`,
   `prefill_cost = first_token_at − admitted_at`. Defaults
   `READBACK_US_PER_TOKEN=48` (seeded from M_e.13 measured
   `read_us+decode_us+import_us` ≈ 48µs/tok at 2064 tokens) and `SAFETY=2.0`,
   env-tunable via `INFER_METAL_PREFIX_READBACK_US_PER_TOKEN` /
   `INFER_METAL_PREFIX_PERSIST_SAFETY`.

3. **Async persist = off the serving thread.** Serving thread does the cheap
   snapshot + budget gate + `encode_for_disk` (the `eval`/`to_bytes` **stays on
   the serving thread** for MLX safety — no global MLX lock, dedicated GPU
   streams, see `feedback_mlx_async_eval_is_caller_thread`). The encoded
   `Vec<u8>` + key + fingerprint is sent over an `std::sync::mpsc` channel to a
   new dedicated `std::thread` (`metal-prefix-ssd-writer`) that does only
   `put_disk_block_with_fsync` + index bookkeeping. The disk index
   (`disk_entries` + `disk_bytes`) moved behind `Arc<Mutex<MetalDiskPrefixIndex>>`
   shared between the serving thread (lookup/import/reconcile) and the writer
   (persist). The lock is never held across the disk syscall or any MLX `eval`.

4. **Dead code deleted (no half-states).** Removed `stream_prefix_snapshots_at_lengths`,
   `export_qwen35_disk_prompt_prefixes`, `qwen35_disk_publish_prefix_lens`,
   `longest_reusable_aligned_prefix_len`, `export_current_cpp_snapshot`,
   `drain_replay_after_result`, the old `persist_snapshot` + `ensure_disk_capacity_for`
   + `touch_disk`, and their tests. Import lookup kept at strict `<`.

### Accepted tradeoff
SSD warm-restart reuse of **shorter block-aligned prefixes** depended on the
deleted replay and is intentionally dropped (user-approved: "短前缀直接丢弃").
**In-memory warm reuse of the full-length snapshot is retained** (the M_e.13
multi-turn win). The block-alignment guard on persist/reconcile was also dropped
because the live full-length snapshot is at `cache_len` (GDR state can't be
truncated) and need not be block-aligned — only `token_ids.len() == cache_len`
is required by the on-disk format. A future prompt that *extends* this prefix
can import it across restarts.

## Verification (live, Qwen3.6-35B-A3B-4bit, M-series, c=1)

`metal_serve --max-running-requests 1 --max-batch-tokens 4096`, temp 0.

### Correctness gate — PASS
- Needle ("The vault code is 73519." mid a ~6k-token prompt) → returned
  `73519`, coherent (elapsed 7.8s).
- Short factual prompts → coherent (Qwen3.6 reasoning-mode traces, no garbage).

### Phase-B core metric — token1→token2 stall killed

| Prompt | token1→token2 BEFORE (measured `export_us`) | token1→token2 AFTER | steady TPOT AFTER | TTFT AFTER |
|---|---|---|---|---|
| 2k | ~7.0s | **124ms** | 12.0ms | 3.14s |
| 8k | ~29.6s (≈30.5s `export_us`) | **487ms** | 14.5ms | 13.9s |

Both well under the <1s target. Steady TPOT unchanged (~12–14ms). TTFT
(prefill→token1) not regressed — it never contained the replay. The 8k
after-gap (487ms) is the serving-thread `encode_for_disk` of the ~290MB
full-length snapshot before the channel hand-off; the disk write itself
(138ms) is off-thread.

### Async SSD writer — works end-to-end
8 budget-passing publishes → 8 new `.kv` files written by the writer thread:
```
m_e13_trace ssd_writer persist: tokens=10799 payload_bytes=289853343 write_us=138562
m_e13_trace ssd_writer persist: tokens=5394  payload_bytes=...        write_us=...
```

### Memory warm reuse (multi-turn extension) — retained, 53× faster
Two-turn session (turn2 extends turn1's prompt):

| Turn | resume_prefill_tokens | TTFT | E2E |
|---|---|---|---|
| turn1 (5394-tok ctx, cold) | 5394 (full prefill) | 11491ms | 11.78s |
| turn2 (extends turn1) | **18** (matched 5394) | **132ms** | **0.22s** |

Trace: `memory_match_len=Some(5394) … matched_tokens=5394 request_hit_rate=1.0`.

## Honest finding — the budget gate does NOT filter short prompts on this model

The user-approved formula + measured defaults (`48µs/tok × 2`) passes
`worth_persist=true` for **essentially every prompt ≥ block_size (16)**,
including a 339-token prompt (`prefill_cost_us=906682` vs
`readback_cost_us=32544`). Reason, by measurement: on Qwen3.6 a re-prefill
costs ~1.5–3ms/token while snapshot readback is ~48µs/token, so re-prefill is
always 30–60× more expensive per token. The premise "short prompts have cheap
prefill relative to the readback threshold" is **falsified** for this model.

So the actual filtering is: (a) prompts `< 16` tokens get no snapshot at all
(block-size floor) — verified a ~10-token prompt produces no publish; (b) the
budget gate, with measured defaults, lets everything else through because
re-prefill genuinely costs more than readback. Operators who want to suppress
short-prompt persists can raise `INFER_METAL_PREFIX_READBACK_US_PER_TOKEN`
(~1000 would gate out 256-tok prompts). The formula is implemented exactly as
specified; this entry records that the "256-token must not write" expectation
in the plan rests on an intuition the measurement contradicts, per §0
(推断 ≠ SOLID; wall-clock measurement is ground truth).

## Bug caught during verification
First implementation deadlocked at runtime/test teardown: `DiskWriteHandle::drop`
joined the writer thread **before** dropping the channel sender, so the writer's
`rx.recv()` never returned. Fixed by dropping the sender first
(`self.tx.take()`), then joining. A single unit test hung 4 min until killed,
which surfaced it.

## Rule
Killing the serving-thread replay is the whole win — the per-token-1→2 stall
collapses from seconds to tens-to-hundreds of ms. When moving work to a writer
thread, drop the channel sender before joining or the join deadlocks. And a
"budget gate" that compares prefill vs readback per token won't filter short
prompts when prefill is uniformly more expensive than readback — measure the
gate's actual decisions before claiming it filters anything.

Cross-link: root cause from the earlier instrumented session; supersedes the
replay-based disk publish in `2026-05-08-bench-m_e13-ssd-persistence-c1-win.md`.
