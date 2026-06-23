# Write-through tiered KV memory on CUDA dense-Qwen3 (Phase 1+2) — pending-remote

`pending-remote`: implemented + Mac-cross-compiled (no nvcc/GPU locally). The
4×H20 flat-VRAM pod validation below is the bench/correctness gate; the human runs
it via `bin/pod`. This stub lands per the §Benchmarks rule (every runtime change →
a wins/ entry or pending-remote stub).

## Context

Implements Phase 1 (seam `KvTier` contract) + Phase 2 (CUDA dense-Qwen3) of the
**write-through tiered KV memory** model
(`docs/plans/2026-06-23-writethrough-tiered-kv-memory.md`), which supersedes the
swap (mid-decode demote/promote) model — the swap contends with the single
`KvPool` allocator, the wall both backends hit. The write-through model dissolves
it: HBM is a bounded write-through cache, the host tier (DRAM L2 → NVMe L3) is the
source of truth. Four timing rules — **write** (async mirror HBM→tier off the
decode stream), **HBM full** (drop coldest unpinned page, no write-back),
**prefill** (prefetch relevant history tier→HBM), **decode** (append + attend
resident, no synchronous tier read).

This extends — does NOT replace — the shipped per-step recall (commit `898bb979`):
that landing's reps + restricted page table ARE the write-through model's R6 reps
and decode attend-resident verb. The new parts are the seam contract and the
write-through/prefetch verbs.

## Phase 1 — seam `KvTier` contract (cross-backend, for human review)

Device-neutral trait in `crates/infer-seam/src/kv_tier.rs`, default-disabled
(`tier_capacity_pages() == 0` → host never calls the verbs → baseline
byte-identical). Three verbs named for the four timing rules + a session-keyed
block key (`TierBlockKey { session, block }`) for tenant isolation. The existing
`BackendExecutor::demote_prefix_pages`/`promote_prefix_pages` + `CudaKvTierStore`
ARE the transport — ONE session-keyed store (R5), not a parallel one. The trait is
satisfiable by both the CUDA paged pool (page == natural grain) and the Metal
windowed session KV. **Reviewed as the cross-backend API before anything builds on
it.**

## What's real vs stubbed

| Piece | Status |
|---|---|
| Phase 1: `KvTier` seam trait + `TierBlockKey` + no-tier byte-identical default | **real** (`infer-seam`, 2 unit tests) |
| Phase 2: `KvTier` impl on `QwenCudaExecutor` (delegates to the existing `CudaKvTierStore`) | **real** |
| write-through verb (`write_through`: D2H mirror a filled page → tier, keyed by `(session,block)`) | **real** (reuses `copy_pages_to_host` + `tier.insert`) |
| prefetch verb (`prefetch`: H2D tier → fresh device pages) | **real** (reuses `promote_prefix_pages`/`copy_pages_from_host`) |
| `tier_block_u64` key flattening (write-through high-bit namespace, never aliases prefix-tier keys, per-session isolation) | **real** |
| Decode attend-resident (restricted page table, resident variant) + R6 reps + `q·rep` scoring → `plan_recall` | **real** (shipped `898bb979`, reconciled to write-through framing) |
| infer-core prefetch-selection policy (`prefetch_query` R3, `prefetch_blocks`, `plan_working_set`) | **real** (`writethrough.rs`, unit-tested) |
| infer-core eviction policy (`evict_drop_pages` LRU+sink/local pins, `cap_rep_pool` R1) | **real** (`writethrough.rs`, unit-tested) |
| `evict_drop` verb — actual mid-decode **device-page free** (the flat-VRAM win) | **stubbed** (blocker below) |
| Engine-driven prefill prefetch wired to `session_id` (turn-based recall) | **stubbed** (blocker below) |
| GPU correctness + flat-VRAM numbers | **pending-remote** (this doc) |

### The remaining blocker (honest, no self-deception)

The **flat-VRAM win** needs the non-resident middle pages freed OUT of HBM. The
host `CudaKvPool` is the single page allocator and the executor re-publishes the
slot's full contiguous page table every decode step via `mirror_slot`, guarded by
the `SlotProgress` continuity watermark. Freeing a live slot's middle device page
collides with that — the same single-allocator wall the prior recall landing
documented. So `evict_drop` is a no-op backend hook (the page's tier copy from
`write_through` remains the source of truth) and the resident variant — full KV
stays in HBM, decode attends a budget-bounded restricted page table — is what
ships. The write-through mirror + prefetch transport are real and live; the
device-page lifecycle ownership move is the next increment.

Dissolving it (next pass): give the executor a post-decode device-page-free path
that the host pool honors (e.g. a `truncate-middle` allocator op + epoch bump), or
move the slot page table to executor-owned so `mirror_slot` does not re-publish the
full contiguous list. Both are allocator-contract changes, out of the
byte-identical-default scope here.

`session_id` is carried on the request (`infer-api/types.rs:104`) but advisory —
not threaded into the engine/radix. The engine-driven prefill prefetch (score
history → `prefetch` tier→HBM at each turn) needs that plumbing; the policy
(`infer_core::prefetch_blocks`/`prefetch_query`) is ready and unit-tested, the
wiring is deferred.

## Files changed

- `crates/infer-seam/src/kv_tier.rs` — **new**: `KvTier` trait, `TierBlockKey`,
  no-tier default + 2 unit tests.
- `crates/infer-seam/src/lib.rs` — register `kv_tier` module; export `KvTier`,
  `TierBlockKey`.
- `crates/infer-core/src/writethrough.rs` — **new**: device-neutral policy —
  `prefetch_query` (R3), `prefetch_blocks`, `plan_working_set`, `evict_drop_pages`
  (LRU + sink/local pins), `cap_rep_pool` (R1) + 11 unit tests.
- `crates/infer-core/src/lib.rs` — register `writethrough` module + exports.
- `crates/infer-cuda/src/executor.rs` — `tier_block_u64` key flattening;
  `QwenCudaExecutor::write_through`/`prefetch_pages`; `impl KvTier for
  QwenCudaExecutor` (verbs delegate to `CudaKvTierStore`; `evict_drop` no-op per
  blocker).
- `crates/infer-cuda/src/recall.rs` — reconcile module + `TODO` docs to the
  write-through framing (reps = R6 write-through-time reps; restricted page table =
  decode attend-resident verb).
- `crates/cli/src/args.rs` — `--kv-recall` docstring now names the write-through
  model + plan.

## Pod test plan — flat-VRAM-vs-history (4×H20, run via `bin/pod`)

Run on the **4×H20** allocation: `INFER_CUDA_DEVICES=4,5,6,7` (TP=4), dense
**Qwen3.5 bf16** (the paged-KV arm; recall is BF16-only — do NOT pass
`--kv-cache-dtype int8/fp8`, it falls back to full attention, logged once).

> NOTE on what this stage proves: this pass ships the **resident variant** — full
> KV stays in HBM, decode attends a bounded restricted page table. So **VRAM does
> NOT yet flatten** (the device-page free is the documented blocker). What flattens
> NOW is the **per-step decode KV-read volume / attention cost** (bounded
> working set regardless of session length). The full flat-VRAM curve is the
> `evict_drop`-device-free increment. Run the test to confirm (a) correctness under
> recall and (b) the decode-cost flattening; record the VRAM curve as the baseline
> the device-free increment must then bend.

### 1. Serve (recall ON, 4×H20 TP=4, eager decode)

```bash
# Recall is eager-only; the engine skips the captured graph for recall-active
# slots, but disable it explicitly so the path is unambiguous.
INFER_CUDA_DEVICES=4,5,6,7 INFER_CUDA_DECODE_GRAPH=0 \
arle serve --backend cuda \
  --kv-recall \
  --kv-cache-dtype bf16 \
  --model-path <Qwen3.5-dense-bf16 checkpoint> \
  --port 8000
```

Baseline control (recall OFF, same binary/shell/model/devices): drop `--kv-recall`.

### 2. Correctness — long-context needle (correct-inference gate, NOT byte-identity)

Recall + restricted attention deliberately deviates from a single full-KV run →
the gate is **needle retrieval = the full-attention answer**, not token-exact vs
baseline (`feedback_correct_inference_not_baseline_identity`). Plant a passkey at
mid-depth in a prompt LONGER than the working-set budget (sink 32 + local 256 +
8×32 = **544 tokens** with the shipped `default_recall_config`); e.g. a ~6K-token
context, passkey at depth 0.5.

```bash
curl -s localhost:8000/v1/completions -d '{
  "model": "<id>", "session_id": "needle-1",
  "prompt": "<~6K-token filler with PASSKEY 84213 at depth 0.5> ... What is the passkey?",
  "max_tokens": 32, "temperature": 0
}' | jq -r '.choices[0].text'
```

Pass = the recalled answer contains the planted passkey (= the full-attention
answer). Run x3 same-config repeats (needle ladder vs the baseline envelope) to
absorb MoE/run-to-run non-determinism. A streaming control (sink+local, no recall)
should MISS at the same budget — that's what makes recall load-bearing.

### 3. Flat-VRAM / decode-cost vs history (the §6 decisive evidence)

Drive ONE session far past the HBM working-set budget (grow it turn-by-turn to 4K,
8K, 16K, 32K tokens using the same `session_id`), recording per-step decode ITL and
`nvidia-smi` VRAM for both arms:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv -l 1 &   # snapshot both arms
scripts/bench_guidellm.sh cuda-writethrough-on \
  --model <Qwen3.5-dense-bf16> --extra-serve-args "--kv-recall --kv-cache-dtype bf16"
scripts/bench_guidellm.sh cuda-writethrough-off-baseline \
  --model <Qwen3.5-dense-bf16>
```

**Assertions:**
- **VRAM**: at THIS (resident) stage the two arms' VRAM curves OVERLAP and both
  grow with session length (full KV resident). Record the curve — it is the
  baseline the `evict_drop`-device-free increment must flatten. (When that lands,
  recall ON holds flat while OFF grows → the win.)
- **Decode ITL / tok-s**: recall ON should hold ~flat once the session passes 544
  tokens (bounded working set); recall OFF rises with `cache_len`. This is the
  measurable win of the resident variant.
- **Needle**: the early-planted needle is still retrieved at the longest session
  length under recall ON (§2 re-run at 32K) — proves prefetch-by-relevance, not a
  fixed window.

### 4. Default-off regression (mandatory, byte-identical)

Confirm recall OFF is unchanged: §3 baseline arm with the captured decode graph ON
(default), tok-s matches the latest dense-Qwen3 CUDA baseline wins entry within
noise (the recall code is behind `if !self.kv_recall` and the graph path is
untouched). Multi-tenant: two concurrent `session_id`s must never cross-retrieve
(distinct `TierBlockKey.session` → distinct `tier_block_u64` namespaces).

## Gates (local, Mac — no nvcc)

- Mac CUDA typecheck: `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release
  --no-default-features --features cuda,no-cuda --lib` → clean.
- `cargo test -p infer-core` → 77/77 (was 66; +11 write-through policy tests).
- `cargo test -p infer-seam` → 26/26 (+2 `KvTier` tests).
- `cargo clippy -p infer-cuda` (cuda,no-cuda) → no findings in changed files
  (7 pre-existing `attention.rs` rust-1.95 lints, untouched).
- CPU/no-cuda build (`agent-infer` cpu,no-cuda,cli) → clean (ungated seam trait +
  policy module compile without the cuda feature).

## Rule

Write-through ≠ swap: HBM is a bounded write-through cache, the tier is the source
of truth. The four timing rules dissolve the single-allocator blocker by making
eviction a free-with-no-write-back and confining tier reads to prefill. The CUDA
paged pool is the natural grain (page == mirror/evict/prefetch unit); the
write-through mirror + prefetch + decode-attend-resident are real, the mid-decode
device-page free (the flat-VRAM win) is the documented next increment — never claim
the VRAM flatten before the device-free path lands.
