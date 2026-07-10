# DSpark per-layer draft-KV cap — sliding-window ring, 2560→544 MB/slot

> Status: Shipped — LICENSED (pod H20 GPU1: byte-correct + −4.7× draft VRAM)

## Context

DSpark reserved `cap = max_seq_len + block` draft K/V for **all** draft layers,
but 4/5 z-lab DFlash layers are `sliding_window`(2048) and, post-P2.5
([partial-ctx drafting](2026-07-10-dspark-partial-ctx-drafting.md)), only ever
read the last `window` keys. That over-reservation (~2560 MB/slot when ~600
would do) clamped concurrent slots on shared/loaded GPUs. Commit `23fa8f3e2`.

## What Worked

**Per-layer cache stride, scheme A (absolute-position ring).** Each draft ctx
K/V buffer is sized by layer type:

- **sliding (4×)**: `cap_sw = window + block`, addressed as an absolute-position
  ring — `row = pos % cap_sw`. Keeps only the last `window` keys the sliding
  attention ever reads.
- **full (1×)**: `cap_full = max_seq_len + block`, linear — `row = pos − ctx_base`
  (unchanged).

The RoPE cos/sin cache stays full `max_seq_len` (positions are absolute).

**No kernel rewrite, only a `ring_modulus` param.** The existing prep-write and
nonpaged-attention kernels already parameterize the per-head stride. Added a
`ring_modulus` arg to both `__global__` kernels + two new `*_ring_cuda` extern
wrappers; the existing wrappers pass `ring_modulus=0` → byte-identical for every
trunk / decode / autograd caller. Softmax is permutation-invariant, so a wrapped
ring read accumulates identically to a contiguous one — only the physical row
differs. Chosen over scheme B (per-layer advancing base): no per-layer base
state, no per-step compaction copies.

**Why `cap_sw = window + block` is the minimum (no live aliasing).** Within one
draft block the live span = `window` ctx keys + `block` noise keys = `cap_sw`.
Two distinct absolute positions `p ≠ q` with `|p−q| < cap_sw` never collide mod
`cap_sw`, so a noise write never overwrites a live ctx key still needed in that
block. One append launch writes `≤ cap_sw` rows (guarded by `ensure!`) → no
intra-launch write race; chunk sizes (≤32) sit far below `cap_sw`.

### §0.1 buffer table (draft K/V, per layer type)

| buffer | type | per-head stride | writer | index formula | read span | aliasing precondition |
|---|---|---|---|---|---|---|
| `k_ctx[li]`/`v_ctx[li]` | full 1× | `max_seq+block` | `prep_cuda` | linear `abs − ctx_base` | `[ctx_base, start+block)` | `abs−ctx_base < cap_full` (ctx grows ≤ max_seq) |
| `k_ctx[li]`/`v_ctx[li]` | sliding 4× | `window+block` | `prep_ring_cuda` | ring `abs % cap_sw` | `[max(ctx_base, q_pos−win+1), start+block)`, read via `nonpaged_ring` row `(lo+l) % cap_sw` | live span (win ctx + block noise) = `cap_sw`; distinct `p≠q`, `|p−q|<cap_sw` never collide mod `cap_sw`; one launch writes `≤cap_sw` rows |

**Byte-identity (memory fix, not accuracy).** For `ctx_base==0`, `abs < cap_sw`:
ring row `abs % cap_sw == abs ==` old linear row, same value; the per-row read
visits the same key set in the same ascending-abs order → attention output
byte-identical wherever the old linear code was correct (any `q` whose window
fits). For `ctx_end > cap_sw` the old buffer kept spare rows but the sliding read
still only took the last `window` — identical keys, still byte-identical; the new
code additionally frees the never-read prefix (and fixes the region where the old
linear buffer would have overflowed at `ctx_end − ctx_base > cap_full`).

### Per-slot draft VRAM

`before = 2·5·kv·hd·cap_full`; `after = 2·kv·hd·(4·cap_sw + cap_full)`.
Measured on H20 GPU1 at the KV-pool floor `max_seq_len = 131072` (128K,
`kv=8`, `hd=128`, `block=16`) — the `draft {}MB` line at `qwen35.rs:2210`:

| metric | before | after | Δ |
|---|---|---|---|
| draft bytes/slot | **2560 MB** (analytic, same config) | **544 MB** (measured) | **−4.7× (−2016 MB)** |
| per-slot total | ~2707 MB | 691 MB | 691 = K/V 0 + gdr 144 + conv 2 + draft 544 |
| affordable slots (18965 MB free) | ~6 | **24** | 4× more concurrency |

Note the earlier "~640" estimate was loose (used `max_seq≈32K`); the real pool
floor is 128K, so `cap_full` is larger and the measured after is 544 MB (the
4 sliding layers collapse to ~2064 rows each, near-zero vs the 128K full layer).

## Verification

- **Local gates (green):** `infer-api` cuda,no-cuda + `cli` metal,no-cuda
  typecheck; `clippy -D warnings` clean on `infer-cuda`+`cuda-kernels`;
  `arle` cpu tests pass.
- **Pod license — LICENSED** (H20 GPU1, `arle serve --backend cuda --model-path
  Qwen3.6-27B-FP8 --spec-type dspark --mtp-draft-model Qwen3.6-27B-DFlash`,
  drafter `dflash-backbone block=16 taps=[1,16,31,46,61]`; dspark is single-GPU,
  the binary rejects TP for it):
  - Build `BUILD_EXIT=0` (27 crates); `strings arle | grep dspark-sp` → hit.
  - **Correctness (byte-correct sliding attention):** needle 738291 greedy,
    exact **3/3** at ctx 1000 / 4000 / **8000 (> window, ring wrap exercised)**.
    Warm re-runs byte-identical (DET); the one cold-8000 NONDET was a leading
    `</think>` template token on run0 only — trunk prefix-cache boundary, not the
    ring (warm pair already identical).

### Concurrency c-sweep (deferred #17) — Qwen3.6-27B-FP8, ctx=1000, max_tokens=256, eager, errs=0

| c | dspark tok/s | dspark p50 | plain tok/s | plain p50 | dspark active_req |
|---|---|---|---|---|---|
| 1 | **71.9** | 3.56s | 34.8 | 7.35s | 1 |
| 4 | 72.9 | 14.04s | 73.1 | 13.62s | 3 |
| 8 | 80.6 | 25.41s | **127.0** | 14.82s | 7 |

- **C=1: dspark 2.07× plain** single-stream (spec-decode latency win).
- **dspark aggregate is flat (72→73→81); plain batches 3.65×.** The scheduler
  admits all requests (`active_requests=7` at C=8) — flatness is draft/verify
  per-step overhead scaling with batch, not queueing. Crossover ~C=4; at C=8
  dspark loses on both tok/s (0.63×) and p50. No OOM/timeout, `kv_free_pages`
  7100–8100 (no KV pressure).
- **Reading (SCOPED — corrected against the DSpark paper, arXiv:2607.05147):**
  what does not batch is **OUR config** — DFlash-backbone-only, static
  `conf` threshold, fixed block=16, and NO hardware-aware scheduler. That is
  exactly the "indiscriminate fixed-length verification wastes batch capacity"
  failure mode the paper's §1/§3.2 names as the problem it solves. DSpark *as
  designed* is built to hold throughput at high concurrency via the confidence
  head + **Hardware-Aware Prefix Scheduler (Algorithm 1)**, which shrinks each
  request's verify length ℓ against a profiled `SPS(B)` curve to maximize
  `Θ = τ·SPS(B)` — DeepSeek-V4 production reports 57–85% per-user speedup *at
  matched aggregate throughput* and specifically unlocking strict-SLA tiers.
  We run neither the scheduler nor a trained confidence head (NO-LICENSE here),
  so our flat aggregate is the DFlash baseline, not a DSpark-with-scheduler
  verdict. **For OPD (B=1 rollout) the config is right and the 2× stands.** To
  get the concurrency leg would take implementing Algorithm 1 + a trained
  confidence head (P3) — not attempted; gate dspark off above c≈4 *for this
  config*. The per-layer memory fix makes it affordable at all (24 vs ~6 slots).

## Rule

- Size per-slot caches by what each layer **reads**, not a uniform pool cap — a
  sliding layer needs only `window+block`; the ring modulus = that stride keeps
  the existing contiguous-cache kernels usable via a single `ring_modulus` param.
- `cap = window + block` is the minimum with no live-key aliasing: the live span
  (window ctx + block noise) exactly equals the modulus, so distinct positions
  within a block never collide.
