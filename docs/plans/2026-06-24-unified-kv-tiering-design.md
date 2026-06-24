# Unified model-agnostic L1/L2/L3 KV tiering — design of record

**Goal (ckl):** 扩充 KV cache 用量 — turn 850 GB DRAM (L2) + 380 GB NVMe (L3) into usable KV
for **all** models under **one** model-agnostic architecture, not per-model patches.

Produced by a 20-agent judge-panel workflow (`unified-kv-tiering-design`, 2026-06-24): 3 code
mappers → 4 independent unified designs → 3-lens adversarial judges → synthesis. It **corrected
several premises** (see §6) and found the unified abstraction **already exists in the seam** — the
work is to NAME + WIRE it, not to add a new abstraction.

## 1. The core — Tier-Tagged Radix + a 3-grain Restore Contract (already in the seam)

The model-agnostic unit lives in `infer-core` (radix.rs/prefix.rs/planner.rs), reached only through
the two host-only traits `infer_seam::{BackendExecutor, KvPool}`. Two pieces, both present:

- **Tier-tagged radix** (`radix.rs:52`) = the SGLang HiCache node: each block carries
  `page_id: Option<BlockId>` (L1 resident) + `tier_key: u64` (L2/L3 demoted);
  `PrefixBlock::{ResidentPage,DemotedKey}` (`radix.rs:47`) is the discriminant. Model-agnostic by
  construction — only host page ids + opaque u64 keys, never device memory.
- **The correctness gate** = `BackendExecutor::reusable_prefix_blocks` (`infer-seam/lib.rs:186`),
  default **fail-closed 0**, doc enumerating the failure class ("recurrent state, ring cursors,
  compressor metadata, mirrored snapshots"). `clamp_prefix_to_backend` (`prefix.rs:38-54`)
  **truncates** every radix match to this count before any promote/attach. **GDR-reuse is
  impossible-by-default, not silently-wrong — correctness is a return value, not a per-model patch.**

The unification: the seam already has **three restore grains** — recognize they are ONE concept,
*"the smallest unit at which a backend can resume"*:

| Grain | Verbs (already in seam) | Semantics | Use |
|---|---|---|---|
| **G1 PAGE-REATTACH** | `reusable_prefix_blocks` + `demote/promote_prefix_pages` (`executor.rs:919/936`) | position-independent KV pages | arbitrary-mid-prefix **cross-request reuse** |
| **G2 POSITION-0 IMAGE** | `cached_prefix_match_len`/`capture`/`restore_cached_prefix` (`lib.rs:283-326`) | `[0,len)` prefix image (pages + position-locked sidecars) | **pos-0 reuse** (leading prefix only) |
| **G3 WHOLE-SLOT IMAGE** | `kv_slot_tier_enabled`/`demote_slot`/`promote_slot` (`lib.rs:258-281`) | complete per-request image (KV + ALL sidecars) | same-request **capacity spill** |

All three move bytes through the **same** `CudaKvTierStore` (`kv_tier.rs:158`, DRAM-L2 BTreeMap →
NVMe-L3 spill) by opaque u64 key. **The engine orchestrates by key + length only, never sees a
device type.** GDR/MLA/conv state rides as **opaque sidecar bytes below the seam** in the backend's
own demote/promote — the engine cannot memcpy model state (that would cfg-leak backend types into
device-neutral infer-core, violating the CRITICAL isolation rule).

## 2. Why this design (and why NOT the alternatives)

`infer-core` **never branches on model type** — dispatch is purely on capability RETURNS
(`reusable_prefix_blocks()` count, `kv_slot_tier_enabled()` bool, `cached_prefix_match_len()` len).
The two reuse routes merge by a model-blind `.max()` (`lib.rs:971-977`). Baseline is byte-identical
when the returns are the fail-closed defaults — a model that implements nothing is exactly today.

**Rejected: a unifying `StateBlob`/`SoundnessTag`** (the capability-descriptor angle, adversarially
scored 6). A binary FullAttnReusable/CapacityOnly tag **cannot represent DSv4's actual shipped
capability** (position-0-only reuse) → adopting it would **regress** live DSv4 reuse and create a
dual-source-of-truth (if the tag disagrees with `reusable_prefix_blocks`, which wins? — the exact
silent-wrong risk this design exists to prevent). **The genuine boundary IS the existing host-only
traits; the 3-grain contract is the unification, one level down, already there.**

## 3. Per-model (what each gets, honestly)

- **Dense Qwen3** — G1, the reference. EMPTY sidecar (zero side state). Gets BOTH: cross-request
  page-radix reuse (shipped) + per-page capacity spill. **No new code.**
- **Qwen3.6 hybrid** — G3 (capacity, NEW) + G2 (pos-0 reuse, NEW). Sidecar = the per-linear-layer
  recurrent slab: `gdr_states` (`qwen35.rs:366`) + conv1d ring (`:367`), content-based/sequential,
  NOT page-addressable → `reusable_prefix_blocks=0` (mid-prefix reuse structurally impossible).
  Same-request spill carries the recurrent state verbatim → byte-exact resume (no reuse
  unsoundness). **Does NOT get** mid-prefix page-radix reuse (physics) — only coarse pos-0.
- **DSv4 MLA** — G2 (pos-0 reuse, **already ships**, `executor.rs:2338-2440`) + G3 (capacity,
  **already ships**, `demote/promote_slot` `:2245-2334`). The MLA latent **IS paged** (TokenKVPool,
  block-table-addressed — §6 premise correction). Sidecar = the position-locked mirror set (SW
  ring, compressor/indexer rings, DSA RoPE-rotated keys). **HARD GAP:** G2+G3 are **disabled at
  `world_size>1`** (`executor.rs:2246-2260` — multi-rank lockstep demote unimplemented → NCCL
  deadlock). So the **production 8×H20 TP8/EP8 DSv4 gets the paged prefix tier ONLY** until Phase 6.

## 4. Write policy + prefetch (corrected — ckl's write-back was right)

- **HBM→L2 = WRITE-BACK** (lazy, on eviction). SGLang's eager per-page write-through is **rejected
  for ARLE** — a continuous device→host mirror on the hot path is too expensive. This matches ckl's
  decided design ([writethrough-tiered-kv-memory](2026-06-23-writethrough-tiered-kv-memory.md)).
  *(Correction: an earlier claim that ARLE should switch to write-through was wrong — the mirror
  cost dominates; write-back is correct here.)*
- **L2→L3 = WRITE-THROUGH** (DRAM full → spill coldest to NVMe, `kv_tier.rs:539`).
- **Sidecars (G2/G3)** — write-back at the spill/finish moment ONLY (a recurrent blob mutates every
  decode step; eager mirroring is pure waste; the D2H happens once, on a slot being freed anyway).
- **Prefetch** — the ONE tier read is at the single batched prefill sync point (G1
  `promote_prefix_pages`, G2 `restore_cached_prefix`, G3 `restore_swapped_slot`); never during
  decode. **256-token threshold** (SGLang revoke-below): a demoted match shorter than 256 tok is
  recomputed, not round-tripped. L3→L2 background staging + L2→L1 layer-ahead are **DEFERRED
  perf-only**, gated behind an nsys trace proving the prefetch sync actually stalls prefill.

## 5. Decode zero tier I/O (structural, three guarantees)

(1) An active slot's pages are `lock_ref`-pinned (`radix.rs:427`) → `attach_count>0` is never in
`lru_evictable_pages` → tier eviction can never select a decoding page. (2) The seam gives decode
**no tier-read verb** — a decode-time tier read is unexpressible. (3) Sidecars are read/written
in-place every step, L1-resident the whole active lifetime; D2H/H2D happens only at spill (request
leaving decode) and re-admission — both admission-time. A spilled request is by definition not
decoding.

## 6. Premise corrections from the map

- **DSv4 MLA IS paged** (TokenKVPool, `KVFormat::PackedBytes`, page=64 tok, 584/656 B/tok,
  `attention.rs:235`; shared free-VRAM pool sized total) — not the per-slot arena assumed.
- **DSv4 already ships G2 + G3** (single-rank) — the position-identity invariants (RoPE/SW-ring/DSA)
  the naive designs called "future work" are already coded.
- **The unified mechanism already exists** — Phase 0 is documentation, not new abstraction.

## 7. Phased implementation

| Phase | Scope | What | Gate |
|---|---|---|---|
| **P0** | docs | NAME the unification (tier-tagged radix + gate + 3-grain contract); REJECT StateBlob/SoundnessTag | bench-exempt |
| **P1** | shared | dense eviction-path `write_through` D2H → async (side stream, drain at prefill sync); fire ONLY under page pressure, never steady-state decode | **license-or-kill** matched A/B (sync vs async) under SLO workload, decode tok/s no regress; needle ×3 |
| **P2** | shared | collapse the scattered capability queries → ONE cached read, branch on the 3 grains (deletion-refactor; bodies unchanged) | `cargo test` infer-core/seam; baseline byte-identical |
| **P3** | **Qwen3.6** | capacity G3: `kv_slot_tier_enabled`/`demote_slot`/`promote_slot` (single-rank); back with the built-but-unconnected `recall_tier` (`executor.rs:3069`) | **enumerate EVERY buffer** (pages + every `gdr_states[l]` + `conv_states[l]` + seq_len + **spec-decode state** — the EAGLE-rollback landmine); needle ×3 same-session spill+restore = byte-restore |
| **P4** | **Qwen3.6** | pos-0 reuse G2: `capture`/`restore_cached_prefix` | needle ×3 on the **failing slice** (case-as-fact) + post-restore linear-layer state-tensor vs fresh-prefill check |
| **P5** | shared | the capacity LEVER — second spill trigger (slot/QoS pressure) beside `retract_decode_to_fit`; reuse the G3 machinery, only trigger+victim-selector new | opt-in flag; multi-shape c-sweep clearing TTFT **and** ITL **and** throughput |
| **P6** | **DSv4** | multi-rank capacity — lockstep `demote/promote_slot` so `kv_slot_tier_enabled=true` at `world_size>1` (deterministic per-rank order + coordinator broadcast, NCCL-safe). The only genuinely-new INFRA; the only thing bringing the 850+380 GB to the flagship 8×H20 | DSv4 needle on pod + NCCL no-deadlock under concurrency; **defer if not needed** (single-rank works for dev/V100) |
| **P7** | shared | durability — wire `set_disk_durable`/`load` (unit-tested, not API-wired) from `loaded.rs`; `--kv-recall-durable-path` + weights-epoch tag (auto-discard on OPD weight flip) | guidellm shared-prefix L2/L3 hit-rate + recall persistence across serve restart |

## 8. Residual risks (license-or-kill, not silently passed)

1. **Sidecar size UNMEASURED** — the "850 GB holds tens of thousands of images" math needs the
   actual per-slot blob bytes (Qwen3.6 `gdr_states` f32 × ~48 layers + conv; DSv4 SW-ring +
   compressor rows + rotated_keys, large at long ctx). If a blob is tens of MB the spill D2H is a
   real stall. **Profile per-model blob size BEFORE the capacity claim ships.**
2. **Asymmetric reuse is physics, not a gap** — dense gets fine-grained mid-prefix reuse (G1);
   Qwen3.6 + DSv4 get only coarse pos-0 (G2) because recurrent/RoPE-rotated/ring state is
   position-locked. "One abstraction" is true at the store+gate level; reuse GRANULARITY is weaker
   for the two non-dense models. A per-layer-span reuse (reuse full-attn KV + recompute only the ~48
   linear layers) would recover it but risks the silent-wrong bug — **DEFERRED**, ship G2 first.
3. **Async write_through (P1) could wash** — B=1 decode is GPU-bound; an async side-stream mirror
   still contends HBM bandwidth + copy engines. A missed drain at the prefill sync reads a stale
   tier copy → silently wrong (EAGLE-rollback ordering class). Needle gate, matched A/B.
4. **DSv4 8×H20 production gets the weakest variant** — `world_size>1` disables BOTH G2 + G3; the
   flagship KV-hungry model gets paged-tier-only until P6. The capacity story must not oversell it.
5. **P3 spec-state enumeration is the silent-wrong landmine** — a Qwen3.6 slot demoted mid-draft has
   live `Qwen35SpecSlotState` a naive "pages+gdr+conv+seq_len" image MISSES → corrupted restore. The
   full buffer enumeration (incl. spec rings) MUST clear the design-to-file:line gate before P3.
6. **Capacity lever (P5) can thrash** — new admission heuristic; opt-in, multi-shape c-sweep before
   default-on.

**NOT struggling:** the soundness story is SOLID — GDR-reuse is structurally fenced by the
fail-closed `reusable_prefix_blocks=0` + the fail-loud `ensure!(seq_len==start_pos)`
(`executor.rs:3467`); a bad sidecar degrades to lost-reuse (perf), never wrong output. Decode
zero-tier-IO is structural. The needle gate **on the failing slice** is the single line of defense
between "sidecar works" and "sidecar silently never reused" — it MUST run per-case, not aggregate.
