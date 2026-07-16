use super::*;

/// Whole-step Qwen3.5/3.6 decode graph enabled? `--qwen35-decode-graph true`
/// opt-in, default OFF until the pod license (≥ +10% tok/s + needle gate +
/// replay-reuse evidence per the bench spec). The eager path stays the
/// correctness floor; `Qwen35CudaExecutor::warmup` additionally gates TP and
/// host-routed MoE off regardless of this.
fn qwen35_decode_graph_enabled() -> bool {
    crate::runtime_flags::qwen35_decode_graph()
}

/// Batched rows>1 decode for Qwen3.5/3.6 (stage 1, contiguous KV) — re-port
/// of the deleted monolith's `decode_batch` design
/// ([`crate::qwen35::Qwen35BatchDecodeState`]). DEFAULT ON: before this path
/// existed, a rows>1 plan was a hard executor error (`rows == 1` ensure →
/// engine death), so even a conservative batched path strictly dominates the
/// status quo. `--qwen35-batched-decode false` is the escape hatch AND the
/// honest same-binary A/B arm: it processes rows>1 decode plans as sequential
/// per-row single-row forwards instead (a NEW loop — the old behavior was
/// death, not a fallback). rows==1 plans never consult this gate (the
/// single-row path is byte-identical either way).
fn qwen35_batched_decode_enabled() -> bool {
    crate::runtime_flags::qwen35_batched_decode()
}

fn merge_tier_io_stats(
    slot: &kv_native_sys::TierIoStats,
    recall: &kv_native_sys::TierIoStats,
) -> infer_seam::KvTierIoStats {
    let mode = if [slot.mode, recall.mode].contains(&kv_native_sys::DiskIoMode::Direct) {
        infer_seam::KvTierIoMode::Direct
    } else if [slot.mode, recall.mode].contains(&kv_native_sys::DiskIoMode::Mmap) {
        infer_seam::KvTierIoMode::Mmap
    } else {
        infer_seam::KvTierIoMode::Disabled
    };
    infer_seam::KvTierIoStats {
        mode,
        useful_read_bytes: slot
            .useful_read_bytes
            .saturating_add(recall.useful_read_bytes),
        useful_write_bytes: slot
            .useful_write_bytes
            .saturating_add(recall.useful_write_bytes),
        submitted_read_bytes: slot
            .submitted_read_bytes
            .saturating_add(recall.submitted_read_bytes),
        submitted_write_bytes: slot
            .submitted_write_bytes
            .saturating_add(recall.submitted_write_bytes),
        metadata_write_bytes: slot
            .metadata_write_bytes
            .saturating_add(recall.metadata_write_bytes),
        failures: slot.failures.saturating_add(recall.failures),
        completion_wait_ns: slot
            .completion_wait_ns
            .saturating_add(recall.completion_wait_ns),
    }
}

/// Decode-graph replay/capture probes — static so the pod bench can PROVE
/// replay reuse from server logs (license requires reuse evidence, not
/// capture-exists). Expected steady state: captures == live slots (≤
/// num_slots keys; B=1/R=8/seq=1 are shape constants and kv enters via the
/// staged device scalar, so there is exactly ONE graph per slot), replays ≈
/// decoded tokens.
static QWEN35_GRAPH_CAPTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static QWEN35_GRAPH_REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Device addresses a slot's captured decode graph was baked against. The
/// graph replays kernel launches against FIXED pointers, so if any anchor
/// moved (workspace `release()` → re-alloc, a length-flip on the staging
/// slots) the capture is stale and must be dropped — replaying it would read
/// freed memory (the 2026-06-10 IMA class). `ws_epoch` covers wholesale
/// releases; the three pointer anchors cover the staged-input and output
/// buffers directly.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Qwen35GraphBake {
    token_ids_ptr: u64,
    start_pos_ptr: u64,
    logits_ptr: u64,
    ws_epoch: u64,
}

/// Per-executor whole-step decode-graph state: a DEDICATED decode workspace
/// (only ever sees `seq_len == 1` shapes, so its buffer addresses are stable
/// across prefill interleaves — the main workspace re-shapes on every
/// prefill chunk and would invalidate captures every request) plus one
/// [`CudaGraphState`] per slot (captured buffers bake the slot's k/v cache +
/// GDR/conv state addresses, which live for the executor's lifetime and are
/// only memset by `reset`).
struct Qwen35DecodeGraph {
    ws: crate::qwen35::Qwen35Workspace,
    graphs: Vec<crate::graph::CudaGraphState>,
    baked: Vec<Option<Qwen35GraphBake>>,
}

impl Qwen35DecodeGraph {
    fn new(num_slots: usize, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Self {
        Self {
            ws: crate::qwen35::Qwen35Workspace::new(),
            graphs: (0..num_slots)
                .map(|_| crate::graph::CudaGraphState::new(stream.clone()))
                .collect(),
            baked: vec![None; num_slots],
        }
    }
}

/// Qwen3.5 / Qwen3.6 HYBRID executor: drives
/// [`crate::qwen35::Qwen35Model::forward_tokens`] per single-row sub-step and
/// [`crate::qwen35::Qwen35Model::forward_decode_batch`] for rows>1 pure-decode
/// sub-batches (stage-1 batched decode, contiguous KV). Owns per-slot KV state
/// inside the model (full-attn contiguous caches + gated-delta recurrent
/// state), so it does NOT use a [`PagedKVPool`]; it relies on the host
/// [`KvPool`] only for the slot's logical `seq_len` to derive `start_pos`.
/// The whole-step decode graph is opt-in (`--qwen35-decode-graph`,
/// single-GPU + device-routed MoE only, rows==1 plans only).
///
/// Scope: prefill stays single-row (mixed plans run per-prefill sub-steps,
/// then one decode sub-batch — the DSv4 executor pattern), uncached
/// full-prefix (each full-attn layer recomputes over its contiguous cache;
/// each linear-attn layer advances the recurrent state in place). A
/// continuous-batching paged + packed-batch path is the stage-2 follow-up
/// (legacy `infer/src/model/qwen35`).
pub(crate) struct Qwen35CudaExecutor {
    pub(crate) model: crate::qwen35::Qwen35Model,
    /// Per-slot KV + recurrent state (one [`crate::qwen35::Qwen35SlotState`] per slot).
    /// A slot's recurrent state is EMPTY until its first request activates it.
    slots: Vec<crate::qwen35::Qwen35SlotState>,
    /// G3 whole-slot capacity spill: an inactive/retracted request parks its
    /// whole slot snapshot here (instead of being dropped + recomputed) and is
    /// restored byte-exact on resume. Routes through the SAME `KvTierStore`
    /// transport as the page-granular `recall_tier` (the unified-tier plan — all
    /// grains move bytes through ONE store kind by opaque `u64` key), so G3 gets
    /// the managed 850 GB DRAM budget + NVMe spill + durability for free instead
    /// of a private DRAM map. Sized for whole-slot snapshots (one entry ≈ one
    /// recurrent-block image), so its count cap = budget / image-bytes (the
    /// budget-aware ~5500 cap, not the old `num_slots*2`). Always built — G3 is
    /// default-on and independent of `--kv-recall` (the page grain). Keyed by the
    /// engine's session key, a namespace disjoint from `recall_tier`'s
    /// `tier_block_u64(slot, page)` keys (separate store ⇒ no aliasing).
    slot_tier: KvTierStore,
    /// Per-unit size (one whole-slot recurrent-block image) for the `slot_tier`
    /// budget, stored so `set_kv_tier_budget_bytes` can
    /// re-budget the tier post-construction without recomputing it (L2).
    /// Free-list of detached recurrent blocks (~147 MiB each). Released here by a
    /// finished request, popped by the next — so only ACTIVE slots hold a block,
    /// not all `num_slots`. Grows to `num_slots` at full concurrency (then HBM ==
    /// the old upfront reservation); idle/partial load costs proportionally less.
    recurrent_pool: Vec<crate::qwen35::RecurrentBlock>,
    /// Persistent forward workspace (exact-shape buffer reuse): forwards are
    /// strictly serial on this executor, so ONE workspace serves every slot.
    /// Mirrors the DSv4 persistent decode scratch (passed `&mut` per forward).
    workspace: crate::qwen35::Qwen35Workspace,
    pub(crate) num_slots: usize,
    /// Whole-step decode graph armed (env gate + TP-single + device-routed
    /// MoE, resolved at construction; `warmup` logs the verdict). ANY capture
    /// failure clears this — eager is the permanent fallback, never fatal.
    decode_graph_armed: bool,
    /// Lazily-built per-slot graph state (`None` until the first gated decode,
    /// and re-`None`d whenever baked addresses go stale: OPD weight
    /// offload/reload, student-LoRA re-merge).
    decode_graph: Option<Qwen35DecodeGraph>,
    /// Lazily-built batched rows>1 decode state: a dedicated `[*, B]`
    /// workspace plus per-layer recurrent-state pointer tables (see
    /// [`crate::qwen35::Qwen35BatchDecodeState`]). `None` until the first
    /// batched decode.
    batch_decode: Option<crate::qwen35::Qwen35BatchDecodeState>,
    /// Session KV-recall opt-in (`--kv-recall`, default off). When true the
    /// recall *cycle* (score → evict → prefetch) layers a working-set
    /// restriction on the SAME paged `full_attn_kv` pool the default path uses;
    /// off → the default attends the full resident page set (no eviction). The
    /// full-attn KV is paged in BOTH cases since the shared-paged migration.
    kv_recall: bool,
    /// Recall budget (sink/local/block/top_k). Mirrors the dense arm.
    recall_cfg: infer_core::RecallConfig,
    /// Per-slot recall state (reps + next-step page plan + evict list).
    recall: Vec<crate::recall::CudaRecallState>,
    /// One-shot warn latch when --kv-recall is requested with a non-BF16 pool.
    #[allow(dead_code)]
    recall_quant_warned: bool,
    /// Shared paged full-attn KV pool (HD256, `num_full` layers), the DEFAULT
    /// full-attn KV substrate since the shared-paged migration — built eagerly
    /// in the constructor, profile-sized from measured free VRAM (not
    /// `num_slots × max_seq_len`). Both the default forward and the `--kv-recall`
    /// cycle read/write THIS one pool; `Option` only so the OPD-offload path can
    /// drop it. Device-only, self-allocating.
    full_attn_kv: Option<PagedKVPool>,
    /// KV pool format (BF16 / FP8E4M3 / INT8). Captured at construction from the
    /// `--kv-cache-dtype` flag; stored so `ensure_kv_pool` can rebuild the pool
    /// with the same format after `release_kv_pool` drops it on the OPD path.
    kv_format: KVFormat,
    /// L3 write-through tier for the recall cycle (host DRAM, optional NVMe spill):
    /// the source of truth for evict-dropped middle blocks. Allocated lazily on
    /// the first `set_kv_recall(true)` and sized to ONE pool page image. Keyed by
    /// `tier_block_u64(slot, logical_page)` — a session-scoped namespace, so slot A
    /// never prefetches slot B's KV. `None` until `--kv-recall` is enabled.
    recall_tier: Option<KvTierStore>,
    /// Per-slot one-step eviction keepalive (the race fix). A page evict-dropped at
    /// decode step N is parked here, NOT returned to `free_pages`, until the START
    /// of step N+1 — by which point step N's attention (and its argmax `ctx.sync()`
    /// at the step boundary) has completed, so `alloc_tokens` can never hand the
    /// in-flight attention's page to the new token. Drained at the top of each
    /// `decode_row_recall` for that slot. Holds (logical, physical) so the parked
    /// physical page is the one actually returned to the pool one step later.
    recall_keepalive: Vec<Vec<(usize, u32)>>,
    /// Resolved per-rank L2 byte budget (`--kv-dram` ÷ world size). Recorded by
    /// `set_kv_tier_budget_bytes` alongside the `slot_tier` rebuild; consumed
    /// when `set_kv_recall(true)` lazily builds `recall_tier`.
    recall_budget_bytes: usize,

    /// Model checkpoint path for deriving the weights epoch tag at durable
    /// NVMe spill time (`set_kv_recall` / `set_kv_tier_disk`).
    #[allow(dead_code)]
    model_path: std::path::PathBuf,
    /// Weights-version tag from the checkpoint (`weights_epoch_tag`). Stamped
    /// into the durable recall manifest so a restart drops stale KV after an
    /// OPD weight update.
    weights_epoch: String,
    /// Operator-provided NVMe root for durable recall spill (`--kv-disk`).
    /// `None` until `set_kv_tier_disk` wires it.
    disk_root: Option<std::path::PathBuf>,
    /// Budget bytes for durable NVMe recall spill (`--kv-disk-limit`).
    /// `None` until `set_kv_tier_disk` wires it.
    disk_budget: Option<usize>,
    /// `mem_fraction_static` + requested page floor captured at construction so
    /// `ensure_kv_pool` can RE-PROFILE and rebuild `full_attn_kv` identically
    /// after `release_kv_pool` dropped it (agent-OPD rollout→writeback→rollout:
    /// the dead pool is freed for the co-resident autograd writeback, then
    /// re-acquired before the next round's rollout). Not on the serve path.
    kv_pool_mem_fraction_static: f64,
    kv_pool_requested_pages: usize,
    /// Recurrent prefix sidecars live in `slot_tier` under `NS_SIDECAR` (blob =
    /// recurrent state + full-attn KV, keyed by token-prefix hash), so they share
    /// the ONE budget-managed store instead of a private capped RAM map — the
    /// 32-cap that always-missed beyond the hottest prefixes is gone (issue #85).
    /// This map is eviction coordination ONLY: the tail host-pool page of each
    /// published prefix → its sidecar key. `release_prefix_pages` drops the blob
    /// when the radix evicts that page, so the sidecar's lifetime rides the radix
    /// blocks — no independent LRU. (The store's own budget is the memory
    /// backstop for any blob whose tail page LRU-evicted before drop reached it.)
    sidecar_page_key: std::collections::HashMap<u32, u64>,
    /// Per-slot recurrent snapshot captured DURING prefill exactly at the
    /// last-page boundary `L* = align_down16(prompt_len - 1)` — the restore
    /// target for an exact resend (un-popped for non-aligned, pop-trimmed for
    /// aligned; see `prefix.rs:72`). The device recurrent state has no position
    /// index and cannot rewind, so a snapshot taken at prompt-completion (`P`)
    /// keyed at `L*` would bake the residue `[L*..P]` and double-advance it on
    /// restore's re-prefill — corrupting page-aligned lengths. Capturing at `L*`
    /// makes snapshot-position == key-position. Consumed by
    /// `save_recurrent_sidecar`; cleared on the next request (`start_pos == 0`)
    /// so no cross-request state leaks.
    prefill_boundary_snapshot: Vec<Option<(usize, crate::qwen35::Qwen35RecurrentSnapshot)>>,
    /// Per-slot recurrent snapshots captured DURING prefill at every periodic
    /// stride boundary `S = k * SIDECAR_SNAPSHOT_STRIDE_PAGES * 16` crossed in
    /// `[start_pos..prompt)` (excluding `L*`, which `prefill_boundary_snapshot`
    /// owns). Each carries the state at EXACTLY `S` (snapshot-position ==
    /// key-position), so `save_recurrent_sidecar` can store a sidecar keyed at
    /// `hash(tokens[..S])` for a future cross-conversation restore at `S`. Drained
    /// at save; cleared on a fresh request (`start_pos == 0`) and on prefix-reuse
    /// attach so no prior occupant's snapshots leak into this request's saves.
    periodic_boundary_snapshots: Vec<Vec<(usize, crate::qwen35::Qwen35RecurrentSnapshot)>>,
    /// DSpark block-draft runtime (`--spec-type dspark`): draft head + per-slot
    /// ctx caches + tap/scratch buffers. `None` (the default) keeps the
    /// baseline executor byte-identical — nothing is allocated or captured.
    pub(crate) dspark: Option<crate::qwen35::dspark::Qwen35DsparkExec>,
    /// MTP spec-decode per-slot state (`--spec-type mtp`): the spec draft/verify
    /// state + the seed (pending token + hidden) for the next spec step.
    /// `None` (the default) keeps the baseline executor byte-identical.
    mtp: Option<MtpExec>,
}

/// Per-slot MTP spec-decode runtime state. The spec state is created lazily on
/// the first decode step that needs it (a warm forward captures the seed
/// hidden + pending token); subsequent steps run `spec_step` directly.
struct MtpExec {
    slots: Vec<Option<MtpSlotState>>,
}

struct MtpSlotState {
    spec: crate::qwen35::Qwen35SpecSlotState,
    pending: u32,
    hidden: DeviceVec,
}

impl std::fmt::Debug for Qwen35CudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35CudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .field("decode_graph_armed", &self.decode_graph_armed)
            .field(
                "captured_decode_slots",
                &self
                    .decode_graph
                    .as_ref()
                    .map_or(0, |dg| dg.graphs.iter().filter(|g| g.is_captured()).count()),
            )
            .finish()
    }
}

impl Qwen35CudaExecutor {
    /// `|_| false`: demote/promote is a no-op for Qwen35, so demoted pages are never restorable.
    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        pages_only_reusable_prefix_blocks(blocks, |_| false)
    }

    /// Serialize the slot's recurrent state + full-attn KV as ONE atomic blob
    /// and store it into `slot_tier` under `NS_SIDECAR`, keyed by the token hash
    /// of the published radix prefix. Records the prefix's tail page → key so
    /// `release_sidecar_pages` drops the blob when the radix evicts that page.
    ///
    /// Boundary: the sidecar is keyed at `L* = align_down16(tokens.len() - 1)` —
    /// the largest page boundary strictly below the sequence length, i.e. the
    /// restore target a future exact resend lands on (pop-trimmed for aligned,
    /// un-popped for non-aligned; `prefix.rs:72`). When a prefill boundary
    /// snapshot was captured at exactly `L*` (`submit_prefill_row`), it is used
    /// verbatim — snapshot-position == key-position, so restore's re-prefill of
    /// `[L*..len]` advances the recurrent state exactly once. Otherwise (finish/
    /// decode publish, recall path, or tiny prompt) fall back to the live
    /// snapshot keyed at the sealed page grain — the pre-existing best-effort
    /// path for supersequence (agentic follow-up) reuse.
    pub(crate) fn save_recurrent_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> anyhow::Result<()> {
        if !self.slots[slot].has_recurrent() {
            return Ok(()); // pure full-attn path — nothing to capture
        }
        // Drain the periodic snapshots up front so EVERY exit path (incl. the
        // mat_len==0 / D2H-fail early returns) leaves the slot clean for its next
        // occupant — a leaked entry would double-save under a later publish.
        let periodic = std::mem::take(&mut self.periodic_boundary_snapshots[slot]);
        let boundary = tokens.len().saturating_sub(1) / SUPPORTED_PAGE_SIZE * SUPPORTED_PAGE_SIZE;
        let pending = self.prefill_boundary_snapshot[slot].take();
        let (mat_len, mut snap) = match pending {
            Some((pos, snap)) if pos == boundary && boundary > 0 => (pos, snap),
            _ => {
                // Legacy best-effort: raw seq_len aligns to a page ~1/16 of the
                // time, so page-align to keep capture-key == restore's matched_len.
                let mat_len = matched_len
                    .min(self.slots[slot].seq_len())
                    .min(tokens.len())
                    / SUPPORTED_PAGE_SIZE
                    * SUPPORTED_PAGE_SIZE;
                if mat_len == 0 {
                    return Ok(());
                }
                (
                    mat_len,
                    self.slots[slot].snapshot_recurrent(&self.model.ctx)?,
                )
            }
        };
        let key = crate::qwen35::hash_prefix_tokens(&tokens[..mat_len]);
        if std::env::var_os("ARLE_KVDRIFT_DEBUG").is_some() {
            eprintln!(
                "[kvdrift] SAVE-SIDECAR slot={slot} mat_len={mat_len} key={key:#018x} \
                 tokens_len={} boundary={boundary} matched_len_in={matched_len}",
                tokens.len(),
            );
        }
        // Copy KV[0..mat_len] to host ONCE (snapshot_recurrent synchronized the
        // stream). full-attn KV is position-indexed and append-only, so pages
        // [0..mat_len) are byte-identical whether copied at `mat_len` or the
        // current end, AND every periodic boundary `pos ≤ mat_len` is a page-major
        // PREFIX of this buffer — slice it instead of re-copying (avoids O(N²) D2H
        // across the periodic boundaries).
        let host_kv = match self.full_attn_kv.as_ref() {
            Some(pool) => {
                let n_pages = mat_len / SUPPORTED_PAGE_SIZE;
                let all_pages = pool.page_indices(slot);
                let pages = all_pages[..n_pages.min(all_pages.len())].to_vec();
                if pages.is_empty() {
                    None
                } else {
                    match pool.copy_pages_to_host(&self.model.ctx, &pages) {
                        Ok(data) => Some(data),
                        Err(e) => {
                            log::warn!(
                                "slot {slot}: full-attn KV D2H failed: {e}; skipping sidecar entry"
                            );
                            return Ok(()); // don't store an incomplete (KV-less) blob
                        }
                    }
                }
            }
            None => None,
        };
        // Bytes per KV page (page-major layout): slice `[..pages * page_bytes]`
        // gives KV[0..pages*16] for any periodic boundary.
        let n_pages = (mat_len / SUPPORTED_PAGE_SIZE).max(1);
        let page_bytes = host_kv.as_ref().map(|d| d.len() / n_pages);

        // PERIODIC sidecars at each stride boundary `0 < pos ≤ mat_len` snapshotted
        // during this prefill — for a FUTURE cross-conversation restore that shares
        // only a leading prefix. Each blob = the recurrent state captured at EXACTLY
        // `pos` + KV[0..pos] sliced from `host_kv` (snapshot-position == key-position).
        // Saved BEFORE the primary so the primary can MOVE `host_kv` (no clone).
        for (pos, mut psnap) in periodic {
            if pos == 0 || pos > mat_len {
                continue; // beyond the copied/published prefix — nothing to key
            }
            psnap.full_attn_kv = match (host_kv.as_ref(), page_bytes) {
                (Some(kv), Some(pb)) => Some(kv[..(pos / SUPPORTED_PAGE_SIZE) * pb].to_vec()),
                _ => None,
            };
            let pkey = crate::qwen35::hash_prefix_tokens(&tokens[..pos]);
            self.store_sidecar_blob(pos, pkey, psnap.to_bytes(), prefix_pages);
        }

        // PRIMARY sidecar at `mat_len` (the exact-resend restore target) — the
        // within-conversation reuse path, unchanged.
        snap.full_attn_kv = host_kv;
        self.store_sidecar_blob(mat_len, key, snap.to_bytes(), prefix_pages);
        Ok(())
    }

    /// Insert a sidecar blob under `key` and coordinate its eviction off the last
    /// radix page it COVERS (`prefix_pages[pos/16 - 1]`): leaves evict
    /// deepest-first, so the blob drops the moment its own prefix erodes. Shared
    /// by the primary (`mat_len`) and periodic (`pos ≤ mat_len`) saves.
    fn store_sidecar_blob(&mut self, pos: usize, key: u64, bytes: Vec<u8>, prefix_pages: &[u32]) {
        if !self
            .slot_tier
            .insert_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key, &bytes)
        {
            return;
        }
        let cover_idx = (pos / SUPPORTED_PAGE_SIZE).saturating_sub(1);
        if let Some(&tail) = prefix_pages.get(cover_idx).or_else(|| prefix_pages.last())
            && let Some(old) = self.sidecar_page_key.insert(tail, key)
            && old != key
        {
            self.slot_tier
                .remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, old);
        }
    }

    /// Radix evicted these host-pool prefix pages: drop the sidecar blob keyed to
    /// any of them. Eviction rides the radix — no independent sidecar LRU.
    pub(crate) fn release_sidecar_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            if let Some(key) = self.sidecar_page_key.remove(&page) {
                self.slot_tier
                    .remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key);
            }
        }
    }

    /// Engine freed `slot`'s host pages (finish/park/requeue): return its device
    /// pages NOW. The old lazy free at the next occupant's prefill left the host
    /// admission pool over-reporting free pages by the whole dead slot, so the
    /// planner licensed prefill chunks the device pool could not hold — an
    /// engine-fatal alloc error at submit. Safe here: sidecar/tier capture runs
    /// before the engine frees host pages, and no forward is in flight for a
    /// slot being released. Idempotent with the prefill-start `free_slot`.
    pub(crate) fn release_kv_slot(&mut self, slot: usize) {
        if slot >= self.num_slots {
            return;
        }
        let parked = std::mem::take(&mut self.recall_keepalive[slot]);
        if let Some(pool) = self.full_attn_kv.as_mut() {
            for (_logical, physical) in parked {
                pool.release_evicted_page(physical);
            }
            pool.free_slot(slot);
        }
    }

    /// Restore the recurrent sidecar for a page-radix prefix hit, returning the
    /// ABSOLUTE token length restored — `matched_len` on an exact hit, or the
    /// largest periodic stride boundary `B ≤ matched_len` whose sidecar is present
    /// (a cross-conversation hit where no sidecar was saved exactly at
    /// `matched_len`). Acquires recurrent buffers, restores state + full-attn
    /// KV[0..B], and sets the device pool seq_len to `B` for the
    /// `prefill_row_paged_default` invariant; the engine truncates the host pool
    /// to `B` and re-prefills `[B..prompt]`. A MISS at every boundary returns
    /// `Err` so the caller full-recomputes.
    ///
    /// Correctness of the `[B..matched_len]` re-prefill: the blob keyed at
    /// `hash(tokens[..B])` holds the recurrent state and KV captured at EXACTLY
    /// `B` (snapshot-position == key-position). The tail prefill re-runs
    /// `[B..prompt]`, advancing the recurrent state and recomputing full-attn KV
    /// from `B` — identical to a fresh prefill, since both are deterministic
    /// functions of the token prefix. The radix KV pages `[B..matched_len]` are
    /// NOT laid over the `B`-boundary state (device pool is rebuilt to `B` here,
    /// re-prefill allocates fresh tail pages) — avoiding the "one block too many"
    /// corruption.
    pub(crate) fn restore_recurrent_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        _prefix_pages: &[u32],
    ) -> anyhow::Result<usize> {
        let matched_len = matched_len.min(tokens.len());
        if std::env::var_os("ARLE_KVDRIFT_DEBUG").is_some() {
            let pool_len = self
                .full_attn_kv
                .as_ref()
                .map_or(usize::MAX, |p| p.seq_len(slot));
            eprintln!(
                "[kvdrift] RESTORE-SIDECAR slot={} matched_len={} prompt_tokens={} prefix_pages={} slot.seq_len(before)={} pool.seq_len(before)={}",
                slot,
                matched_len,
                tokens.len(),
                _prefix_pages.len(),
                self.slots[slot].seq_len(),
                pool_len,
            );
        }
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        // Reused slot: start_pos != 0 skips the normal release+acquire in submit_prefill_row.
        self.slots[slot].release_recurrent(&mut self.recurrent_pool);
        self.slots[slot].acquire_recurrent(
            &self.model.ctx,
            num_linear,
            gdr_len,
            conv_len,
            &mut self.recurrent_pool,
        )?;

        // A sidecar-restored prefix has no draft ctx features for the matched
        // span — clear stale DSpark state; the tail prefill rebases the draft
        // ctx at start_pos (suffix-only drafting).
        if let Some(df) = self.dspark.as_mut().and_then(|ds| ds.slots[slot].as_mut()) {
            df.reset();
        }

        // Probe boundaries largest-first: the exact `matched_len` (today's
        // within-conversation fast path), then descending stride multiples
        // `≤ matched_len`. The first present blob wins — a miss on a higher
        // boundary reads nothing heavy (key-absent short-circuits), and we never
        // read-then-discard a hit. Each boundary is page-aligned (matched_len is
        // a radix page match, stride is a page multiple), so `hash(tokens[..B])`
        // rendezvous with the save keys.
        let stride = SIDECAR_SNAPSHOT_STRIDE_PAGES * SUPPORTED_PAGE_SIZE; // const, > 0
        let mut candidates: Vec<usize> = Vec::new();
        if matched_len > 0 {
            candidates.push(matched_len);
        }
        let mut b = matched_len / stride * stride;
        while b >= stride {
            if b != matched_len {
                candidates.push(b);
            }
            b -= stride;
        }
        // Read the atomic blob from the tier (transparently spans L1/L2 host →
        // L3 disk); a corrupt/foreign payload deserializes to None and is skipped.
        let restored = candidates.into_iter().find_map(|b| {
            let key = crate::qwen35::hash_prefix_tokens(&tokens[..b]);
            self.slot_tier
                .read_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key)
                .ok()
                .and_then(|bytes| crate::qwen35::Qwen35RecurrentSnapshot::from_bytes(&bytes).ok())
                .map(|snap| (b, snap))
        });

        let Some((boundary, snap)) = restored else {
            // MISS at every boundary → reset slot state, then Err so the caller
            // full-recomputes. Must clean up full_attn_kv and seq_len here — the
            // caller's Err handler won't do it.
            if let Some(pool) = self.full_attn_kv.as_mut() {
                pool.free_slot(slot);
            }
            self.slots[slot].set_seq_len(0);
            return Err(anyhow::anyhow!(
                "no recurrent sidecar for prefix matched_len={matched_len} \
                 (probed stride={stride}); falling back to full recompute"
            ));
        };

        // restore_recurrent_from_snapshot doesn't advance seq_len; set it to the
        // restored boundary for the submit_decode_row / prefill readiness invariant.
        self.slots[slot].restore_recurrent_from_snapshot(&self.model.ctx, &snap)?;
        self.slots[slot].set_seq_len(boundary);

        // Rebuild the device KV pool to EXACTLY `boundary` pages, restoring
        // KV[0..boundary] from the blob; the re-prefill of [boundary..prompt]
        // allocates fresh tail pages (never reuses the radix [B..matched_len] KV).
        if let Some(pool) = self.full_attn_kv.as_mut() {
            pool.free_slot(slot);
            let new_pages = pool.alloc_tokens(slot, boundary).map_err(|e| {
                anyhow::anyhow!("device pool prefix alloc failed for slot {slot}: {e}")
            })?;
            if let Some(kv_data) = snap.full_attn_kv.as_deref() {
                pool.copy_pages_from_host(&self.model.ctx, &new_pages, kv_data)
                    .map_err(|e| {
                        anyhow::anyhow!("device pool KV H2D restore failed for slot {slot}: {e}")
                    })?;
            }
            // Stream-order: H2D above is ordered before subsequent kernels; no explicit sync needed.
        }
        // Fresh prefill sequence for [boundary..prompt]: drop any prior occupant's
        // periodic snapshots so this request's saves never inherit stale entries.
        self.periodic_boundary_snapshots[slot].clear();
        Ok(boundary)
    }

    /// Whole-slot swap is rank-local bytes plus TP-wide scalar consensus
    /// ([`Self::tp_min_usize`] in demote/promote), mirroring DSv4's arm.
    pub(crate) fn kv_slot_tier_enabled(&self) -> bool {
        true
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        self.slot_tier.host_demoted_pages()
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        self.slot_tier.disk_pages()
    }

    pub(crate) fn kv_tier_io_stats(&self) -> infer_seam::KvTierIoStats {
        let slot = self.slot_tier.io_stats();
        let recall = self
            .recall_tier
            .as_ref()
            .map_or_else(kv_native_sys::TierIoStats::default, KvTierStore::io_stats);
        merge_tier_io_stats(&slot, &recall)
    }

    fn tp_min_usize(&self, value: usize, what: &str) -> Result<usize> {
        let capped = i32::try_from(value.min(i32::MAX as usize)).unwrap_or(i32::MAX);
        self.model
            .tp
            .all_reduce_min_scalar_i32(&self.model.ctx, capped)
            .map(|v| v.max(0) as usize)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 TP min-reduce {what} failed: {e}"))
    }

    /// `BackendExecutor::tp_sync_min` — see there for why the scheduler needs
    /// this (2026-07-05 TP=4 admission livelock).
    pub(crate) fn tp_sync_min(&self, local: usize) -> Result<usize> {
        self.tp_min_usize(local, "admission free pages")
    }

    /// Demote `slot`'s entire device state into the host `slot_tier` under `key`.
    /// The copy is complete before returning (`swap_out_image` ends in
    /// `ctx.sync()`), so the engine may free the slot immediately. The image is
    /// serialized byte-exact ([`Qwen35SlotImage::to_bytes`]) and handed to the
    /// shared `KvTierStore` transport, which manages the 850 GB DRAM budget
    /// (+ optional NVMe spill). Returns `Ok(false)` when the tier is at budget
    /// on ANY rank (engine falls back to plain recompute, same contract as the
    /// old cap). Exactly TWO collectives on every path (capture consensus,
    /// insert consensus), so the lockstep collective count is rank-invariant.
    pub(crate) fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
        ensure!(
            slot < self.num_slots,
            "Qwen3.6 demote slot {slot} outside executor slots {}",
            self.num_slots
        );
        let image = {
            let Self {
                model,
                slots,
                full_attn_kv,
                recurrent_pool,
                ..
            } = &mut *self;
            let pool = full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (whole-slot demote)");
            slots[slot].swap_out_image(&model.ctx, slot, pool, recurrent_pool)
        };
        let capture_ok = usize::from(image.is_ok());
        if self.tp_min_usize(capture_ok, "slot demote capture")? == 0 {
            return Err(image.err().unwrap_or_else(|| {
                anyhow::anyhow!("peer rank failed Qwen3.6 slot demote capture")
            }));
        }
        // Chunked like DSv4 (16 MiB store pages): the whole image never fit a
        // recurrent-floor-sized page — the review-6 blocker where every insert
        // was refused post-oversize-guard and park degraded to pure recompute.
        // Per-rank DRAM headroom can diverge, so the verdict is min-reduced; on
        // a mixed verdict, locally-successful ranks roll their insert back so
        // no rank keeps a blob the others lack.
        let bytes = image?.to_bytes();
        let inserted = self
            .slot_tier
            .insert_chunked(NS_SLOT, NS_SLOT_CHUNK, key, &bytes);
        if self.tp_min_usize(usize::from(inserted), "slot demote insert")? == 0 {
            if inserted {
                self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
            }
            return Ok(false);
        }
        Ok(true)
    }

    /// Restore the whole-slot snapshot stored under `key` into `slot`. The engine
    /// resumes decode at the demoted position right after this returns, and drops
    /// the entry via [`Self::drop_kv_slot_entries`] — the entry intentionally
    /// stays in the tier here. `swap_in_image` ends in `ctx.sync()`, so the
    /// device restore is complete before the host image can be dropped. Exactly
    /// TWO collectives on every path (read+parse consensus, restore consensus) —
    /// nothing rank-local errs between them.
    pub(crate) fn promote_slot(&mut self, key: u64, slot: usize) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "Qwen3.6 promote slot {slot} outside executor slots {}",
            self.num_slots
        );
        // The slot image carries no draft ctx cache — clear stale state; the
        // first warm decode rebases the draft ctx at the promoted position.
        if let Some(df) = self.dspark.as_mut().and_then(|ds| ds.slots[slot].as_mut()) {
            df.reset();
        }
        // Read the serialized image out of the tier (host hit or NVMe read) and
        // deserialize it byte-exact before touching the device.
        let image = self
            .slot_tier
            .read_chunked(NS_SLOT, NS_SLOT_CHUNK, key)
            .map_err(|err| anyhow::anyhow!("Qwen3.6 whole-slot tier read key {key}: {err}"))
            .and_then(|bytes| crate::qwen35::Qwen35SlotImage::from_bytes(&bytes));
        let image_ok = usize::from(image.is_ok());
        if self.tp_min_usize(image_ok, "slot promote read")? == 0 {
            return Err(image
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("peer rank failed Qwen3.6 slot promote read")));
        }
        let image = image?;
        // Infallible config data — safe between the collectives.
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        let restored = {
            let Self {
                model,
                slots,
                full_attn_kv,
                recurrent_pool,
                ..
            } = &mut *self;
            let pool = full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (whole-slot promote)");
            slots[slot].swap_in_image(
                &model.ctx,
                slot,
                pool,
                recurrent_pool,
                num_linear,
                gdr_len,
                conv_len,
                &image,
            )
        };
        let restore_ok = usize::from(restored.is_ok());
        if self.tp_min_usize(restore_ok, "slot promote restore")? == 0 {
            return Err(restored.err().unwrap_or_else(|| {
                anyhow::anyhow!("peer rank failed Qwen3.6 slot promote restore")
            }));
        }
        restored
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        for &key in keys {
            self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
        }
    }

    pub(crate) fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        max_total_tokens: usize,
        kv_dtype: CudaKvCacheDtype,
        mem_fraction_static: f64,
        dspark_draft_model: Option<&Path>,
        dspark_conf_threshold: f32,
        mtp_draft_tokens: Option<usize>,
    ) -> Result<Self> {
        let total_t0 = Instant::now();
        ensure!(
            num_slots > 0,
            "Qwen35CudaExecutor requires at least one slot"
        );
        ensure!(
            total_pages > 0,
            "Qwen35CudaExecutor requires at least one KV page"
        );
        // `max_seq_len` is the per-request token ceiling (the model's positional
        // budget + the host CudaKvPool admission span). The full-attn KV is now a
        // SHARED profile-sized paged pool (built below), not a per-slot
        // contiguous cache, so this no longer scales VRAM by num_slots.
        let max_seq_len = total_pages * SUPPORTED_PAGE_SIZE;
        let model_t0 = Instant::now();
        // `None`: the checkpoint-native NextN-MTP head stays unwired here; the
        // baseline load is byte-identical (no draft head loaded). DSpark is the
        // external block drafter, loaded below from its own checkpoint dir.
        let mut model = crate::qwen35::Qwen35Model::from_safetensors(
            model_path.as_ref(),
            max_seq_len,
            mtp_draft_tokens,
        )?;
        cuda_startup_log(
            "qwen35_model_load",
            model_t0,
            format_args!(
                "requested_slots={num_slots} total_pages={total_pages} max_seq_len={max_seq_len}"
            ),
        );
        // DSpark draft head: loaded BEFORE the KV budget/pool profiling so its
        // weights are subtracted from the measured free VRAM.
        let dspark_head = dspark_draft_model
            .map(|dir| {
                ensure!(
                    model.tp.is_single(),
                    "--spec-type dspark is single-GPU only (draft lm_head/argmax are rank-local)"
                );
                let head = crate::qwen35::dspark::load_dspark_head(
                    &model.ctx,
                    dir,
                    max_seq_len,
                    max_total_tokens,
                    model.config.hidden_size,
                    model.config.num_hidden_layers,
                    model.config.vocab_size,
                    dspark_conf_threshold,
                )?;
                model.set_spec_draft_tokens(head.block_size());
                log::info!(
                    "CUDA Qwen3.6 DSpark drafter loaded from {}: mode={} block={} taps={:?}",
                    dir.display(),
                    head.mode_label(),
                    head.block_size(),
                    head.target_layer_ids(),
                );
                Ok(head)
            })
            .transpose()?;
        // Dynamic KV mem budget (unified with DSv4 via the infer-seam kernel):
        // clamp num_slots to what post-weights free VRAM affords. Qwen3.5/3.6
        // previously admitted the requested count as-is → OOM at large
        // max_seq_len. Deterministic + NCCL min-reduced ⇒ TP-consistent.
        let budget_t0 = Instant::now();
        // Reclaim retained loading scratch before the budget measures free VRAM
        // (see the DSv4 path above — same starvation otherwise).
        if let Err(e) = model.ctx.trim_memory_pool() {
            log::warn!("pre-KV-budget trim_memory_pool failed (non-fatal): {e}");
        }
        let dspark_slot_bytes = dspark_head.as_ref().map_or(0, |h| h.slot_state_bytes());
        let num_slots = model.kv_budget_num_slots(num_slots, dspark_slot_bytes)?;
        cuda_startup_log(
            "qwen35_kv_budget",
            budget_t0,
            format_args!("effective_slots={num_slots}"),
        );
        let slots_t0 = Instant::now();
        // Empty slots — zero recurrent HBM upfront. Each slot's ~147 MiB
        // recurrent block is drawn from `recurrent_pool` on its first request
        // (`acquire_recurrent` at the `start_pos == 0` prefill) and returned on
        // request finish. The pool starts empty and grows on demand.
        let slots: Vec<_> = (0..num_slots).map(|_| model.new_slot_state()).collect();
        cuda_startup_log(
            "qwen35_slot_alloc",
            slots_t0,
            format_args!("slots={num_slots} max_seq_len={max_seq_len}"),
        );

        // Shared paged full-attn KV pool — the DEFAULT substrate (Phase 2 of the
        // shared-paged-KV migration). Built EAGERLY here, profile-sized from
        // MEASURED free VRAM after weights load (SGLang's mem_fraction_static),
        // NOT `num_slots × max_seq_len` (the per-slot contiguous waste). The same
        // build is re-run by `ensure_kv_pool` after `release_kv_pool` drops it on
        // the agent-OPD writeback path — extracted into `build_full_attn_kv_pool`.
        let pool_t0 = Instant::now();
        let requested_pages = total_pages.max(1);
        let kv_format = kv_dtype.kv_format();
        let full_attn_kv = Self::build_full_attn_kv_pool(
            &model,
            num_slots,
            requested_pages,
            mem_fraction_static,
            kv_format,
            dspark_slot_bytes,
        )?;
        cuda_startup_log("qwen35_paged_pool_alloc", pool_t0, format_args!("built"));

        // G3 whole-slot spill tier: snapshots stored as 16 MiB chunked blobs
        // (manifest + chunks, the DSv4 pattern) — a whole image never fits one
        // fixed page, and the store's size contract is per-page. Same
        // `KvTierStore` transport as the page-granular recall tier — the
        // unified plan: every grain moves bytes through ONE store kind. NVMe
        // spill is opt-in via `set_kv_tier_disk` (shared with the dense arm).
        let tier_budget_bytes = default_t1_budget_bytes(DEFAULT_DRAM_FRACTION);
        let slot_tier = KvTierStore::with_budget(tier_budget_bytes, BLOB_CHUNK_BYTES);

        // Whole-step decode graph: env opt-in ∧ single-GPU (NCCL all-reduce is
        // not graph-capturable on this stack — TP≥2 stays eager, same as
        // dense) ∧ every layer's decode step is a pure device-kernel sequence.
        let decode_graph_armed = qwen35_decode_graph_enabled()
            && model.tp.is_single()
            && model.decode_graph_unsupported_reason().is_none();
        let model_path_buf = model_path.as_ref().to_path_buf();
        let weights_epoch = kv_native_sys::weights_epoch_tag(&model_path_buf);
        let executor = Self {
            model,
            slots,
            slot_tier,
            recurrent_pool: Vec::new(),
            workspace: crate::qwen35::Qwen35Workspace::new(),
            num_slots,
            decode_graph_armed,
            decode_graph: None,
            batch_decode: None,
            kv_recall: false,
            recall_cfg: crate::recall::default_recall_config(),
            recall: (0..num_slots)
                .map(|_| crate::recall::CudaRecallState::default())
                .collect(),
            recall_quant_warned: false,
            full_attn_kv: Some(full_attn_kv),
            kv_format,
            recall_tier: None,
            recall_keepalive: (0..num_slots).map(|_| Vec::new()).collect(),
            recall_budget_bytes: tier_budget_bytes,
            model_path: model_path_buf,
            weights_epoch,
            disk_root: None,
            disk_budget: None,
            kv_pool_mem_fraction_static: mem_fraction_static,
            kv_pool_requested_pages: total_pages.max(1),
            sidecar_page_key: std::collections::HashMap::new(),
            prefill_boundary_snapshot: (0..num_slots).map(|_| None).collect(),
            periodic_boundary_snapshots: (0..num_slots).map(|_| Vec::new()).collect(),
            dspark: dspark_head.map(|h| crate::qwen35::dspark::Qwen35DsparkExec::new(h, num_slots)),
            mtp: mtp_draft_tokens.map(|_| MtpExec {
                slots: (0..num_slots).map(|_| None).collect(),
            }),
        };
        cuda_startup_log(
            "qwen35_executor_total",
            total_t0,
            format_args!("slots={num_slots} max_seq_len={max_seq_len}"),
        );
        Ok(executor)
    }

    /// Build the shared paged full-attn KV pool, profile-sized from MEASURED free
    /// VRAM (SGLang's `mem_fraction_static`), floored at `requested_pages`. The
    /// constructor's eager build and `ensure_kv_pool`'s post-release rebuild both
    /// go through here so the re-acquired pool matches the original sizing recipe.
    fn build_full_attn_kv_pool(
        model: &crate::qwen35::Qwen35Model,
        num_slots: usize,
        requested_pages: usize,
        mem_fraction_static: f64,
        kv_format: KVFormat,
        dspark_slot_bytes: usize,
    ) -> Result<PagedKVPool> {
        let num_full = model.config.num_full_attention_layers();
        let local_kv_heads = model.local_kv_heads();
        let head_dim = model.config.head_dim;
        let cell_bytes_per_token =
            PagedKVPool::budget_bytes_for_tokens(num_full, local_kv_heads, head_dim, 1, kv_format)
                as u64;
        // H1: the per-slot recurrent block (gdr + conv) is reserved out of the
        // SAME free VRAM `kv_budget_num_slots` clamped against. Subtract that
        // reservation from free BEFORE profiling the full-attn pool, so
        // weights + recurrent×slots + pool ≤ mem_fraction×free (mirrors DSv4
        // dsv4.rs:1687-1691). Per-rank free, num_slots already min-reduced.
        let (_per_slot, _kv_bytes, gdr_bytes, conv_bytes) = model.per_slot_kv_bytes();
        // DSpark draft ctx caches are per-slot lazily-allocated device buffers —
        // reserve them here too, or the pool eats the VRAM the first dspark
        // prefill needs.
        let per_slot_recurrent = gdr_bytes
            .saturating_add(conv_bytes)
            .saturating_add(dspark_slot_bytes);
        let recurrent_reservation = per_slot_recurrent.saturating_mul(num_slots) as u64;
        let total_pool_pages = match model.ctx.mem_info_bytes() {
            Ok((free, total)) => {
                let free_after_recurrent = (free as u64).saturating_sub(recurrent_reservation);
                let profiled_tokens = infer_seam::profile_kv_pool_tokens(
                    free_after_recurrent,
                    total as u64,
                    cell_bytes_per_token,
                    mem_fraction_static,
                );
                let profiled_pages = (profiled_tokens / SUPPORTED_PAGE_SIZE as u64) as usize;
                let sized = profiled_pages.max(requested_pages).max(1);
                log::info!(
                    "CUDA Qwen3.6 full-attn KV pool profiled from measured VRAM: free {}MB / \
                     total {}MB, recurrent reservation {}MB ({num_slots} slots × {}MB), \
                     free_after_recurrent {}MB, mem_fraction_static {mem_fraction_static}, cell \
                     {cell_bytes_per_token}B/tok ({num_full} full-attn layers × {local_kv_heads} \
                     kv-heads × {head_dim} hd) -> max_total_tokens {profiled_tokens} \
                     ({profiled_pages} pages); requested floor {requested_pages} pages \
                     -> sizing {sized} pages",
                    free >> 20,
                    total >> 20,
                    recurrent_reservation >> 20,
                    per_slot_recurrent >> 20,
                    free_after_recurrent >> 20,
                );
                sized
            }
            Err(e) => {
                log::warn!(
                    "CUDA Qwen3.6 full-attn KV pool: free-VRAM probe failed ({e}); falling back \
                     to requested floor {requested_pages} pages"
                );
                requested_pages
            }
        };
        let pool_token_budget = total_pool_pages * SUPPORTED_PAGE_SIZE;
        let pool_budget_bytes = PagedKVPool::budget_bytes_for_tokens(
            num_full,
            local_kv_heads,
            head_dim,
            pool_token_budget,
            kv_format,
        );
        let full_attn_kv = PagedKVPool::with_format(
            &model.ctx,
            num_full,
            local_kv_heads,
            head_dim,
            num_slots,
            pool_budget_bytes,
            kv_format,
        )?;
        ensure!(
            full_attn_kv.page_size == SUPPORTED_PAGE_SIZE,
            "Qwen3.6 full-attn paged pool page_size={} != {SUPPORTED_PAGE_SIZE}",
            full_attn_kv.page_size
        );
        Ok(full_attn_kv)
    }

    /// Drop the shared paged full-attn KV pool, returning its HBM to the device
    /// async pool (trimmed to the OS so a co-resident allocator sees it free).
    /// agent-OPD ONLY: the masked-CE writeback's `forward_hidden_states` is a
    /// FRESH autograd forward that does NOT read this engine's KV cache, so the
    /// rollout pool is DEAD during the writeback — freeing it widens the writeback
    /// headroom. `ensure_kv_pool` re-acquires it before the next round's rollout.
    /// Precondition: all in-flight rollout work has synced (the round's rollout
    /// completed + LoRA synced before the writeback closure runs). No-op if the
    /// pool was already released. Errors if any slot still holds resident pages
    /// (a live request would lose its KV) — agent-OPD finishes all rollouts before
    /// the writeback, so the pool is empty.
    pub(crate) fn release_kv_pool(&mut self) -> Result<()> {
        let Some(pool) = self.full_attn_kv.as_ref() else {
            return Ok(());
        };
        let freed = pool.device_bytes();
        // Drain in-flight device work referencing the pool BEFORE dropping it (no
        // kernel may still read the slices we are about to free).
        self.model.ctx.sync()?;
        // Drop enqueues `cuMemFreeAsync` for every pool slice on the model stream
        // — the frees are ASYNC, so they have NOT executed yet here.
        self.full_attn_kv = None;
        // Sync AGAIN so the enqueued async frees actually complete and the blocks
        // return to the device async pool; only THEN can the trim return them to
        // the OS. Without this second sync the trim runs before any block is back
        // in the pool → nothing trimmed → `mem_get_info` still shows them used (the
        // bug the first version hit: pool dropped but VRAM not reclaimed).
        self.model.ctx.sync()?;
        // Return the freed async-pool blocks to the OS so the co-resident autograd
        // writeback's allocator + `mem_get_info` see them free (the pool's release
        // threshold is u64::MAX, so a drop alone only caches the blocks). The
        // device default async pool is per-DEVICE (shared across the infer + train
        // contexts on this GPU), so the trimmed bytes are available to autograd.
        if let Err(e) = self.model.ctx.trim_memory_pool() {
            log::warn!("release_kv_pool: trim_memory_pool failed (non-fatal): {e}");
        }
        log::info!(
            "Qwen3.6 released full-attn KV pool: freed {}MB (agent-OPD writeback headroom)",
            freed >> 20
        );
        Ok(())
    }

    /// Re-acquire the shared paged full-attn KV pool after `release_kv_pool`
    /// dropped it (agent-OPD next-round rollout). Re-profiles from current free
    /// VRAM with the construction-time `mem_fraction_static` + requested floor.
    /// No-op (idempotent) if the pool is already resident.
    pub(crate) fn ensure_kv_pool(&mut self) -> Result<()> {
        if self.full_attn_kv.is_some() {
            return Ok(());
        }
        let pool = Self::build_full_attn_kv_pool(
            &self.model,
            self.num_slots,
            self.kv_pool_requested_pages,
            self.kv_pool_mem_fraction_static,
            self.kv_format,
            self.dspark
                .as_ref()
                .map_or(0, |ds| ds.head.slot_state_bytes()),
        )?;
        log::info!(
            "Qwen3.6 re-acquired full-attn KV pool: {}MB (agent-OPD next-round rollout)",
            pool.device_bytes() >> 20
        );
        self.full_attn_kv = Some(pool);
        Ok(())
    }

    /// Boot-time decode-graph verdict log (mirrors the dense `warmup` info
    /// messages). Capture itself is lazy — one whole-step capture per slot on
    /// its first gated decode (after `CudaGraphState`'s universal eager warm
    /// run), so unused slots never pay capture/instantiation memory.
    pub(crate) fn warmup(&mut self) -> Result<()> {
        let warmup_t0 = Instant::now();
        let dense_t0 = Instant::now();
        let (warmed_shapes, warm_m) = self.model.warm_fp8_deepgemm_dense_prefill()?;
        cuda_startup_log(
            "qwen35_warm_dense_deepgemm",
            dense_t0,
            format_args!("shapes={warmed_shapes} warm_m={warm_m}"),
        );
        if warmed_shapes > 0 {
            info!(
                "Qwen3.5 FP8 dense DeepGEMM warmed {warmed_shapes} projection shape(s) at M={warm_m}"
            );
        }
        let grouped_t0 = Instant::now();
        let (grouped_shapes, grouped_tokens, grouped_min_rows, grouped_max_rows) =
            self.model.warm_fp8_deepgemm_grouped_prefill()?;
        cuda_startup_log(
            "qwen35_warm_grouped_deepgemm",
            grouped_t0,
            format_args!(
                "shapes={grouped_shapes} tokens={grouped_tokens} rows={grouped_min_rows}..{grouped_max_rows}"
            ),
        );
        if grouped_shapes > 0 {
            info!(
                "Qwen3.5 FP8 grouped DeepGEMM warmed {grouped_shapes} GEMM shape(s) at tokens<={grouped_tokens} rows={grouped_min_rows}..{grouped_max_rows}"
            );
        }
        if !qwen35_decode_graph_enabled() {
            info!(
                "Qwen3.5 whole-step decode graph disabled \
                 (set --qwen35-decode-graph to enable)"
            );
            cuda_startup_log(
                "qwen35_warmup_total",
                warmup_t0,
                format_args!("graph=disabled"),
            );
            return Ok(());
        }
        if !self.model.tp.is_single() {
            info!(
                "Qwen3.5 whole-step decode graph disabled under tensor parallelism \
                 (world_size>1, NCCL collectives are not graph-capturable); \
                 using eager forward"
            );
            cuda_startup_log(
                "qwen35_warmup_total",
                warmup_t0,
                format_args!("graph=tp_disabled"),
            );
            return Ok(());
        }
        if let Some(reason) = self.model.decode_graph_unsupported_reason() {
            info!("Qwen3.5 whole-step decode graph disabled: {reason}; using eager forward");
            cuda_startup_log(
                "qwen35_warmup_total",
                warmup_t0,
                format_args!("graph=unsupported"),
            );
            return Ok(());
        }
        debug_assert!(self.decode_graph_armed);
        info!(
            "Qwen3.5 whole-step decode graph ARMED ({} slots; lazy capture per slot, \
             one eager warm run before each first capture; eager fallback on any failure)",
            self.num_slots
        );
        cuda_startup_log(
            "qwen35_warmup_total",
            warmup_t0,
            format_args!("graph=armed"),
        );
        Ok(())
    }

    /// Resolved per-rank L2 byte cap (`--kv-dram` ÷ world size) for the G3
    /// `slot_tier`, also recorded for the lazily-built recall tier. Pre-serve
    /// only (drops any existing entries).
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        self.recall_budget_bytes = bytes;
        self.slot_tier = KvTierStore::with_budget(bytes, BLOB_CHUNK_BYTES);
    }

    /// Attach NVMe spill (`--kv-disk`): an ephemeral disk level under the
    /// G3 `slot_tier` (whole-slot capacity spill) plus a durable level for the
    /// recall tier (attached now if built, else stashed for `set_kv_recall`).
    /// Pre-serve only. The budget is a per-store cap, not a reservation — both
    /// stores are sparse mmaps, so disk is consumed only by actual spill.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        self.disk_root = Some(root.clone());
        self.disk_budget = Some(budget_bytes);
        let recall_attached = match self.recall_tier.as_mut() {
            Some(tier) => {
                let page_bytes = tier.page_bytes();
                tier.load(
                    root.clone(),
                    budget_bytes,
                    page_bytes,
                    self.weights_epoch.clone(),
                    self.kv_format
                        .stable_tag()
                        .expect("persisted KV format must have a stable tag"),
                    self.model.tp.config().world_size,
                    self.model.tp.config().rank,
                )
            }
            None => false,
        };
        self.slot_tier
            .set_disk(root, budget_bytes, BLOB_CHUNK_BYTES)
            || recall_attached
    }

    /// Opt into session KV-recall (`--kv-recall`, default off). The paged
    /// full-attn pool (`full_attn_kv`) is ALWAYS resident since the shared-paged
    /// migration — the DEFAULT forward already attends it (full resident set, no
    /// eviction). Enabling recall layers the eviction/scoring *cycle* on the
    /// SAME pool (working-set restriction) and lazily builds its L3 tier on the
    /// first enable. Default-off keeps the full resident attention.
    pub(crate) fn set_kv_recall(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            !(enabled && self.dspark.is_some()),
            "--kv-recall is not supported with --spec-type dspark (the verify \
             forward would race the recall eviction cycle)"
        );
        self.kv_recall = enabled;
        if enabled && self.recall_tier.is_none() {
            // L3 write-through tier: source of truth for evict-dropped middle
            // blocks. One entry == one pool page image (all `num_full` layers,
            // K+V). Host-DRAM budget is the resolved per-rank `--kv-dram` share
            // (same cap as the slot tier); NVMe spill is opt-in via the
            // prefix-tier `--kv-disk` wiring. Reuses the SAME `KvTierStore`
            // transport the dense arm's prefix/write-through tier uses (R5 —
            // one store kind).
            let page_bytes = self
                .full_attn_kv
                .as_ref()
                .map(|p| p.storage_bytes_per_page())
                .ok_or_else(|| {
                    anyhow::anyhow!("--kv-recall: full-attn paged pool not allocated")
                })?;
            let mut tier = KvTierStore::with_budget(self.recall_budget_bytes, page_bytes);
            // Try loading prior session durable NVMe spill if disk is configured.
            // Falls through to set_disk_durable on first run or epoch mismatch.
            if let (Some(root), Some(budget)) = (self.disk_root.as_ref(), self.disk_budget) {
                let format_tag = self
                    .kv_format
                    .stable_tag()
                    .expect("persisted KV format must have a stable tag");
                let rank = self.model.tp.config().rank;
                let world_size = self.model.tp.config().world_size;
                let loaded = tier.load(
                    root.clone(),
                    budget,
                    page_bytes,
                    self.weights_epoch.clone(),
                    format_tag,
                    world_size,
                    rank,
                );
                if !loaded {
                    tier.set_disk_durable(
                        root.clone(),
                        budget,
                        page_bytes,
                        self.weights_epoch.clone(),
                        format_tag,
                        world_size,
                        rank,
                    );
                }
            }
            self.recall_tier = Some(tier);
        }
        Ok(())
    }

    /// Whether the full-attn KV path is paged (the shared pool is resident).
    /// TRUE by default since the shared-paged migration — the default forward
    /// attends the full resident page set. (`Option` is `None` only after an
    /// OPD weight offload dropped the pool.)
    fn full_attn_paged(&self) -> bool {
        self.full_attn_kv.is_some()
    }

    /// Actual shared full-attn paged-pool page count, so the host admission pool
    /// mirrors the device pool 1:1 (like dense). `None` only if the pool was
    /// dropped (OPD offload), in which case the caller falls back to the
    /// requested config value.
    pub(crate) fn full_attn_pool_pages(&self) -> Option<usize> {
        self.full_attn_kv.as_ref().map(|p| p.max_total_pages)
    }

    /// Whether the recall eviction/scoring *cycle* runs this session
    /// (`--kv-recall`). When false the default paged path still runs — it just
    /// attends the full resident set (no working-set restriction, no tier I/O).
    fn recall_active(&self) -> bool {
        self.kv_recall && self.recall_tier.is_some() && self.full_attn_kv.is_some()
    }

    /// One DEFAULT paged prefill row (no recall cycle): self-allocate the prompt
    /// tokens into the shared `full_attn_kv` pool, build the FULL-resident page
    /// table (`PageMeta::for_slot`), and run the paged forward over it. This is
    /// the dense-Qwen3 model applied to Qwen3.6 — full attention over every
    /// resident page, no eviction/scoring/prefetch. Mirrors the prefill steps of
    /// [`Self::prefill_row_recall`] but stops after the forward.
    fn prefill_row_paged_default(
        &mut self,
        row: &infer_plan::PrefillRow,
        position: u64,
    ) -> Result<u32> {
        let slot = row.slot;
        if std::env::var_os("ARLE_KVDRIFT_DEBUG").is_some() {
            let pool_len = self
                .full_attn_kv
                .as_ref()
                .map_or(usize::MAX, |p| p.seq_len(slot));
            eprintln!(
                "[kvdrift] PREFILL slot={} start_pos={} tokens={} total={} slot.seq_len={} pool.seq_len={}",
                slot,
                row.start_pos,
                row.tokens.len(),
                row.total_tokens,
                self.slots[slot].seq_len(),
                pool_len,
            );
        }
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (full_attn_paged)");
            // The pool must hold exactly `start_pos` tokens before appending the
            // tail. `free_slot` runs at `start_pos == 0` in `submit_prefill_row`;
            // prefix reuse (start_pos > 0) is gated off for hybrid models by
            // `reusable_prefix_blocks` returning 0, so this path always sees
            // start_pos == 0. A mismatch here would mean a soundness gate failed.
            ensure!(
                pool.seq_len(slot) == row.start_pos,
                "Qwen3.6 default-paged prefill: device pool seq_len {} != start_pos {} for slot {} \
                 (reusable_prefix_blocks soundness gate bypassed)",
                pool.seq_len(slot),
                row.start_pos,
                slot
            );
            pool.alloc_tokens(slot, row.tokens.len())?;
        }
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            crate::loader::PageMeta::for_slot(
                &self.model.ctx,
                pool,
                slot,
                row.start_pos,
                row.tokens.len(),
            )?
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        let Some(ds) = dspark.as_mut() else {
            return model.forward_tokens_recall(
                &mut slots[slot],
                workspace,
                &row.tokens,
                row.start_pos,
                &row.params,
                position,
                &mut rc,
            );
        };
        // DSpark: capture the trunk taps, extend the draft ctx cache with this
        // chunk's rows, and stage the pending anchor on the final chunk.
        ds.taps.prepare(
            ds.head.target_layer_ids(),
            model.config.hidden_size,
            row.tokens.len(),
        );
        let token = model.forward_tokens_recall_tapped(
            &mut slots[slot],
            workspace,
            &row.tokens,
            row.start_pos,
            &row.params,
            position,
            &mut rc,
            Some(&mut ds.taps),
        )?;
        if ds.slots[slot].is_none() {
            ds.slots[slot] = Some(crate::qwen35::dspark::Qwen35DsparkSlotState::new(
                &model.ctx, &ds.head,
            )?);
        }
        let df = ds.slots[slot].as_mut().expect("built above");
        if df.ctx_end != row.start_pos {
            // Gap (prefix-cache / sidecar-restored prefix): rebase the draft
            // ctx at the suffix instead of degrading to plain decode — exact
            // for sliding draft layers once the tail ≥ window, approximate
            // only for the full-attention layer.
            df.rebase(row.start_pos);
        }
        model.dspark_append_ctx(
            &ds.head,
            df,
            &mut ds.taps,
            &mut ds.scratch,
            row.tokens.len(),
            row.start_pos,
        )?;
        let is_final = row.start_pos + row.tokens.len() == row.total_tokens;
        df.pending = (is_final && df.ctx_end == row.total_tokens).then_some(token);
        Ok(token)
    }

    /// One DEFAULT paged decode row (no recall cycle): append this step's token
    /// to the shared `full_attn_kv` pool and attend the FULL resident page set
    /// (`PageMeta::for_slot`). Mirrors [`Self::decode_row_recall`] but without
    /// the working-set restriction and without any tier I/O.
    fn decode_row_paged_default(&mut self, row: &DecodeRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (full_attn_paged)");
            // The pool must already hold exactly this step's cache length before
            // appending the new token; a mismatch means an upstream prefill left
            // the slot's `seq_len` inconsistent with the engine's `kv_seq_len`
            // (e.g. a leaked radix-reuse prefill — see `prefill_row_paged_default`).
            // Catch it at the append, not via `PageMeta::for_slot` math downstream.
            ensure!(
                pool.seq_len(slot) == row.kv_seq_len,
                "Qwen3.6 default-paged decode: pool seq_len {} != kv_seq_len {} for slot {}",
                pool.seq_len(slot),
                row.kv_seq_len,
                slot
            );
            pool.alloc_tokens(slot, 1)?;
        }
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            crate::loader::PageMeta::for_slot(&self.model.ctx, pool, slot, row.kv_seq_len, 1)?
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        model.forward_tokens_recall(
            &mut slots[slot],
            workspace,
            &[row.last_token],
            row.kv_seq_len,
            &row.params,
            position,
            &mut rc,
        )
    }

    /// One DSpark decode row: the block spec step when the slot is seeded
    /// (pending anchor + contiguous draft ctx), else a tapped warm step that
    /// re-seeds it. Emits 1..=block_size+1 tokens for this slot.
    fn dspark_decode_row(&mut self, row: &DecodeRow) -> Result<Vec<SlotToken>> {
        ensure!(
            row.slot < self.num_slots,
            "decode slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(
            self.slots[row.slot].seq_len() == row.kv_seq_len,
            "Qwen3.5 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
            self.slots[row.slot].seq_len(),
            row.kv_seq_len,
            row.slot
        );
        let position = row.kv_seq_len.saturating_add(1) as u64;
        let start = row.kv_seq_len;
        let ds = self
            .dspark
            .as_ref()
            .expect("dspark_decode_row without dspark");
        // The verify appends block_size+1 rows, so the whole chain must fit the
        // trunk cap; near the cap, degrade to single-token steps. Greedy rows
        // verify by argmax match; sampling rows by rejection sampling.
        let seeded = self.full_attn_paged()
            && start + ds.head.block_size() < self.model.max_seq_len()
            && matches!(
                ds.slots[row.slot].as_ref(),
                Some(s) if s.pending == Some(row.last_token) && s.ctx_end == start
            );
        if !seeded {
            let token = self.dspark_warm_decode_row(row, position)?;
            return Ok(vec![SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            }]);
        }
        self.dspark_spec_row(row)
    }

    /// One MTP spec-decode row (`--spec-type mtp`): the spec draft/verify step
    /// when the slot is seeded (pending token + hidden captured by the warm
    /// step), else a warm forward that captures the seed. Emits 1..=depth+1
    /// tokens for this slot. Greedy-only (sampling rows fall back to no-spec).
    fn mtp_decode_row(&mut self, row: &DecodeRow) -> Result<Vec<SlotToken>> {
        ensure!(
            row.slot < self.num_slots,
            "decode slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        // MTP spec-decode is greedy-only; sampling rows fall back to no-spec.
        if row.params.temperature != 0.0 {
            let token = self.submit_decode_row(row, /* allow_graph = */ true)?;
            return Ok(vec![SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            }]);
        }
        let depth = self.model.spec_draft_tokens().max(1);
        // Seeded iff we have a stored pending+hidden AND the pending matches the
        // token the scheduler will feed (a gap = re-seed at the warm step).
        let seeded = matches!(
            self.mtp.as_ref().and_then(|m| m.slots[row.slot].as_ref()),
            Some(s) if s.pending == row.last_token
        );
        if !seeded {
            let token = self.mtp_warm_decode_row(row)?;
            return Ok(vec![SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            }]);
        }
        self.mtp_spec_row(row, depth)
    }

    /// Warm step: a single forward that ALSO captures the seed hidden + pending
    /// (greedy argmax) for the next spec step. Returns the pending token as the
    /// decode output (the prefill already returned the first token; this is the
    /// second). The spec step starts from the stored (pending, hidden).
    fn mtp_warm_decode_row(&mut self, row: &DecodeRow) -> Result<u32> {
        let slot = row.slot;
        let start = row.kv_seq_len;
        if self.full_attn_paged() {
            {
                let pool = self.full_attn_kv.as_mut().expect("paged (gated)");
                ensure!(
                    pool.seq_len(slot) == start,
                    "MTP warm decode: pool seq_len {} != start {start} for slot {slot}",
                    pool.seq_len(slot),
                );
                pool.alloc_tokens(slot, 1)?;
            }
            let meta = {
                let pool = self.full_attn_kv.as_ref().expect("paged (gated)");
                crate::loader::PageMeta::for_slot(&self.model.ctx, pool, slot, start, 1)?
            };
            let Self {
                model,
                slots,
                workspace,
                full_attn_kv,
                mtp,
                ..
            } = self;
            let pool = full_attn_kv.as_mut().expect("paged (gated)");
            let mut rc = crate::qwen35::Qwen35RecallForward {
                pool,
                meta: &meta,
                layer0_query: Vec::new(),
            };
            let (logits, dims, hidden) = model.forward_tokens_with_hidden(
                &mut slots[slot],
                workspace,
                &[row.last_token],
                start,
                Some(&mut rc),
            )?;
            let vocab = dims[1];
            let mut spec = model.new_spec_slot_state()?;
            let pending = crate::ops::argmax_row_into(
                &model.ctx,
                &logits,
                dims[0] - 1,
                vocab,
                spec.argmax_scratch_mut(),
            )?;
            mtp.as_mut().expect("mtp (gated)").slots[slot] = Some(MtpSlotState {
                spec,
                pending,
                hidden,
            });
            Ok(pending)
        } else {
            let Self {
                model,
                slots,
                workspace,
                mtp,
                ..
            } = self;
            let (logits, dims, hidden) = model.forward_tokens_with_hidden(
                &mut slots[slot],
                workspace,
                &[row.last_token],
                start,
                None,
            )?;
            let vocab = dims[1];
            let mut spec = model.new_spec_slot_state()?;
            let pending = crate::ops::argmax_row_into(
                &model.ctx,
                &logits,
                dims[0] - 1,
                vocab,
                spec.argmax_scratch_mut(),
            )?;
            mtp.as_mut().expect("mtp (gated)").slots[slot] = Some(MtpSlotState {
                spec,
                pending,
                hidden,
            });
            Ok(pending)
        }
    }

    /// Spec step: draft a depth-token chain, verify in ONE trunk forward, accept
    /// the longest matching prefix. On partial accept the trunk linear state is
    /// rolled back + the paged pool truncated to the accepted prefix. Returns the
    /// emitted tokens (accepted drafts + bonus); updates the seed state for the
    /// next step.
    fn mtp_spec_row(&mut self, row: &DecodeRow, depth: usize) -> Result<Vec<SlotToken>> {
        let slot = row.slot;
        let start = row.kv_seq_len;
        // The verify forward appends depth+1 tokens; allocate the paged pool
        // upfront (paged path only). On partial accept we truncate the excess.
        if self.full_attn_paged() {
            let pool = self.full_attn_kv.as_mut().expect("paged (gated)");
            ensure!(
                pool.seq_len(slot) == start,
                "MTP spec decode: pool seq_len {} != start {start} for slot {slot}",
                pool.seq_len(slot),
            );
            pool.alloc_tokens(slot, depth + 1)?;
        }
        let meta = if self.full_attn_paged() {
            let pool = self.full_attn_kv.as_ref().expect("paged (gated)");
            Some(crate::loader::PageMeta::for_slot(
                &self.model.ctx,
                pool,
                slot,
                start,
                depth + 1,
            )?)
        } else {
            None
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            mtp,
            ..
        } = self;
        let (emitted, next_pending, next_hidden) = {
            let mtp_exec = mtp.as_mut().expect("mtp (gated)");
            let st = mtp_exec.slots[slot].as_mut().expect("seeded (gated)");
            let mut rc = full_attn_kv
                .as_mut()
                .map(|pool| crate::qwen35::Qwen35RecallForward {
                    pool,
                    meta: meta.as_ref().expect("paged (gated)"),
                    layer0_query: Vec::new(),
                });
            model.spec_step(
                &mut slots[slot],
                &mut st.spec,
                workspace,
                st.pending,
                &st.hidden,
                start,
                depth,
                rc.as_mut(),
            )?
        };
        // Partial accept: truncate the paged pool to the accepted prefix.
        if full_attn_kv.is_some() && emitted.len() < depth + 1 {
            full_attn_kv
                .as_mut()
                .expect("paged (gated)")
                .truncate_slot(slot, start + emitted.len())?;
        }
        // Update the seed for the next spec step.
        if let Some(st) = mtp.as_mut().expect("mtp (gated)").slots[slot].as_mut() {
            st.pending = next_pending;
            st.hidden = next_hidden;
        }
        Ok(emitted
            .into_iter()
            .map(|token| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// with tap capture, extending the ctx cache + pending anchor when the
    /// cache is contiguous. Non-paged rows run plain and clear the draft state
    /// (speculation re-seeds at the next paged step).
    fn dspark_warm_decode_row(&mut self, row: &DecodeRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        if !self.full_attn_paged() {
            if let Some(df) = self.dspark.as_mut().and_then(|ds| ds.slots[slot].as_mut()) {
                df.pending = None;
            }
            return self.submit_decode_row(row, false);
        }
        {
            let pool = self.full_attn_kv.as_mut().expect("paged (checked)");
            ensure!(
                pool.seq_len(slot) == row.kv_seq_len,
                "Qwen3.6 dspark warm decode: pool seq_len {} != kv_seq_len {} for slot {}",
                pool.seq_len(slot),
                row.kv_seq_len,
                slot
            );
            pool.alloc_tokens(slot, 1)?;
        }
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("paged (checked)");
            crate::loader::PageMeta::for_slot(&self.model.ctx, pool, slot, row.kv_seq_len, 1)?
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("paged (checked)");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        let ds = dspark.as_mut().expect("dspark warm without dspark");
        // Gap (whole-slot promote / restored prefix): rebase at the current
        // position and start rebuilding the suffix-only ctx from this step.
        if let Some(df) = ds.slots[slot].as_mut()
            && df.ctx_end != row.kv_seq_len
        {
            df.rebase(row.kv_seq_len);
        }
        let taps = if ds.slots[slot].is_some() {
            ds.taps
                .prepare(ds.head.target_layer_ids(), model.config.hidden_size, 1);
            Some(&mut ds.taps)
        } else {
            None
        };
        let token = model.forward_tokens_recall_tapped(
            &mut slots[slot],
            workspace,
            &[row.last_token],
            row.kv_seq_len,
            &row.params,
            position,
            &mut rc,
            taps,
        )?;
        if let Some(df) = ds.slots[slot].as_mut() {
            model.dspark_append_ctx(
                &ds.head,
                df,
                &mut ds.taps,
                &mut ds.scratch,
                1,
                row.kv_seq_len,
            )?;
            df.pending = Some(token);
        }
        Ok(token)
    }

    /// One DSpark block spec step: draft a block from the draft head's ctx
    /// cache, verify the chain in ONE trunk forward over the paged pool
    /// (linear-capture + taps), accept the longest matching prefix, roll the
    /// trunk back on partial accept, crop the pool, and extend the draft ctx
    /// with the accepted rows' features. Greedy rows are token-exact to no-spec
    /// decode (every committed token is a trunk argmax); sampling rows commit
    /// rejection-sampled tokens distributed exactly as filtered target sampling.
    fn dspark_spec_row(&mut self, row: &DecodeRow) -> Result<Vec<SlotToken>> {
        let slot = row.slot;
        let start = row.kv_seq_len;
        if self.dspark.as_ref().expect("dspark").spec[slot].is_none() {
            let st = self.model.new_spec_slot_state()?;
            self.dspark.as_mut().expect("dspark").spec[slot] = Some(st);
        }
        let mut pt = crate::qwen35::dspark_phase_start(&self.model.ctx);
        // 1. Draft the block (draft-head-only; no trunk/pool state touched —
        //    the speculative noise K/V rows in the draft cache self-heal).
        let (chain, partial_ctx) = {
            let Self { model, dspark, .. } = self;
            let ds = dspark.as_mut().expect("dspark");
            let df = ds.slots[slot].as_mut().expect("seeded slot");
            let partial_ctx = df.ctx_base > 0;
            let chain = model.dspark_draft_block(
                &ds.head,
                df,
                &mut ds.scratch,
                row.last_token,
                start,
                &row.params,
            )?;
            (chain, partial_ctx)
        };
        let draft_ms = crate::qwen35::mtp_phase_lap(&self.model.ctx, &mut pt);
        if chain.len() < 2 {
            // Confidence head rejected the whole block — warm step instead.
            let position = start.saturating_add(1) as u64;
            let token = self.dspark_warm_decode_row(row, position)?;
            return Ok(vec![SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            }]);
        }

        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let ds = dspark.as_mut().expect("dspark");
        let spec = ds.spec[slot].as_mut().expect("built above");
        // 2. Snapshot the trunk linear state (partial-accept rollback base).
        spec.snapshot_trunk(&model.ctx, &slots[slot])?;
        let snap_ms = crate::qwen35::mtp_phase_lap(&model.ctx, &mut pt);
        // 3. Verify the chain over the paged pool, capturing linear inputs
        //    (replay substrate) + trunk taps (ctx features of accepted rows).
        let pool = full_attn_kv.as_mut().expect("paged (gated by seeded)");
        ensure!(
            pool.seq_len(slot) == start,
            "Qwen3.6 dspark verify: pool seq_len {} != start {start} for slot {slot}",
            pool.seq_len(slot)
        );
        pool.alloc_tokens(slot, chain.len())?;
        let meta = crate::loader::PageMeta::for_slot(&model.ctx, pool, slot, start, chain.len())?;
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        ds.taps.prepare(
            ds.head.target_layer_ids(),
            model.config.hidden_size,
            chain.len(),
        );
        let logits = model.dspark_verify_logits(
            &mut slots[slot],
            workspace,
            &chain,
            start,
            spec,
            &mut rc,
            &mut ds.taps,
        )?;
        let verify_ms = crate::qwen35::mtp_phase_lap(&model.ctx, &mut pt);
        // 4. Accept scan + trunk rollback (linear replay; full-attn KV
        //    self-heals under the pool crop below).
        let (emitted, bonus, k) = if row.params.is_greedy() {
            model.dspark_accept_commit(&mut slots[slot], spec, workspace, &chain, &logits, start)?
        } else {
            model.dspark_accept_commit_sampled(
                &mut slots[slot],
                spec,
                workspace,
                &ds.head,
                &mut ds.scratch,
                &chain,
                &logits,
                start,
                &row.params,
            )?
        };
        drop(rc);
        if k + 1 < chain.len() {
            full_attn_kv
                .as_mut()
                .expect("paged (gated by seeded)")
                .truncate_slot(slot, start + k + 1)?;
        }
        let accept_ms = crate::qwen35::mtp_phase_lap(&model.ctx, &mut pt);
        // 5. Extend the draft ctx cache with the accepted rows' features and
        //    stage the bonus as the next block's anchor.
        let df = ds.slots[slot].as_mut().expect("seeded slot");
        model.dspark_append_ctx(&ds.head, df, &mut ds.taps, &mut ds.scratch, k + 1, start)?;
        let append_ms = crate::qwen35::mtp_phase_lap(&model.ctx, &mut pt);
        if pt.is_some() {
            eprintln!(
                "[dspark-phase] chain={} accept={k} base={} draft={draft_ms:.2} snap={snap_ms:.2} verify={verify_ms:.2} accept_commit={accept_ms:.2} append={append_ms:.2} ms",
                chain.len(),
                df.ctx_base
            );
        }
        df.pending = Some(bonus);
        ds.accepts += k;
        ds.rejects += chain.len() - 1 - k;
        ds.chains += 1;
        ds.partial_ctx_chains += usize::from(partial_ctx);
        Ok(emitted
            .into_iter()
            .map(|token| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// One prefill row over the recall pool (`--kv-recall`). The ONLY place the whole
    /// recall cycle runs (write-through model: "decode 不召回; prefetch 只在 prefill").
    ///
    /// 1. Self-allocate tokens, run paged forward — captures layer-0 query for scoring.
    /// 2. Score history (`recompute_recall_plan`): select evict + prefetch pages.
    /// 3. Batched-H2D prefetch tier-resident blocks, resolve fixed `recall_pages` working set.
    /// 4. Write-back-evict cold middle to tier, free physical pages immediately (no in-flight attn race).
    ///
    /// After return, `recall[slot].recall_pages()` is the immutable working set for decode.
    fn prefill_row_recall(&mut self, row: &infer_plan::PrefillRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        let cfg = self.recall_cfg;
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (recall_active)");
            pool.alloc_tokens(slot, row.tokens.len())?;
        }
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            crate::loader::PageMeta::for_slot(
                &self.model.ctx,
                pool,
                slot,
                row.start_pos,
                row.tokens.len(),
            )?
        };
        // (1) Forward. Borrow split: &model + &mut slot + &mut workspace + &mut pool (via rc). `rc` carries back layer-0 query.
        let (token, layer0_query) = {
            let Self {
                model,
                slots,
                workspace,
                full_attn_kv,
                ..
            } = self;
            let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
            let mut rc = crate::qwen35::Qwen35RecallForward {
                pool,
                meta: &meta,
                layer0_query: Vec::new(),
            };
            let token = model.forward_tokens_recall(
                &mut slots[slot],
                workspace,
                &row.tokens,
                row.start_pos,
                &row.params,
                position,
                &mut rc,
            )?;
            (token, rc.layer0_query)
        };

        let cache_len = row.start_pos + row.tokens.len();
        let ps = self.full_attn_kv.as_ref().expect("full_attn_kv").page_size;
        ensure!(
            cfg.n_init.is_multiple_of(ps)
                && cfg.n_local.is_multiple_of(ps)
                && cfg.l_bs.is_multiple_of(ps),
            "KV-recall config (n_init {}, n_local {}, l_bs {}) must be multiples of page_size {}",
            cfg.n_init,
            cfg.n_local,
            cfg.l_bs,
            ps
        );

        // Score history against prefill query, plan fixed working set. `allow_prefetch=true` lets tier-resident blocks re-enter.
        let num_q_heads = self.model.local_q_heads();
        let num_kv_heads = self.model.local_kv_heads();
        let head_dim = self.model.config.head_dim;
        let (evict_pages, prefetch_pages) = {
            let Self {
                recall,
                full_attn_kv,
                model,
                ..
            } = self;
            let pool = full_attn_kv.as_ref().expect("full_attn_kv");
            if let Some(state) = recall.get_mut(slot) {
                state.recompute_recall_plan(
                    &model.ctx,
                    pool,
                    slot,
                    cache_len,
                    &cfg,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    &layer0_query,
                    /* allow_prefetch = */ true,
                )?;
                (state.take_evict_pages(), state.take_prefetch_pages())
            } else {
                (Vec::new(), Vec::new())
            }
        };

        // Prefetch tier-resident blocks back into HBM. Decode never does this — one batched prefill sync point.
        for logical in prefetch_pages {
            let key = tier_block_u64(slot as u64, logical as u64);
            let Some(tier) = self.recall_tier.as_mut() else {
                break;
            };
            let payload = match tier.read(key) {
                Ok(p) => p.into_owned(),
                Err(_) => continue,
            };
            if let Some(pool) = self.full_attn_kv.as_mut()
                && let Some(new_page) = pool.reinstate_slot_page(slot, logical)
            {
                pool.copy_pages_from_host(&self.model.ctx, &[new_page], &payload)?;
            }
        }

        if let Some(state) = self.recall.get_mut(slot)
            && let Some(pool) = self.full_attn_kv.as_ref()
        {
            state.resolve_recall_pages(pool, slot);
        }

        // Write-back-evict cold middle: mirror to L3, free physical pages. Prefill drained compute stream — no in-flight attn race.
        for logical in evict_pages {
            let physical = {
                let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                pool.page_indices(slot)
                    .get(logical)
                    .copied()
                    .filter(|&p| p != cuda_kernels::prelude::EVICTED_PAGE)
            };
            let Some(physical) = physical else {
                continue;
            };
            let key = tier_block_u64(slot as u64, logical as u64);
            let mirrored = {
                let payload = {
                    let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                    pool.copy_pages_to_host(&self.model.ctx, &[physical])?
                };
                match self.recall_tier.as_mut() {
                    Some(tier) if !tier.is_full() => tier.insert(key, payload),
                    _ => false,
                }
            };
            if !mirrored {
                continue; // tier full → keep page resident (no KV loss)
            }
            if let Some(pool) = self.full_attn_kv.as_mut() {
                pool.evict_slot_page(slot, logical);
            }
        }
        Ok(token)
    }

    /// One decode row over the recall pool (`--kv-recall`): append + attend, ZERO tier I/O.
    /// Working set fixed at prefill. Alloc token, read fixed `recall_pages`, forward + sample.
    fn decode_row_recall(&mut self, row: &DecodeRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (recall_active)");
            pool.alloc_tokens(slot, 1)?;
        }
        let cache_len = row.kv_seq_len + 1;
        let recall_pages: Vec<u32> = match self.recall.get(slot).and_then(|s| s.recall_pages()) {
            Some(p) => p.to_vec(),
            None => {
                let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                let num_pages = cache_len.div_ceil(pool.page_size);
                pool.page_indices(slot)[..num_pages].to_vec()
            }
        };
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
            crate::loader::PageMeta::for_recall_decode(
                &self.model.ctx,
                pool,
                cache_len,
                &recall_pages,
            )?
        };
        // Forward (paged decode) + sample. `rc.layer0_query` collected but unused on decode.
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        model.forward_tokens_recall(
            &mut slots[slot],
            workspace,
            &[row.last_token],
            row.kv_seq_len,
            &row.params,
            position,
            &mut rc,
        )
    }

    /// Run one decode step through the captured whole-step graph. Returns
    /// `Ok(None)` for eager fallback (gate off, out-of-budget, or capture/replay
    /// failure which permanently downgrades to eager, never fatal).
    ///
    /// See [`crate::qwen35::Qwen35Model::forward_decode_step_captured`] for
    /// capture-safety table and perf formula.
    fn try_graph_decode(&mut self, row: &DecodeRow, position: u64) -> Result<Option<u32>> {
        if !self.decode_graph_armed {
            return Ok(None);
        }
        // Budget check must run every step — host ensures inside captured closure
        // only run on warm/capture steps. Out-of-budget falls back to eager.
        if row.kv_seq_len + 1 > self.model.max_seq_len() {
            return Ok(None);
        }
        if self.decode_graph.is_none() {
            self.decode_graph = Some(Qwen35DecodeGraph::new(
                self.num_slots,
                &self.model.ctx.stream,
            ));
        }
        // Split borrows: capture closure needs &model + &mut slot + &mut workspace.
        let Self {
            model,
            slots,
            decode_graph,
            ..
        } = self;
        let dg = decode_graph
            .as_mut()
            .expect("decode_graph built above when armed");
        let Qwen35DecodeGraph { ws, graphs, baked } = dg;
        let slot_idx = row.slot;

        // Stage per-step device scalars outside graph (dense stage1_write pattern).
        let (token_ids_ptr, start_pos_ptr) =
            model.stage_step_inputs(ws, &[row.last_token], row.kv_seq_len)?;
        let logits_ptr = model.workspace_logits_ptr(ws)?;
        let bake = Qwen35GraphBake {
            token_ids_ptr,
            start_pos_ptr,
            logits_ptr,
            ws_epoch: ws.epoch(),
        };
        match baked[slot_idx] {
            Some(prev) if prev != bake => {
                // Workspace addresses drifted since capture (release → re-alloc).
                // Drop and re-capture against new addresses.
                info!(
                    "[qwen35-decode-graph] slot {slot_idx}: workspace addresses changed; \
                     dropping stale capture and recapturing"
                );
                graphs[slot_idx] = crate::graph::CudaGraphState::new(model.ctx.stream.clone());
                baked[slot_idx] = Some(bake);
            }
            None => baked[slot_idx] = Some(bake),
            _ => {}
        }

        let state = &mut graphs[slot_idx];
        let was_captured = state.is_captured();
        let will_replay = was_captured && !state.is_armed_warm();
        let slot_state = &mut slots[slot_idx];
        let run = state
            .run_or_capture(|| model.forward_decode_step_captured(slot_state, ws, row.kv_seq_len));
        if let Err(e) = run {
            // Any capture/replay failure downgrades to eager permanently — never fatal.
            // Mid-capture error only recorded kernels, so device state untouched.
            warn!(
                "Qwen3.5 whole-step decode graph failed (slot {slot_idx}), \
                 downgrading to eager forward: {e}"
            );
            self.decode_graph_armed = false;
            self.decode_graph = None;
            return Ok(None);
        }
        // Host-side state advance happens here — captured closure is host-state-free.
        slots[slot_idx].advance_seq_len(1);

        // Reuse-evidence probes (license needs replay counts, not capture-exists).
        let state = &self.decode_graph.as_ref().expect("still present").graphs[slot_idx];
        if !was_captured && state.is_captured() {
            let captures =
                QWEN35_GRAPH_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let keys = self
                .decode_graph
                .as_ref()
                .expect("still present")
                .graphs
                .iter()
                .filter(|g| g.is_captured())
                .count();
            info!(
                "[qwen35-decode-graph] captured slot {slot_idx} \
                 (captures_total={captures}, live_keys={keys}, max_keys={})",
                self.num_slots
            );
        }
        if will_replay {
            let replays =
                QWEN35_GRAPH_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if replays.is_multiple_of(100) {
                info!(
                    "[qwen35-decode-graph] replay_total={replays} captures_total={}",
                    QWEN35_GRAPH_CAPTURES.load(std::sync::atomic::Ordering::Relaxed)
                );
            }
        }

        // Sampling stays OUTSIDE the graph (argmax + D2H + sync), reading the
        // logits the replay (or warm run) just wrote.
        let dg = self.decode_graph.as_mut().expect("still present");
        let token = self
            .model
            .sample_workspace_logits(&mut dg.ws, &row.params, position)?;
        Ok(Some(token))
    }

    /// Offload the model's device weights to host RAM (OPD teacher time-share),
    /// returning the device bytes freed. Per-slot KV / recurrent state is left
    /// resident — only the shared model weights move. The forward workspace is
    /// released too (AFTER the offload's full device sync, so no in-flight
    /// kernel can still reference the dropped buffers) — it may hold
    /// prefill-shaped scratch the student backward wants as headroom.
    pub(crate) fn offload_engine_weights(&mut self) -> Result<usize> {
        self.ensure_not_collective("offload_engine_weights")?;
        let freed = self.model.offload_engine_weights()?;
        self.workspace.release();
        // The batched-decode workspace holds `[*, B]` scratch (incl. MoE) —
        // release it for the same headroom reason. Its pointer TABLES survive:
        // they address per-slot state, which the weight offload leaves
        // resident (see `Qwen35BatchDecodeState::release`).
        if let Some(bd) = self.batch_decode.as_mut() {
            bd.release();
        }
        // Captured decode graphs bake the (now freed/placeholder) weight
        // addresses — drop them wholesale. Re-built + re-captured lazily after
        // reload (the gate flag stays armed).
        self.decode_graph = None;
        Ok(freed)
    }

    /// Release the inference forward scratch (24K-shaped workspace + batched-decode
    /// scratch + captured decode graphs) WITHOUT offloading weights or touching KV.
    /// The freed device blocks return to the shared CUmemAllocAsync caching pool the
    /// co-resident OPD writeback's autograd reuses (OPD `EngineOffloadMode::Off`
    /// rollout->writeback never offloads, so the scratch otherwise OOMs the writeback).
    /// This is `offload_engine_weights`'s scratch-teardown MINUS the weight offload —
    /// weights and per-slot KV / recurrent state stay resident.
    pub(crate) fn release_inference_scratch(&mut self) -> Result<()> {
        self.workspace.release();
        if let Some(bd) = self.batch_decode.as_mut() {
            bd.release();
        }
        self.decode_graph = None;
        Ok(())
    }

    /// Reload the model's device weights from the host snapshot.
    pub(crate) fn reload_engine_weights(&mut self) -> Result<()> {
        self.ensure_not_collective("reload_engine_weights")?;
        self.model.reload_engine_weights()
    }

    /// OPD surfaces are rank-0 control-seam calls: under multi-rank TP they
    /// would run on one rank only, desyncing the per-step NCCL collective
    /// sequence (and, for LoRA/offload, diverging the resident weights across
    /// ranks). Bail loudly instead of silently corrupting the lockstep.
    fn ensure_not_collective(&self, what: &str) -> Result<()> {
        ensure!(
            !self.model.tp.is_collective(),
            "{what} is single-GPU only: the Qwen3.5/3.6 OPD surfaces are not \
             wired for multi-rank tensor parallelism (world_size={})",
            self.model.tp.config().world_size
        );
        Ok(())
    }

    pub(crate) fn submit(&mut self, plan: &ForwardPlan) -> Result<StepOutput> {
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        if rows == 0 {
            return Ok(StepOutput { tokens: Vec::new() });
        }

        // rows == 1 keeps the existing single-row path byte-identical
        // (including the whole-step B=1 decode-graph lane, which is gated to
        // rows==1 PLANS — batched/mixed steps never capture or replay).
        if rows == 1 {
            let (slot, token) = if let Some(row) = plan.prefill_rows.first() {
                (row.slot, self.submit_prefill_row(row)?)
            } else {
                let row = &plan.decode_rows[0];
                if self.dspark.is_some() {
                    return Ok(StepOutput {
                        tokens: self.dspark_decode_row(row)?,
                    });
                }
                if self.mtp.is_some() {
                    return Ok(StepOutput {
                        tokens: self.mtp_decode_row(row)?,
                    });
                }
                (
                    row.slot,
                    self.submit_decode_row(row, /* allow_graph = */ true)?,
                )
            };
            return Ok(StepOutput {
                tokens: vec![SlotToken {
                    slot,
                    token,
                    logprob: None,
                    finish: None,
                }],
            });
        }

        // Multi-row plans (DSv4 executor pattern): per-prefill single-row
        // sub-steps, then ONE decode sub-batch. Plan rows always address
        // disjoint slots (a request is either Prefilling or Decoding), so the
        // sequential sub-steps are math-identical to consecutive single-mode
        // ticks.
        let mut seen_slots = std::collections::BTreeSet::new();
        for slot in plan
            .prefill_rows
            .iter()
            .map(|row| row.slot)
            .chain(plan.decode_rows.iter().map(|row| row.slot))
        {
            ensure!(
                seen_slots.insert(slot),
                "Qwen3.5 plan schedules slot {slot} more than once per tick"
            );
        }

        let mut tokens = Vec::with_capacity(rows);
        for row in &plan.prefill_rows {
            let token = self.submit_prefill_row(row)?;
            tokens.push(SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            });
        }
        if self.dspark.is_some() {
            // DSpark decode is per-row (each row runs its own draft + block
            // verify); batched spec verify is a later increment.
            for row in &plan.decode_rows {
                tokens.extend(self.dspark_decode_row(row)?);
            }
            return Ok(StepOutput { tokens });
        }
        match plan.decode_rows.len() {
            0 => {}
            1 => {
                // Mixed tick with a single decode row: eager (the B=1 graph
                // lane stays gated to rows==1 plans in stage 1).
                let row = &plan.decode_rows[0];
                let token = self.submit_decode_row(row, /* allow_graph = */ false)?;
                tokens.push(SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                });
            }
            _ => tokens.extend(self.submit_decode_batch(&plan.decode_rows)?),
        }
        Ok(StepOutput { tokens })
    }

    /// One prefill row as its own single-row sub-step (the pre-batching
    /// single-row prefill arm, factored).
    fn submit_prefill_row(&mut self, row: &infer_plan::PrefillRow) -> Result<u32> {
        ensure!(
            row.slot < self.num_slots,
            "prefill slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
        // A fresh prefill (start_pos == 0) rewinds this slot's recurrent +
        // conv state and cache cursor before appending. Request-boundary
        // rearm discipline (the e95e11b6 lesson): the slot's captured
        // decode graph stays valid (state buffers are memset, not
        // re-allocated), but its FIRST decode of the new occupant runs one
        // eager warm step so any per-request host work executes without
        // dropping the capture — capture cost stays once per slot, not
        // once per request.
        if row.start_pos == 0 {
            // Request boundary: the prior occupant is finished (the scheduler
            // only reassigns a slot after its request completes — no in-flight
            // forward references the slot), so return its recurrent block to the
            // free-list, then acquire (pop the same block back, or alloc) and
            // zero it for the new occupant. This replaces the old in-place
            // `reset()` zeroing; the block MUST be resident before the forward.
            self.slots[row.slot].release_recurrent(&mut self.recurrent_pool);
            // Drop any prior occupant's pending boundary snapshots: this slot is a
            // new request, so a stale snapshot must never key a future sidecar.
            self.prefill_boundary_snapshot[row.slot] = None;
            self.periodic_boundary_snapshots[row.slot].clear();
            // New occupant: the prior request's draft ctx cache is dead.
            if let Some(df) = self
                .dspark
                .as_mut()
                .and_then(|ds| ds.slots[row.slot].as_mut())
            {
                df.reset();
            }
            // New occupant: drop the prior request's MTP seed (pending+hidden);
            // the warm step re-seeds from the prefill's last token.
            if let Some(mtp) = self.mtp.as_mut() {
                mtp.slots[row.slot] = None;
            }
            let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
            self.slots[row.slot].acquire_recurrent(
                &self.model.ctx,
                num_linear,
                gdr_len,
                conv_len,
                &mut self.recurrent_pool,
            )?;
            // The new block's `gdr_states`/`conv_states` addresses differ from the
            // prior occupant's, so the batched-decode pointer-table cache (keyed
            // on slot_indices only) must restage on the next decode batch.
            if let Some(bd) = self.batch_decode.as_mut() {
                bd.invalidate_staged_pointers();
            }
            // The captured decode graph (legacy contiguous lane) bakes this slot's
            // recurrent-block addresses; a different block invalidates it. Drop the
            // slot's capture (the `baked` staleness check only tracks workspace
            // ptrs, not recurrent ptrs) so it re-captures against the new block —
            // `rearm_warm` alone keeps the stale capture and would replay freed mem.
            if let Some(dg) = self.decode_graph.as_mut() {
                dg.graphs[row.slot] =
                    crate::graph::CudaGraphState::new(self.model.ctx.stream.clone());
                dg.baked[row.slot] = None;
            }
            // Default paged path: free the prior occupant's pages back to the
            // shared pool so a fresh prefill starts at logical page 0. (Under
            // recall the keepalive release below also runs.)
            if self.recall_active() {
                self.recall[row.slot].reset();
                // Release any pages still parked in the eviction keepalive from the
                // prior occupant's last decode step BEFORE freeing the slot, so the
                // detached physical pages rejoin the pool instead of leaking (they
                // are sentinels in the table, so `free_slot` alone would not recycle
                // them). The prior request finished, so its attention has long since
                // completed — no race with the one-step deferral.
                let parked = std::mem::take(&mut self.recall_keepalive[row.slot]);
                if let Some(pool) = self.full_attn_kv.as_mut() {
                    for (_logical, physical) in parked {
                        pool.release_evicted_page(physical);
                    }
                    pool.free_slot(row.slot);
                }
                // Stale L3 tier entries keyed by (slot, logical) from the prior
                // occupant are left to the store's LRU/capacity eviction: re-recall
                // only reads a block whose resident REP exists, and `reset()` just
                // cleared all reps, so a fresh occupant can never read a prior
                // occupant's tier block before overwriting that key — tenant-safe
                // without a per-session key registry (the dense arm's
                // `drop_tier_session` makes the same call, see its doc).
            } else if let Some(pool) = self.full_attn_kv.as_mut() {
                // Default paged (no recall): just recycle the slot's pages.
                pool.free_slot(row.slot);
            }
        }
        let position = (row.start_pos + row.tokens.len()) as u64;
        if self.recall_active() {
            return self.prefill_row_recall(row, position);
        }
        // Default paged prefill: full attention over all resident pages, no
        // eviction (the dense model applied to Qwen3.6 — Phase 2).
        if self.slots[row.slot].has_recurrent() {
            return self.prefill_row_snapshotted(row, position);
        }
        self.prefill_row_paged_default(row, position)
    }

    /// Prefill a hybrid row, splitting the forward at recurrent-snapshot
    /// boundaries so a later prefix hit can restore the linear-attn state.
    ///
    /// Recurrent state is only observable at a forward's END, so a snapshot keyed
    /// at position `S` requires a forward that ends at `S` (else keying a
    /// later-position state at `S` bakes the residue `[S..end]` and double-advances
    /// it on restore). Two families of boundary:
    /// - **`L*` = align_down16(total - 1)** (final chunk only) — the exact-resend
    ///   restore target (popped for aligned, un-popped for non-aligned; see
    ///   `prefix.rs:72`). Owns `prefill_boundary_snapshot` (the within-conversation
    ///   fast path), unchanged.
    /// - **Periodic stride multiples `S = k * SIDECAR_SNAPSHOT_STRIDE_PAGES * 16`**
    ///   crossed in `(start, end)` — for a future CROSS-conversation restore. Each
    ///   feeds `periodic_boundary_snapshots`, saved at publish.
    ///
    /// Every cut is `< end`, so a real tail always remains to forward with the
    /// real `position` and return the sampled token. All cuts are page-aligned
    /// (stride is a page multiple, `L*` is `align16`), so `hash(tokens[..S])`
    /// rendezvous with the restore probe's page-aligned boundaries.
    fn prefill_row_snapshotted(
        &mut self,
        row: &infer_plan::PrefillRow,
        position: u64,
    ) -> Result<u32> {
        let start = row.start_pos;
        let end = start + row.tokens.len();
        let is_final = end == row.total_tokens;
        let lstar = row.total_tokens.saturating_sub(1) / SUPPORTED_PAGE_SIZE * SUPPORTED_PAGE_SIZE;
        let stride = SIDECAR_SNAPSHOT_STRIDE_PAGES * SUPPORTED_PAGE_SIZE; // const, > 0

        // Ordered snapshot cuts in `[start, end)`. `bool` = is-`L*` (routes to the
        // exact-resend `prefill_boundary_snapshot`; a boundary coinciding with `L*`
        // is marked here so the L* insert below dedups it).
        let mut cuts: Vec<(usize, bool)> = Vec::new();
        // Boundary at `start`: a prior chunk ended exactly on a stride multiple, so
        // that boundary is never `< end` of any chunk — snapshot the already-
        // materialized state here (no forward). Without this, chunking aligned to
        // the stride (e.g. chunked_prefill_size == stride) would snapshot nothing.
        if start > 0 && start.is_multiple_of(stride) {
            cuts.push((start, is_final && start == lstar));
        }
        // Interior stride multiples (strictly `> start`, `< end`).
        let mut s = (start / stride + 1) * stride;
        while s < end {
            cuts.push((s, is_final && s == lstar));
            s += stride;
        }
        // `L*` on the final chunk (non-aligned residue can land on `start`).
        if is_final
            && lstar > 0
            && lstar >= start
            && lstar < end
            && !cuts.iter().any(|&(p, _)| p == lstar)
        {
            cuts.push((lstar, true));
        }
        cuts.sort_unstable_by_key(|&(p, _)| p);

        // Forward each `[cursor..cut]` segment (state materialized at `cut`),
        // snapshot there, then the final `[cursor..end]` tail returns the token.
        let mut cursor = start;
        let mut did_cut = false;
        for (cut, is_lstar) in cuts {
            if cut > cursor {
                let seg = infer_plan::PrefillRow {
                    slot: row.slot,
                    tokens: row.tokens[cursor - start..cut - start].to_vec(),
                    start_pos: cursor,
                    total_tokens: row.total_tokens,
                    params: row.params.clone(),
                };
                self.prefill_row_paged_default(&seg, cut as u64)?; // token discarded
                cursor = cut;
            }
            // State is now materialized at exactly `cut` (either forwarded to it,
            // or `cut == start` already reached by a prior chunk).
            let snap = self.slots[row.slot].snapshot_recurrent(&self.model.ctx)?;
            if is_lstar {
                self.prefill_boundary_snapshot[row.slot] = Some((cut, snap));
            } else {
                self.periodic_boundary_snapshots[row.slot].push((cut, snap));
            }
            did_cut = true;
        }
        if !did_cut {
            return self.prefill_row_paged_default(row, position); // no boundary crossed
        }
        // Final tail `[cursor..end]` (cuts are all `< end`, so this is non-empty).
        let tail = infer_plan::PrefillRow {
            slot: row.slot,
            tokens: row.tokens[cursor - start..].to_vec(),
            start_pos: cursor,
            total_tokens: row.total_tokens,
            params: row.params.clone(),
        };
        self.prefill_row_paged_default(&tail, position)
    }

    /// One decode row as a single-row forward (the pre-batching single-row
    /// decode arm, factored). `allow_graph` admits the whole-step B=1
    /// decode-graph lane — true only for rows==1 plans.
    fn submit_decode_row(&mut self, row: &DecodeRow, allow_graph: bool) -> Result<u32> {
        ensure!(
            row.slot < self.num_slots,
            "decode slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        if std::env::var_os("ARLE_KVDRIFT_DEBUG").is_some()
            && self.slots[row.slot].seq_len() != row.kv_seq_len
        {
            let pool_len = self
                .full_attn_kv
                .as_ref()
                .map_or(usize::MAX, |p| p.seq_len(row.slot));
            eprintln!(
                "[kvdrift] DECODE-ASSERT slot={} slot.seq_len={} row.kv_seq_len={} pool.seq_len={} (Δslot-row={})",
                row.slot,
                self.slots[row.slot].seq_len(),
                row.kv_seq_len,
                pool_len,
                self.slots[row.slot].seq_len() as i64 - row.kv_seq_len as i64,
            );
        }
        ensure!(
            self.slots[row.slot].seq_len() == row.kv_seq_len,
            "Qwen3.5 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
            self.slots[row.slot].seq_len(),
            row.kv_seq_len,
            row.slot
        );
        let position = row.kv_seq_len.saturating_add(1) as u64;
        // Session KV-recall (opt-in): decode reads the recall-restricted page
        // table (the fixed working set chosen at prefill). The seq_len invariant
        // above still holds — the slot's seq_len is advanced in lockstep inside
        // the recall forward.
        if self.recall_active() {
            return self.decode_row_recall(row, position);
        }
        // Default paged decode: append + attend the full resident page set. The
        // whole-step decode-graph lane bakes per-step device addresses (the page
        // table grows each step), so it is bypassed under the paged default —
        // the paged forward IS the correctness floor; the graph lane (kept below
        // for the legacy contiguous path) is a contiguous-cache optimization a
        // later phase re-enables over a stable paged table.
        if self.full_attn_paged() {
            return self.decode_row_paged_default(row, position);
        }
        // Legacy contiguous path (no paged pool — e.g. OPD weight offload dropped
        // it): the whole-step graph lane first (opt-in), with eager
        // `forward_tokens` as the correctness floor / graph-miss fallback.
        let graph_token = if allow_graph {
            self.try_graph_decode(row, position)?
        } else {
            None
        };
        match graph_token {
            Some(token) => Ok(token),
            None => self.model.forward_tokens(
                &mut self.slots[row.slot],
                &mut self.workspace,
                &[row.last_token],
                row.kv_seq_len,
                &row.params,
                position,
            ),
        }
    }

    /// A rows>1 pure-decode sub-batch: ONE batched forward over all rows
    /// ([`crate::qwen35::Qwen35Model::forward_decode_batch`] — stage 1
    /// re-port of the monolith batched decode; the c=4 amortization formula
    /// and the TP/all-reduce proof live on that method). With
    /// `--qwen35-batched-decode false`, runs the rows sequentially as
    /// single-row forwards instead (the honest A/B arm; the pre-batching
    /// behavior was a hard error, not a fallback).
    ///
    /// Decode-graph interaction (stage 1): batched steps NEVER capture or
    /// replay, and they cannot invalidate existing B=1 captures — the
    /// captured graphs bake (a) per-slot state addresses, which the batched
    /// path mutates strictly IN PLACE through pointer tables over the same
    /// allocations, and (b) the dedicated decode-graph workspace's buffer
    /// addresses, which the batched path never touches (it owns a third,
    /// batch-only workspace). The per-step position is read from a staged
    /// device scalar at replay, so a slot's seq_len advancing via a batched
    /// step replays correctly on its next single-row graphed step.
    fn submit_decode_batch(&mut self, rows: &[DecodeRow]) -> Result<Vec<SlotToken>> {
        debug_assert!(rows.len() > 1);
        // Per-row watermark/validation BEFORE any device mutation (dup-slot
        // ensure ran at plan level in `submit`).
        for row in rows {
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside Qwen3.5 executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                self.slots[row.slot].seq_len() == row.kv_seq_len,
                "Qwen3.5 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
                self.slots[row.slot].seq_len(),
                row.kv_seq_len,
                row.slot
            );
        }

        // Under the shared-paged default (always-on), B>1 decode batches over the
        // shared `full_attn_kv` pool via the PAGED batched kernels (the contiguous
        // `k_caches`/`v_caches` the legacy batched path reads are never allocated).
        // BF16 single-GPU only: the batched-paged HD256 kernels are BF16, and a TP
        // collective per-rank lockstep is handled by `forward_decode_batch_paged`'s
        // all-reduces — but the correctness floor keeps quant-KV and `--kv-recall`
        // (per-row restricted page table) on the serial per-row path.
        if self.full_attn_paged() {
            let paged_bf16 = self
                .full_attn_kv
                .as_ref()
                .map(|p| p.format == KVFormat::BF16)
                .unwrap_or(false);
            if qwen35_batched_decode_enabled()
                && paged_bf16
                && !self.recall_active()
                && self.model.tp.is_single()
            {
                return self.submit_decode_batch_paged(rows);
            }
            // Correctness floor: recall / quant-KV / TP → serial per-row paged.
            let mut tokens = Vec::with_capacity(rows.len());
            for row in rows {
                let token = self.submit_decode_row(row, /* allow_graph = */ false)?;
                tokens.push(SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                });
            }
            return Ok(tokens);
        }

        // Legacy contiguous build (no paged pool — e.g. OPD weight offload): the
        // contiguous batched lane (env-gated) or its serial A/B arm.
        if !qwen35_batched_decode_enabled() || self.recall_active() {
            let mut tokens = Vec::with_capacity(rows.len());
            for row in rows {
                let token = self.submit_decode_row(row, /* allow_graph = */ false)?;
                tokens.push(SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                });
            }
            return Ok(tokens);
        }

        if self.batch_decode.is_none() {
            let num_full = self.model.config.num_full_attention_layers();
            let num_linear = self.model.config.num_hidden_layers - num_full;
            self.batch_decode = Some(crate::qwen35::Qwen35BatchDecodeState::new(
                &self.model.ctx,
                num_full,
                num_linear,
                self.num_slots,
            )?);
        }
        let bd = self
            .batch_decode
            .as_mut()
            .expect("batch_decode built above");

        let slot_indices: Vec<usize> = rows.iter().map(|r| r.slot).collect();
        let tokens_in: Vec<u32> = rows.iter().map(|r| r.last_token).collect();
        let kv_seq_lens: Vec<usize> = rows.iter().map(|r| r.kv_seq_len).collect();
        let params: Vec<SamplingParams> = rows.iter().map(|r| r.params.clone()).collect();
        let positions: Vec<u64> = rows
            .iter()
            .map(|r| r.kv_seq_len.saturating_add(1) as u64)
            .collect();
        let sampled = self.model.forward_decode_batch(
            &mut self.slots,
            bd,
            &slot_indices,
            &tokens_in,
            &kv_seq_lens,
            &params,
            &positions,
        )?;
        ensure!(
            sampled.len() == rows.len(),
            "Qwen3.5 batched decode returned {} tokens for {} rows",
            sampled.len(),
            rows.len()
        );
        Ok(slot_indices
            .into_iter()
            .zip(sampled)
            .map(|(slot, token)| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// A rows>1 pure-decode sub-batch over the SHARED-PAGED default lane: append
    /// each row's new token to its slot in `full_attn_kv`, build ONE B-row page
    /// table ([`PageMeta::for_decode_batch`]), and run a single batched-paged
    /// forward ([`Qwen35Model::forward_decode_batch_paged`]). This is the paged
    /// analogue of the contiguous [`Self::submit_decode_batch`] true-batch arm:
    /// the per-row append + page-table build mirrors the single-row
    /// [`Self::decode_row_paged_default`], so a B-row batch is byte-equivalent to
    /// B sequential single-row paged decodes (each row attends only its own
    /// slot's pages via its `kv_indptr` slice). BF16 single-GPU, no recall (gated
    /// by the caller in `submit_decode_batch`).
    fn submit_decode_batch_paged(&mut self, rows: &[DecodeRow]) -> Result<Vec<SlotToken>> {
        debug_assert!(rows.len() > 1);
        // Append this step's token to every row's slot BEFORE building the page
        // table (the pool must hold the POST-append length the meta encodes). The
        // pool seq_len must equal the engine's kv_seq_len pre-append, exactly as
        // `decode_row_paged_default` checks — a mismatch means an upstream prefill
        // left the slot inconsistent (e.g. leaked radix reuse).
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (full_attn_paged)");
            for row in rows {
                ensure!(
                    pool.seq_len(row.slot) == row.kv_seq_len,
                    "Qwen3.6 paged batched decode: pool seq_len {} != kv_seq_len {} for slot {}",
                    pool.seq_len(row.slot),
                    row.kv_seq_len,
                    row.slot
                );
                pool.alloc_tokens(row.slot, 1)?;
            }
        }

        let slot_indices: Vec<usize> = rows.iter().map(|r| r.slot).collect();
        let tokens_in: Vec<u32> = rows.iter().map(|r| r.last_token).collect();
        let kv_seq_lens: Vec<usize> = rows.iter().map(|r| r.kv_seq_len).collect();
        let params: Vec<SamplingParams> = rows.iter().map(|r| r.params.clone()).collect();
        let sample_positions: Vec<u64> = rows
            .iter()
            .map(|r| r.kv_seq_len.saturating_add(1) as u64)
            .collect();
        // Page table over the POST-append lengths (one row per slot).
        let batch_rows: Vec<(usize, usize)> =
            rows.iter().map(|r| (r.slot, r.kv_seq_len + 1)).collect();

        if self.batch_decode.is_none() {
            let num_full = self.model.config.num_full_attention_layers();
            let num_linear = self.model.config.num_hidden_layers - num_full;
            self.batch_decode = Some(crate::qwen35::Qwen35BatchDecodeState::new(
                &self.model.ctx,
                num_full,
                num_linear,
                self.num_slots,
            )?);
        }

        let Self {
            model,
            slots,
            batch_decode,
            full_attn_kv,
            ..
        } = self;
        let bd = batch_decode.as_mut().expect("batch_decode built above");
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let meta = PageMeta::for_decode_batch(&model.ctx, pool, &batch_rows)?;
        let sampled = model.forward_decode_batch_paged(
            slots,
            bd,
            pool,
            &meta,
            &slot_indices,
            &tokens_in,
            &kv_seq_lens,
            &params,
            &sample_positions,
        )?;
        ensure!(
            sampled.len() == rows.len(),
            "Qwen3.6 paged batched decode returned {} tokens for {} rows",
            sampled.len(),
            rows.len()
        );
        Ok(slot_indices
            .into_iter()
            .zip(sampled)
            .map(|(slot, token)| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// OPD teacher raw-logits forward: run the full hybrid forward over
    /// `(input_ids, positions)` on a FRESH transient slot state and return the
    /// FULL `[seq_len, vocab]` logits (every row) plus the model's device
    /// context, WITHOUT sampling.
    ///
    /// This does not touch the serving slots: the teacher scores a sequence
    /// in one shot (positions are contiguous from `positions[0]`), so a private
    /// slot state is allocated, advanced once, and dropped. `positions` must be
    /// the contiguous absolute positions of `input_ids` (the OPD teacher always
    /// scores a full prompt starting at `positions[0]`).
    pub(crate) fn forward_token_logits(
        &mut self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<(DeviceVec, [usize; 2])> {
        self.ensure_not_collective("forward_token_logits")?;
        ensure!(
            !input_ids.is_empty(),
            "forward_token_logits requires a non-empty token sequence"
        );
        ensure!(
            input_ids.len() == positions.len(),
            "forward_token_logits token/position length mismatch: tokens={} positions={}",
            input_ids.len(),
            positions.len()
        );
        let start_pos = positions[0] as usize;
        for (i, &p) in positions.iter().enumerate() {
            ensure!(
                p as usize == start_pos + i,
                "forward_token_logits requires contiguous positions; positions[{i}]={p} != {}",
                start_pos + i
            );
        }
        // Private transient slot: the teacher scores the whole sequence in one
        // forward, so it never shares the serving slots' KV/recurrent state.
        // The shared workspace IS reused (serial forwards; its buffers reshape
        // to the teacher's seq_len and back on the next serving call).
        // Recurrent state is acquired into a throwaway local pool (the block is
        // dropped with the slot, never pooled — this path is rare and transient).
        let mut slot = self.model.new_slot_state();
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        let mut scratch_pool = Vec::new();
        slot.acquire_recurrent(
            &self.model.ctx,
            num_linear,
            gdr_len,
            conv_len,
            &mut scratch_pool,
        )?;
        self.model
            .forward_token_logits_full(&mut slot, &mut self.workspace, input_ids, start_pos)
    }

    pub(crate) fn device(&self) -> &DeviceContext {
        &self.model.ctx
    }

    /// Fold a fresh student LoRA update into the resident projection weights
    /// (OPD per-step re-merge). Delegates to [`crate::qwen35::Qwen35Model`].
    pub(crate) fn remerge_student_lora(
        &mut self,
        update: crate::qwen35::StudentLoraUpdate,
    ) -> Result<()> {
        self.ensure_not_collective("remerge_student_lora")?;
        // The merge REPLACES `DeviceMatrix` buffers (new device addresses);
        // captured decode graphs bake the old ones — drop and recapture lazily.
        self.decode_graph = None;
        // Weight epoch changed: sidecar recurrent snapshots are as stale as the
        // radix the caller invalidates — drop every tracked sidecar blob from the
        // tier so a skipped capture never serves old-epoch state. (Blobs whose
        // tail page already LRU-evicted are unreachable — the radix is invalidated
        // too — and are reclaimed by the store's budget.)
        for (_, key) in self.sidecar_page_key.drain() {
            self.slot_tier
                .remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key);
        }
        self.model.remerge_student_lora(update)
    }

    /// Read-only borrow of resident FP8 block-scaled base projection pointers
    /// (train-infer weight sharing, `--share-frozen-base`). Delegates to
    /// [`crate::qwen35::Qwen35Model`]. Read-only; does not mutate resident
    /// weights, so no decode-graph invalidation is needed.
    pub(crate) fn frozen_base_fp8_pointers(
        &self,
    ) -> Result<Vec<crate::qwen35::SharedFp8BaseProjection>> {
        self.ensure_not_collective("frozen_base_fp8_pointers")?;
        self.model.frozen_base_fp8_pointers()
    }
}

#[cfg(test)]
mod tier_io_tests {
    use super::*;

    #[test]
    fn merge_prefers_direct_and_saturates_counters() {
        let slot = kv_native_sys::TierIoStats {
            mode: kv_native_sys::DiskIoMode::Mmap,
            useful_read_bytes: u64::MAX,
            failures: 2,
            ..Default::default()
        };
        let recall = kv_native_sys::TierIoStats {
            mode: kv_native_sys::DiskIoMode::Direct,
            useful_read_bytes: 1,
            failures: 3,
            ..Default::default()
        };
        let merged = merge_tier_io_stats(&slot, &recall);
        assert_eq!(merged.mode, infer_seam::KvTierIoMode::Direct);
        assert_eq!(merged.useful_read_bytes, u64::MAX);
        assert_eq!(merged.failures, 5);
    }
}
