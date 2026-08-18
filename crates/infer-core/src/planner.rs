//! Scheduling / planning hot path for [`Engine`].
//!
//! `impl Engine` methods deciding which rows run this tick: chunked-prefill row
//! construction, decode-priority ordering, and the retract/preempt repair that
//! keeps a plan within the KV page budget.

use std::cmp::Reverse;

use anyhow::Result;
use infer_plan::{DecodeRow, ForwardMode, ForwardPlan, PrefillRow};
use infer_seam::{BackendExecutor, KvPool};

use crate::{Engine, RequestPhase, RequestState, WaitingInsertBias};

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

impl<E: BackendExecutor, K: KvPool> Engine<E, K> {
    pub(crate) fn build_forward_plan(&self) -> ForwardPlan {
        let mut prefill_rows = Vec::new();
        let mut decode_rows = Vec::new();

        // Decode rows first (decode-priority). `active` is a BTreeMap, so this
        // iterates in deterministic slot order.
        for (&slot, request) in &self.active {
            if matches!(request.phase, RequestPhase::Decoding) {
                let Some(last_token) = request
                    .generated_tokens
                    .last()
                    .copied()
                    .or_else(|| request.prompt_tokens.last().copied())
                else {
                    continue;
                };
                let (penalty_history, penalty_prompt_len) = request.penalty_history();
                decode_rows.push(DecodeRow {
                    slot,
                    last_token,
                    kv_seq_len: self.kv.seq_len(slot),
                    params: request.sampling.clone(),
                    penalty_history,
                    penalty_prompt_len,
                });
            }
        }

        // Chunked prefill under the per-tick token budget and concurrency cap.
        // A prompt longer than `prefill_chunk_size` is split across ticks so
        // decode rows keep interleaving (interactivity + mixed batching).
        // Cap total plan tokens (decode rows + prefill chunk tokens) to the
        // executor's per-forward limit (deepep_ll LL dispatch buffer). With the
        // default `usize::MAX` both `saturating_sub` and `min` are no-ops.
        let cap = self.max_tokens_per_step;
        let limits = self.executor.step_limits();
        let mut budget = self
            .config
            .prefill_step_budget()
            .min(cap.saturating_sub(decode_rows.len()));
        let chunk_cap = self
            .config
            .prefill_chunk_size()
            .min(cap)
            .min(limits.max_prefill_chunk);
        let max_prefills = self.config.max_concurrent_prefill();
        // D.3: under 2D the ring pass attends only to the row's own KV, so the
        // prompt must land in one row — chunking would break causal attention
        // across chunks. One full-prompt row per tick; decode-interleaving is
        // sacrificed for 2D requests (Option A decision).
        let two_d = self.executor.kv_shard_spec().is_some();
        for (&slot, request) in &self.active {
            if prefill_rows.len() >= max_prefills || (!two_d && budget == 0) {
                break;
            }
            if !matches!(request.phase, RequestPhase::Prefilling { .. }) {
                continue;
            }
            let target = request.committed_len();
            let start_pos = request.prefill_start_pos.min(target);
            let remaining = target - start_pos;
            let mut chunk = if two_d {
                remaining
            } else {
                remaining.min(chunk_cap).min(budget)
            };
            if chunk == 0 {
                continue;
            }
            if !two_d {
                // Align chunk ends to lcm(page_size, restore_alignment) (LCM, not
                // max — neither is guaranteed to divide the other): some backends
                // restore side state (ring/compressor snapshots) only at their own
                // coarser boundary. Chunk SIZE is bounded by max_prefill_chunk()
                // in chunk_cap above; this only aligns where the chunk ends.
                let page_size = self.kv.page_size().max(1);
                let restore_alignment = limits.prefill_restore_boundary_alignment.max(1);
                let alignment_unit = lcm(page_size, restore_alignment);
                let chunk_end = start_pos + chunk;
                let aligned_end = chunk_end - (chunk_end % alignment_unit);
                if aligned_end > start_pos {
                    chunk = aligned_end - start_pos;
                }
            }
            let (penalty_history, penalty_prompt_len) = request.penalty_history();
            prefill_rows.push(PrefillRow {
                slot,
                tokens: request.committed_slice(start_pos, chunk),
                start_pos,
                total_tokens: target,
                params: request.sampling.clone(),
                penalty_history,
                penalty_prompt_len,
            });
            if !two_d {
                budget -= chunk;
            }
        }

        ForwardPlan {
            mode: plan_mode(prefill_rows.is_empty(), decode_rows.is_empty()),
            decode_rows,
            prefill_rows,
        }
    }

    /// Repair a plan whose page demand exceeds the pool's reclaimable
    /// capacity so `allocate_for_plan` cannot fail — a step-loop alloc error
    /// is FATAL (it unwinds the whole TP worker group, #164). Capacity counts
    /// free PLUS cache-evictable pages: `alloc_with_prefix_reclaim` evicts the
    /// radix LRU on demand, so radix-retained pages are evictable-but-not-free
    /// and only demand beyond both would genuinely fail. Budgeting against
    /// `free_pages` alone livelocked a warm cache (free=0 shed every prefill
    /// chunk each tick forever) and cascaded preemptions whose freed pages
    /// merely re-entered the cache. Cheapest loss first: shed demand-reducing
    /// prefill chunks (a dropped chunk retries next tick with no state
    /// change), then preempt-requeue decode victims down to an empty plan —
    /// each retraction makes the victim's pages free or evictable (park or
    /// recompute, the #162 path), so the loop terminates and later ticks fit.
    pub(crate) fn fit_plan_to_kv_pages(&mut self, plan: &mut ForwardPlan) -> Result<()> {
        if !self.kv.is_active() {
            return Ok(());
        }
        // Rank-synced capacity: per-rank KV-tier residuals make the raw
        // locals diverge, and a rank-local read here would diverge the
        // shed/preempt decisions → divergent ForwardPlans → collective shape
        // mismatch → TP hang. Induction: identical synced starting capacity +
        // identical plan → identical decisions → pool mutations stay
        // lockstep. Unconditional at this fixed call point (every rank with a
        // non-idle plan reaches it): one small host collective per tick, the
        // same price admission already pays; single-rank backends return
        // `local` unchanged.
        let mut capacity = self
            .executor
            .tp_sync_min(self.kv.free_pages() + self.kv.resident_evictable_pages())?;
        if self.plan_new_pages_needed(plan) <= capacity {
            return Ok(());
        }
        // Shed only rows that reduce demand — a zero-demand row (fully
        // prefix-reused) frees nothing and would be deferred for nothing.
        while self.plan_new_pages_needed(plan) > capacity {
            let Some(pos) = plan
                .prefill_rows
                .iter()
                .rposition(|row| self.kv.append_pages_needed(row.slot, row.tokens.len()) > 0)
            else {
                break;
            };
            plan.prefill_rows.remove(pos);
        }
        while self.plan_new_pages_needed(plan) > capacity {
            let Some(victim_pos) = self.retract_victim_pos(&plan.decode_rows) else {
                break;
            };
            let victim_slot = plan.decode_rows[victim_pos].slot;
            self.requeue_preempted_decode(victim_slot);
            plan.decode_rows.remove(victim_pos);
            // Re-sync: the victim's pages went free or cache-evictable, but
            // tier demote acceptance is rank-local. Loop iterations are
            // themselves lockstep (condition inputs identical on all ranks),
            // so every rank issues the same number of collectives.
            capacity = self
                .executor
                .tp_sync_min(self.kv.free_pages() + self.kv.resident_evictable_pages())?;
        }
        plan.mode = plan_mode(plan.prefill_rows.is_empty(), plan.decode_rows.is_empty());
        Ok(())
    }

    fn retract_victim_pos(&self, decode_rows: &[DecodeRow]) -> Option<usize> {
        decode_rows
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| {
                self.active
                    .get(&row.slot)
                    .map_or((usize::MAX, Reverse(0)), |request| {
                        (
                            request.generated_tokens.len(),
                            Reverse(request.prompt_tokens.len()),
                        )
                    })
            })
            .map(|(pos, _)| pos)
    }

    /// Oversubscription park — PARK-OR-NOTHING. Demote the victim's whole-slot
    /// image FIRST; only a successful park preempts (parked `AfterEqual` so the
    /// freed slot goes to the existing waiter). A refused/failed demote leaves
    /// the victim running untouched and returns false. The old path reset a
    /// failed park to recompute: with a persistently refusing store that
    /// ping-ponged the running pair at the 8-token min slice forever (pod
    /// round-5 livelock: ~2,060 park→refuse→recompute cycles at 3.6/s, zero
    /// completions). KV-overflow retract keeps its recompute fallback — there
    /// the pages MUST free; here keeping the victim running is strictly better.
    pub(crate) fn try_park_for_oversubscription(&mut self, slot: usize) -> bool {
        let Some(request) = self.active.get(&slot) else {
            return false;
        };
        if !matches!(request.phase, RequestPhase::Decoding) {
            return false;
        }
        let demoted_seq_len = self.kv.seq_len(slot);
        if demoted_seq_len == 0 {
            return false;
        }
        let key = self.next_tier_key;
        self.next_tier_key = self.next_tier_key.wrapping_add(1);
        // The park stalls the whole engine (whole-slot D2H + sync), so its cost
        // is a scheduling input, not a detail: surface it per event.
        let started = std::time::Instant::now();
        let demoted = match self.executor.kv_slot_tier() {
            Some(tier) => tier.demote_slot(slot, key),
            None => Ok(false),
        };
        match demoted {
            Ok(true) => {}
            Ok(false) => return false,
            Err(err) => {
                log::warn!("whole-slot KV demote failed for slot {slot}: {err:#}");
                return false;
            }
        }
        self.kv_tier_stats.demoted_slots = self.kv_tier_stats.demoted_slots.saturating_add(1);
        log::info!(
            "oversubscription park: slot {slot} seq_len {demoted_seq_len} in {:.1} ms (park #{})",
            started.elapsed().as_secs_f64() * 1e3,
            self.kv_tier_stats.demoted_slots
        );
        let mut request = self.active.remove(&slot).expect("checked above");
        // free_slot before release_reused_prefix — same ordering as finish_slot.
        self.free_slot_pages(slot);
        self.release_reused_prefix(&request.reused_prefix_pages);
        // Keep the generation: decode resumes at the demoted position after
        // promote (see requeue_preempted_decode for the length note).
        request.swap_key = Some(key);
        request.swap_seq_len = demoted_seq_len;
        request.reused_prefix_pages.clear();
        request.prefill_start_pos = 0;
        request.phase = RequestPhase::Prefilling { progress: 0 };
        self.enqueue_waiting_request(request, WaitingInsertBias::AfterEqual);
        true
    }

    /// Requeue a preempted decode back to the waiting queue. A retracted decode
    /// has the most progress, so it re-admits ASAP — `BeforeEqual` keeps it
    /// ahead of equal-priority peers.
    pub(crate) fn requeue_preempted_decode(&mut self, slot: usize) {
        let Some(mut request) = self.active.remove(&slot) else {
            return;
        };
        // Swap-style preemption: page-tier backends seal the victim's prompt
        // blocks into the radix and demote those pages, so re-admission can
        // promote the prefill instead of recomputing it. Whole-slot route
        // backends demote the complete slot restore image and resume decode
        // with `generated_tokens` intact. Without a tier store the behavior is
        // the plain recompute path.
        let mut slot_swap_key = None;
        let demoted_seq_len = self.kv.seq_len(slot);
        // The full COMMITTED sequence — the same boundary finish_slot
        // publishes. Prompt-only publish dropped every generated page's
        // provisional backend entry at the free below, so resume / follow-up
        // turns recomputed the whole generated region instead of attaching
        // through it.
        let committed_tokens = request.committed_tokens();
        if self.kv_tier_capacity() > 0 {
            // Publish ensures radix + sidecar are captured (idempotent for
            // already-cached blocks — returns empty in that case).
            let _ = self.publish_prefix_blocks(slot, &committed_tokens);
        } else if matches!(request.phase, RequestPhase::Decoding)
            && demoted_seq_len > 0
            && let Some(tier) = self.executor.kv_slot_tier()
        {
            let key = self.next_tier_key;
            self.next_tier_key = self.next_tier_key.wrapping_add(1);
            match tier.demote_slot(slot, key) {
                Ok(true) => {
                    slot_swap_key = Some(key);
                    self.kv_tier_stats.demoted_slots =
                        self.kv_tier_stats.demoted_slots.saturating_add(1);
                }
                Ok(false) => {}
                Err(err) => {
                    log::warn!("whole-slot KV demote failed for slot {slot}: {err:#}");
                }
            }
        } else {
            // Plain-recompute arms (e.g. DSv4 #154 2b — pool entries are the
            // demotion, device bands free via release_kv_slot): seal the
            // committed sequence BEFORE free_slot_pages drops it. Self-serving
            // (#156): the requeued request keeps its generation and re-attaches
            // through these blocks on resume.
            let _ = self.publish_prefix_blocks(slot, &committed_tokens);
        }
        // free_slot before release_reused_prefix — same ordering fix as finish_slot.
        self.free_slot_pages(slot);
        self.release_reused_prefix(&request.reused_prefix_pages);
        // Demote the cached committed chain to tier (includes already-cached
        // blocks from the normal step() publish, not just newly-published above).
        if self.kv_tier_capacity() > 0 {
            let matched = self.radix.peek_longest_prefix_match(&committed_tokens);
            let local = matched.local_block_ids(self.radix.cp_shard());
            if !local.is_empty() {
                self.demote_published_pages(&local);
            }
        }
        let request = if let Some(key) = slot_swap_key {
            // Keep the generation: decode resumes at the demoted position
            // after promote. Only slot-coupled bookkeeping resets. The
            // materialized restore length is captured verbatim (it is one short
            // of prompt+generated: the newest token's KV is the next step's
            // write, not yet materialized) so host accounting is rebuilt to
            // exactly what the device image holds.
            request.swap_key = Some(key);
            request.swap_seq_len = demoted_seq_len;
            request.reused_prefix_pages.clear();
            request.prefill_start_pos = 0;
            request.phase = RequestPhase::Prefilling { progress: 0 };
            request
        } else {
            // Preempt-requeue without a slot image = recompute fallback:
            // count it (the gate's preempt-fired evidence) and log it.
            self.kv_system_metrics.fallback_recompute =
                self.kv_system_metrics.fallback_recompute.saturating_add(1);
            log::info!(
                "KV-overflow preempt: requeued request {} for recompute (slot {slot}, \
                 seq_len {demoted_seq_len})",
                request.handle.id()
            );
            request.reset_for_recompute()
        };
        self.enqueue_waiting_request(request, WaitingInsertBias::BeforeEqual);
    }

    /// Inverse of the whole-slot demote: rebuild host KV accounting for the
    /// full sequence, restore the device image, and resume decode. On any
    /// failure the request falls back to plain recompute on the same slot.
    pub(crate) fn restore_swapped_slot(
        &mut self,
        slot: usize,
        request: &mut RequestState,
        key: u64,
    ) -> Result<()> {
        // Re-allocate exactly the materialized length captured at demote —
        // NOT prompt+generated, which runs one ahead of the device image
        // (the newest token's KV is the next decode step's write).
        let seq_len = std::mem::take(&mut request.swap_seq_len);
        let started = std::time::Instant::now();
        let restored = self
            .alloc_with_prefix_reclaim(slot, seq_len)
            .and_then(|()| {
                let slot_pages = self.kv.page_indices(slot).to_vec();
                match self.executor.kv_slot_tier() {
                    Some(tier) => tier.promote_slot(key, slot, &slot_pages),
                    None => anyhow::bail!("backend has no whole-slot KV tier store"),
                }
            });
        match restored {
            Ok(()) => {
                if let Some(tier) = self.executor.kv_slot_tier() {
                    tier.drop_kv_slot_entries(&[key]);
                }
                request.phase = RequestPhase::Decoding;
                request.prefill_start_pos = request.prompt_len();
                self.kv_tier_stats.promoted_slots =
                    self.kv_tier_stats.promoted_slots.saturating_add(1);
                log::info!(
                    "oversubscription promote: slot {slot} seq_len {seq_len} in {:.1} ms \
                     (promote #{})",
                    started.elapsed().as_secs_f64() * 1e3,
                    self.kv_tier_stats.promoted_slots
                );
                Ok(())
            }
            Err(err) => {
                log::warn!(
                    "whole-slot KV promote failed for request {}: {err:#}; recomputing",
                    request.handle.id()
                );
                if let Some(tier) = self.executor.kv_slot_tier() {
                    tier.drop_kv_slot_entries(&[key]);
                }
                self.free_slot_pages(slot);
                self.kv_tier_stats.slot_promote_failures =
                    self.kv_tier_stats.slot_promote_failures.saturating_add(1);
                self.kv_system_metrics.fallback_recompute =
                    self.kv_system_metrics.fallback_recompute.saturating_add(1);
                let fresh = request.clone().reset_for_recompute();
                *request = fresh;
                // CP sharding: ring pass recomputes the whole prompt; skip
                // match+attach. reset_for_recompute already set the
                // prefill_start_pos=0 / Prefilling{0} state the empty-attach
                // would set, and the collectives deadlock cross-communicator.
                if self.executor.kv_shard_spec().is_none() {
                    let committed = request.committed_tokens();
                    let prefix_match = if self.config.enable_prefix_cache {
                        self.lookup_prefix_for_attach(&committed)?
                    } else {
                        crate::PrefixMatch::empty()
                    };
                    self.attach_prefix_to_request(slot, request, &committed, prefix_match)?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn plan_new_pages_needed(&self, plan: &ForwardPlan) -> usize {
        let prefill_pages = plan
            .prefill_rows
            .iter()
            .map(|row| self.kv.append_pages_needed(row.slot, row.tokens.len()))
            .sum::<usize>();
        let spec_row_tokens = self.executor.step_limits().spec_row_tokens;
        let decode_pages = plan
            .decode_rows
            .iter()
            .map(|row| self.kv.append_pages_needed(row.slot, spec_row_tokens))
            .sum::<usize>();
        prefill_pages + decode_pages
    }
}

pub(crate) fn plan_mode(prefill_empty: bool, decode_empty: bool) -> ForwardMode {
    match (prefill_empty, decode_empty) {
        (true, true) => ForwardMode::Idle,
        (false, true) => ForwardMode::Prefill,
        (true, false) => ForwardMode::Decode,
        (false, false) => ForwardMode::Mixed,
    }
}
