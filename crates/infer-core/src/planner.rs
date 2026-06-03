//! Scheduling / planning hot path for [`Engine`].
//!
//! These `impl Engine` methods decide which rows run this tick: chunked-prefill
//! row construction, decode-priority ordering, and the retract/preempt repair
//! that keeps a plan within the KV page budget. They are split out of the
//! coordinator (`lib.rs`) because this is the highest-churn axis. Identical
//! numerics — pure reorganization.

use std::cmp::Reverse;

use infer_plan::{DecodeRow, ForwardMode, ForwardPlan, PrefillRow};
use infer_seam::{BackendExecutor, KvPool};

use crate::{Engine, RequestPhase, WaitingInsertBias};

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
                decode_rows.push(DecodeRow {
                    slot,
                    last_token,
                    kv_seq_len: self.kv.seq_len(slot),
                    params: request.sampling.clone(),
                });
            }
        }

        // Chunked prefill under the per-tick token budget and concurrency cap.
        // A prompt longer than `prefill_chunk_size` is split across ticks so
        // decode rows keep interleaving (interactivity + mixed batching).
        let mut budget = self.config.prefill_step_budget();
        let chunk_cap = self.config.prefill_chunk_size();
        let max_prefills = self.config.max_concurrent_prefill();
        for (&slot, request) in &self.active {
            if prefill_rows.len() >= max_prefills || budget == 0 {
                break;
            }
            if !matches!(request.phase, RequestPhase::Prefilling { .. }) {
                continue;
            }
            let start_pos = request.prefill_start_pos.min(request.prompt_tokens.len());
            let remaining = request.prompt_tokens.len() - start_pos;
            let chunk = remaining.min(chunk_cap).min(budget);
            if chunk == 0 {
                continue;
            }
            prefill_rows.push(PrefillRow {
                slot,
                tokens: request.prompt_tokens[start_pos..start_pos + chunk].to_vec(),
                start_pos,
                total_tokens: request.prompt_tokens.len(),
                params: request.sampling.clone(),
            });
            budget -= chunk;
        }

        ForwardPlan {
            mode: plan_mode(prefill_rows.is_empty(), decode_rows.is_empty()),
            decode_rows,
            prefill_rows,
            microbatch: None,
            spec: None,
        }
    }

    pub(crate) fn retract_decode_to_fit(&mut self, plan: &mut ForwardPlan) {
        while self.kv.is_active()
            && self.plan_new_pages_needed(plan) > self.kv.free_pages()
            && plan.decode_rows.len() > 1
        {
            let Some(victim_pos) = self.retract_victim_pos(&plan.decode_rows) else {
                break;
            };
            let victim_slot = plan.decode_rows[victim_pos].slot;
            self.requeue_preempted_decode(victim_slot);
            plan.decode_rows.remove(victim_pos);
            plan.mode = plan_mode(plan.prefill_rows.is_empty(), plan.decode_rows.is_empty());
        }
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

    fn requeue_preempted_decode(&mut self, slot: usize) {
        let Some(request) = self.active.remove(&slot) else {
            return;
        };
        self.release_reused_prefix(&request.reused_prefix_pages);
        self.kv.free_slot(slot);
        self.enqueue_waiting_request(
            request.reset_for_recompute(),
            WaitingInsertBias::BeforeEqual,
        );
    }

    pub(crate) fn plan_new_pages_needed(&self, plan: &ForwardPlan) -> usize {
        let prefill_pages = plan
            .prefill_rows
            .iter()
            .map(|row| self.kv.append_pages_needed(row.slot, row.tokens.len()))
            .sum::<usize>();
        let decode_pages = plan
            .decode_rows
            .iter()
            .map(|row| self.kv.append_pages_needed(row.slot, 1))
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
