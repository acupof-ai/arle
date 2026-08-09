//! Rubric-OPD orchestration (cuda-gated): the DeepSeek-V4-Flash *text* judge
//! (I2-wire) and the RFT loop. The pure judge primitives (rubric/verdict/select)
//! live in [`crate::rubric`] and unit-test on CPU.
//!
//! The judge takes the
//! rollout as **text** and the engine tokenizes it with its OWN (Flash's)
//! tokenizer, so the student's Qwen vocab never enters the judge path — the
//! cross-tokenizer sidestep that makes a DeepSeek teacher usable for a Qwen student.

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use anyhow::{Result, anyhow};
#[cfg(feature = "cuda")]
use infer_api::{CompletionRequest, InferenceEngine, LoadedInferenceEngine, SamplingParams};

#[cfg(feature = "cuda")]
use crate::infer_student::InferStudent;
#[cfg(feature = "cuda")]
use crate::rubric::{Rubric, Verdict, select, select_by_self_consistency};

/// A text judge backed by a strong teacher engine (DeepSeek-V4-Flash). Renders the
/// rubric judge prompt, generates a greedy verdict, and parses it. Vocab-agnostic.
#[cfg(feature = "cuda")]
pub struct FlashJudge {
    engine: Arc<Mutex<LoadedInferenceEngine>>,
    max_verdict_tokens: usize,
}

#[cfg(feature = "cuda")]
impl FlashJudge {
    pub fn new(engine: Arc<Mutex<LoadedInferenceEngine>>, max_verdict_tokens: usize) -> Self {
        Self {
            engine,
            max_verdict_tokens,
        }
    }

    /// Judge one rollout against the rubric. The judge decodes greedily for a
    /// deterministic verdict. A lock-poison (a bug) propagates; the caller is
    /// expected to map a transient engine error to [`Verdict::parse_error`] so one
    /// bad judge call never aborts a round (CLAUDE.md §0 case-as-fact).
    pub fn judge(&self, rubric: &Rubric, problem: &str, rollout: &str) -> Result<Verdict> {
        let prompt = rubric.judge_prompt(problem, rollout);
        let req = CompletionRequest {
            prompt,
            max_tokens: self.max_verdict_tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: None,
            logprobs: false,
            session_id: None,
            trace_context: None,
            cancel: None,
        };
        let output = {
            let mut engine = self
                .engine
                .lock()
                .map_err(|err| anyhow!("Flash judge engine lock poisoned: {err}"))?;
            engine.complete(req)?
        };
        let verdict = rubric.parse_verdict(&output.text);
        if std::env::var("ARLE_RUBRIC_DEBUG").is_ok() {
            let snip: String = output.text.chars().take(900).collect();
            eprintln!(
                "[rubric judge] finish={:?} parse_err={} accepted={} raw_verdict={snip:?}",
                output.finish_reason, verdict.parse_error, verdict.accepted
            );
        }
        Ok(verdict)
    }

    /// Judge a rollout, mapping any transient engine error to a parse-error
    /// verdict (logged, surfaced via `Selection::parse_errors`, never accepted).
    pub fn judge_resilient(&self, rubric: &Rubric, problem: &str, rollout: &str) -> Verdict {
        match self.judge(rubric, problem, rollout) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("rubric_opd: judge call failed (counted as parse error): {err}");
                Verdict::parse_error()
            }
        }
    }

    /// Batched [`judge`]: judges N rollouts of the SAME problem in one engine
    /// batch (the judge engine decodes them concurrently). Returns one Verdict per
    /// rollout, in order; a per-request engine/parse failure maps to
    /// `Verdict::parse_error()` (never aborts the round). Vocab-agnostic.
    pub fn judge_batch(&self, rubric: &Rubric, problem: &str, rollouts: &[String]) -> Vec<Verdict> {
        if rollouts.is_empty() {
            return Vec::new();
        }
        let reqs: Vec<CompletionRequest> = rollouts
            .iter()
            .map(|r| CompletionRequest {
                prompt: rubric.judge_prompt(problem, r),
                max_tokens: self.max_verdict_tokens,
                sampling: SamplingParams {
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                stop: None,
                logprobs: false,
                session_id: None,
                trace_context: None,
                cancel: None,
            })
            .collect();
        let outputs = {
            let engine = match self.engine.lock() {
                Ok(e) => e,
                Err(err) => {
                    eprintln!("rubric_opd: judge_batch lock poisoned: {err}");
                    return vec![Verdict::parse_error(); rollouts.len()];
                }
            };
            engine.complete_batch(reqs)
        };
        match outputs {
            Ok(outs) => outs.iter().map(|o| rubric.parse_verdict(&o.text)).collect(),
            Err(err) => {
                eprintln!("rubric_opd: judge_batch failed (all parse_error): {err}");
                vec![Verdict::parse_error(); rollouts.len()]
            }
        }
    }

    /// Mode B: generate a correct solution for a (rejected) prompt and return it
    /// only if the teacher's own solution passes the rubric self-check (quality
    /// gate). `None` = the teacher could not produce a passing solution; never
    /// trains on an unvalidated target. Caller re-tokenizes the returned text.
    pub fn correct(
        &self,
        rubric: &Rubric,
        problem: &str,
        max_tokens: usize,
    ) -> Result<Option<String>> {
        let req = CompletionRequest {
            prompt: rubric.solve_prompt(problem),
            max_tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: None,
            logprobs: false,
            session_id: None,
            trace_context: None,
            cancel: None,
        };
        let output = {
            let mut engine = self
                .engine
                .lock()
                .map_err(|err| anyhow!("Flash judge engine lock poisoned: {err}"))?;
            engine.complete(req)?
        };
        let solution = output.text;
        // Quality gate: the teacher must pass its own solution (objective for math).
        let verdict = self.judge_resilient(rubric, problem, &solution);
        Ok(verdict.accepted.then_some(solution))
    }

    /// Offload the judge engine's device weights to host RAM, freeing VRAM for the
    /// student CE backward (rubric-OPD time-share). Returns bytes freed.
    pub fn offload_engine_weights(&self) -> Result<usize> {
        self.engine
            .lock()
            .map_err(|err| anyhow!("Flash judge engine lock poisoned: {err}"))?
            .offload_engine_weights()
    }

    pub fn reload_engine_weights(&self) -> Result<()> {
        self.engine
            .lock()
            .map_err(|err| anyhow!("Flash judge engine lock poisoned: {err}"))?
            .reload_engine_weights()
    }
}

/// Rubric-OPD round/sampling configuration.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct RubricOpdConfig {
    /// RFT rounds (generate → judge → select → writeback-CE).
    pub rounds: usize,
    /// On-policy samples generated per prompt each round (rejection sampling).
    pub samples_per_prompt: usize,
    /// Max new tokens per sampled rollout.
    pub max_new_tokens: usize,
    /// Cap on CE writeback steps per round (`None` = train on all accepted). The
    /// 27B-dense autograd CE is ~minutes/step (host-authoritative), so bounding the
    /// accepted set keeps a round tractable; capped to the first N accepted.
    pub writeback_cap: Option<usize>,
    /// Micro-batch size for the CE writeback. The CE is overhead-bound (GPU ~0%
    /// util), so batching B accepted pairs into one forward+backward amortizes the
    /// host op-dispatch (~B× throughput); B bounds the [B, seq, vocab] logit VRAM.
    pub writeback_batch: usize,
    /// Mode B: max Flash corrections to add per round (0 = Mode A / select-only).
    pub correction_cap: usize,
    /// Max new tokens for a Mode B correction solution.
    pub correction_max_tokens: usize,
    /// Train-infer FP8 weight sharing (`--share-frozen-base`). When `true`, the
    /// autograd student's frozen FP8 base ALIASES the rollout engine's resident
    /// base bytes, so the Phase-B rollout-engine weight offload is SKIPPED
    /// (freeing those bytes would crash the CE forward reading the alias). The
    /// VRAM the offload used to free is already saved by sharing the single copy.
    /// The judge offload still runs. Default `false` = offload as today.
    pub share_frozen_base: bool,
    /// SOPD 对且短 ("correct AND short"): when `true`, among the accepted
    /// (self-consistency-correct) rollouts for a prompt, distill ONLY the SHORTEST
    /// (fewest tokens) instead of all of them — biases CE toward the most
    /// token-efficient correct reasoning, teaching the student to think concisely.
    /// Default `false` = write back every accepted rollout (correctness only).
    pub distill_shortest: bool,
}

/// Per-round accounting. `distinct_accepted` is the RFT log-linear x-axis;
/// `parse_errors` surfaces judge timeouts/garbage (never bucketed as fail).
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Default)]
pub struct RoundReport {
    pub round: usize,
    pub prompts: usize,
    pub accepted: usize,
    pub distinct_accepted: usize,
    pub parse_errors: usize,
    pub trained: usize,
    pub corrected: usize,
    pub mean_train_loss: f32,
}

/// The rubric-OPD RFT driver: for each round, sample N rollouts per prompt, judge
/// each with Flash (or, with `judge = None`, self-consistency majority-vote on the
/// `\boxed` answer), select the accepted, and write them back via `train_on_accepted`.
///
/// `judge = None` is SELF-CONSISTENCY mode: one model judges itself by
/// majority-vote — no separate judge engine is loaded (frees its VRAM) and Mode B
/// corrections are disabled (a corrector requires the judge).
///
/// `decode` turns generated student token ids into the text the judge reads;
/// `train_on_accepted(prompt_ids, completion_ids)` runs one student CE step on an
/// accepted rollout and returns its loss. Both are supplied by the CLI handler
/// (tokenizer + autograd CE step), keeping this driver free of the heavy wiring so
/// the GPU operations it composes (`generate_samples`, `FlashJudge`, `select`) stay
/// independently testable.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn run_rubric_rounds<D, T, E>(
    student: &InferStudent,
    judge: Option<&FlashJudge>,
    rubric: &Rubric,
    prompts: &[(String, Vec<u32>)],
    cfg: &RubricOpdConfig,
    sampling: Option<&SamplingParams>,
    mut decode: D,
    mut train_on_accepted: T,
    mut encode: E,
) -> Result<Vec<RoundReport>>
where
    D: FnMut(&[u32]) -> Result<String>,
    T: FnMut(&[(Vec<u32>, Vec<u32>)]) -> Result<f32>,
    E: FnMut(&str) -> Result<Vec<u32>>,
{
    let mut reports = Vec::with_capacity(cfg.rounds);
    for round in 0..cfg.rounds {
        let mut rep = RoundReport {
            round,
            ..Default::default()
        };
        let mut loss_sum = 0.0f32;
        let debug = std::env::var("ARLE_RUBRIC_DEBUG").is_ok();

        // Phase A — sample + judge + select (BOTH inference engines resident).
        // Accumulate accepted (prompt_ids, completion_ids) pairs; defer all CE so
        // the engines can be offloaded as one block before the autograd backward.
        let mut accepted_pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        let mut rejected: Vec<(&str, Vec<u32>)> = Vec::new();
        for (problem, prompt_ids) in prompts {
            rep.prompts += 1;
            eprintln!(
                "[rubric] round {round} phase-A: sampling prompt {}/{} ({} samples)",
                rep.prompts,
                prompts.len(),
                cfg.samples_per_prompt
            );
            let samples = student.generate_samples(
                prompt_ids,
                cfg.max_new_tokens,
                cfg.samples_per_prompt,
                sampling,
            )?;
            let texts = samples
                .iter()
                .map(|s| decode(s))
                .collect::<Result<Vec<String>>>()?;
            if debug {
                for (i, t) in texts.iter().enumerate() {
                    let snip: String = t.chars().take(900).collect();
                    eprintln!("[rubric rollout] sample{i} len={} text={snip:?}", t.len());
                }
            }
            let sel = match judge {
                Some(j) => {
                    let verdicts: Vec<Verdict> = j.judge_batch(rubric, problem, &texts);
                    select(&texts, &verdicts)
                }
                None => select_by_self_consistency(&texts),
            };
            rep.accepted += sel.accepted.len();
            rep.distinct_accepted += sel.distinct_accepted;
            rep.parse_errors += sel.parse_errors;
            if cfg.distill_shortest {
                // 对且短: the shortest correct rollout is the most token-efficient path
                // to the consensus answer; distilling only it teaches the student to
                // reach the same answer with less reasoning (thinking efficiency).
                if let Some(&idx) = sel.accepted.iter().min_by_key(|&&i| samples[i].len()) {
                    accepted_pairs.push((prompt_ids.clone(), samples[idx].clone()));
                }
            } else {
                for &idx in &sel.accepted {
                    accepted_pairs.push((prompt_ids.clone(), samples[idx].clone()));
                }
            }
            if sel.accepted.is_empty() {
                rejected.push((problem.as_str(), prompt_ids.clone()));
            }
        }

        // Mode B — Flash correction for rejected prompts (breaks the best-of-N
        // ceiling). Requires a judge; self-consistency (judge=None) does no
        // corrections.
        if let Some(j) = judge
            && cfg.correction_cap > 0
        {
            let mut made = 0usize;
            for (problem, prompt_ids) in &rejected {
                if made >= cfg.correction_cap {
                    break;
                }
                match j.correct(rubric, problem, cfg.correction_max_tokens) {
                    Ok(Some(solution)) => match encode(&solution) {
                        Ok(tokens) if !tokens.is_empty() => {
                            accepted_pairs.push((prompt_ids.clone(), tokens));
                            rep.corrected += 1;
                            made += 1;
                        }
                        Ok(_) => {}
                        Err(err) => eprintln!("rubric_opd: correction encode failed: {err}"),
                    },
                    Ok(None) => {}
                    Err(err) => eprintln!("rubric_opd: correction failed (skipped): {err}"),
                }
            }
            eprintln!(
                "[rubric] round {round} Mode-B: {made} corrections added (of {} rejected prompts)",
                rejected.len()
            );
        }

        // Cap the CE writeback set (the 27B-dense host-authoritative CE is
        // ~minutes/step; bound it to keep a round tractable). Keep the first N.
        if let Some(cap) = cfg.writeback_cap
            && accepted_pairs.len() > cap
        {
            eprintln!(
                "[rubric] round {round} writeback-cap: {} accepted -> training first {cap}",
                accepted_pairs.len()
            );
            accepted_pairs.truncate(cap);
        }

        // Phase B — offload the inference engines to host RAM, freeing their VRAM
        // (~tens of GB) for the 27B autograd CE forward+backward. Without this the
        // CE step OOMs (`cuda alloc_zeros failed`) at seq ~1k.
        //
        // `--share-frozen-base`: the autograd student's frozen FP8 base ALIASES the
        // rollout engine's resident base bytes, so offloading (freeing) the rollout
        // weights would crash the CE forward reading the alias. SKIP the rollout
        // offload when sharing — the single shared copy already saves that VRAM. The
        // judge engine is independent and still offloads.
        eprintln!(
            "[rubric] round {round} phase-B: offloading engines ({} accepted to train){}",
            accepted_pairs.len(),
            if cfg.share_frozen_base {
                " [share-frozen-base: rollout base kept resident]"
            } else {
                ""
            }
        );
        let freed_rollout = if cfg.share_frozen_base {
            // Weights stay resident (student aliases them), but the inference
            // scratch (workspace + batched-decode buffers + decode graph) and
            // the (dead) KV pool are pure headroom — the writeback's fresh
            // autograd forward never reads them. Release both so the 27B
            // forward+backward fits.
            if let Err(err) = student.release_inference_scratch() {
                eprintln!("[rubric] release inference scratch failed: {err}");
            }
            if let Err(err) = student.release_kv_pool() {
                eprintln!("[rubric] release KV pool failed: {err}");
            }
            0
        } else {
            student.offload_engine_weights().unwrap_or(0)
        };
        let freed_judge = match judge {
            Some(j) => j.offload_engine_weights().unwrap_or(0),
            None => 0,
        };
        eprintln!(
            "[rubric] round {round} phase-B done: freed rollout={freed_rollout} judge={freed_judge} bytes"
        );

        // Phase C — CE writeback on accepted pairs, micro-batched (engines offloaded).
        // Batching amortizes the overhead-bound host op-dispatch over the micro-batch.
        let batch_size = cfg.writeback_batch.max(1);
        let total_chunks = accepted_pairs.len().div_ceil(batch_size);
        for (ci, chunk) in accepted_pairs.chunks(batch_size).enumerate() {
            eprintln!(
                "[rubric] round {round} phase-C: CE micro-batch {}/{} (size {})",
                ci + 1,
                total_chunks,
                chunk.len()
            );
            let chunk_loss = train_on_accepted(chunk)?;
            loss_sum += chunk_loss * chunk.len() as f32;
            rep.trained += chunk.len();
        }

        // Phase D — reload engines. Always: the caller drives rounds one-at-a-time
        // and relies on resident engines after the call (the between-round LoRA sync
        // pushes into the rollout engine; the next call samples/judges from them).
        // `--share-frozen-base`: the rollout engine was never offloaded (its base
        // stays resident, aliased by the student), so skip its reload — symmetric
        // with the Phase-B skip. The judge still reloads.
        eprintln!("[rubric] round {round} phase-D: reloading engines");
        if !cfg.share_frozen_base {
            student.reload_engine_weights()?;
        } else {
            // Weights stayed resident; only the KV pool (released in Phase B)
            // needs re-acquiring before the next round's rollout.
            if let Err(err) = student.ensure_kv_pool() {
                eprintln!("[rubric] ensure KV pool failed: {err}");
            }
        }
        if let Some(j) = judge {
            j.reload_engine_weights()?;
        }

        rep.mean_train_loss = if rep.trained > 0 {
            loss_sum / rep.trained as f32
        } else {
            0.0
        };
        eprintln!(
            "rubric_opd round {round}: prompts={} accepted={} distinct={} parse_err={} trained={} corrected={} mean_loss={:.4}",
            rep.prompts,
            rep.accepted,
            rep.distinct_accepted,
            rep.parse_errors,
            rep.trained,
            rep.corrected,
            rep.mean_train_loss
        );
        reports.push(rep);
    }
    Ok(reports)
}
