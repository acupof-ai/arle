//! Rubric-OPD orchestration (cuda-gated): the DeepSeek-V4-Flash *text* judge
//! (I2-wire) and the RFT loop. The pure judge primitives (rubric/verdict/select)
//! live in [`crate::rubric`] and unit-test on CPU.
//!
//! Plan: [`docs/plans/2026-06-21-opd-ceiling-27b-dense.md`]. The judge takes the
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
use crate::rubric::{Rubric, Verdict, select};

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

    pub fn engine(&self) -> &Arc<Mutex<LoadedInferenceEngine>> {
        &self.engine
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

    /// Reload the judge engine's device weights before the next judging phase.
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
/// each with Flash, select the accepted, and write them back via `train_on_accepted`.
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
    judge: &FlashJudge,
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
            let verdicts: Vec<Verdict> = texts
                .iter()
                .map(|t| judge.judge_resilient(rubric, problem, t))
                .collect();
            let sel = select(&texts, &verdicts);
            rep.accepted += sel.accepted.len();
            rep.distinct_accepted += sel.distinct_accepted;
            rep.parse_errors += sel.parse_errors;
            for &idx in &sel.accepted {
                accepted_pairs.push((prompt_ids.clone(), samples[idx].clone()));
            }
            if sel.accepted.is_empty() {
                rejected.push((problem.as_str(), prompt_ids.clone()));
            }
        }

        // Mode B — Flash correction for rejected prompts (breaks the best-of-N ceiling).
        if cfg.correction_cap > 0 {
            let mut made = 0usize;
            for (problem, prompt_ids) in &rejected {
                if made >= cfg.correction_cap {
                    break;
                }
                match judge.correct(rubric, problem, cfg.correction_max_tokens) {
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
        if let Some(cap) = cfg.writeback_cap {
            if accepted_pairs.len() > cap {
                eprintln!(
                    "[rubric] round {round} writeback-cap: {} accepted -> training first {cap}",
                    accepted_pairs.len()
                );
                accepted_pairs.truncate(cap);
            }
        }

        // Phase B — offload BOTH inference engines (rollout + judge) to host RAM,
        // freeing their VRAM (~tens of GB) for the 27B autograd CE forward+backward.
        // Without this the CE step OOMs (`cuda alloc_zeros failed`) at seq ~1k.
        eprintln!(
            "[rubric] round {round} phase-B: offloading engines ({} accepted to train)",
            accepted_pairs.len()
        );
        let freed_rollout = student.offload_engine_weights().unwrap_or(0);
        let freed_judge = judge.offload_engine_weights().unwrap_or(0);
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
        eprintln!("[rubric] round {round} phase-D: reloading engines");
        student.reload_engine_weights()?;
        judge.reload_engine_weights()?;

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
