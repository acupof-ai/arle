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

    fn judge_resilient(&self, rubric: &Rubric, problem: &str, rollout: &str) -> Verdict {
        match self.judge(rubric, problem, rollout) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("rubric_opd: judge call failed (counted as parse error): {err}");
                Verdict::parse_error()
            }
        }
    }

    fn judge_batch(&self, rubric: &Rubric, problem: &str, rollouts: &[String]) -> Vec<Verdict> {
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
        };
        let output = {
            let mut engine = self
                .engine
                .lock()
                .map_err(|err| anyhow!("Flash judge engine lock poisoned: {err}"))?;
            engine.complete(req)?
        };
        let solution = output.text;
        let verdict = self.judge_resilient(rubric, problem, &solution);
        Ok(verdict.accepted.then_some(solution))
    }

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

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct RubricOpdConfig {
    pub rounds: usize,
    pub samples_per_prompt: usize,
    pub max_new_tokens: usize,
    pub writeback_cap: Option<usize>,
    pub writeback_batch: usize,
    pub correction_cap: usize,
    pub correction_max_tokens: usize,
    /// When `true`, the autograd student's frozen FP8 base aliases the rollout
    /// engine's resident base bytes, so the Phase-B rollout offload is skipped
    /// (freeing those bytes would crash the CE forward reading the alias).
    pub share_frozen_base: bool,
    pub distill_shortest: bool,
}

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

/// Sample N rollouts per prompt, judge, select accepted, write back via CE.
/// `judge = None` = self-consistency mode (no corrections).
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

        // Bound the CE set — 27B host-authoritative CE is minutes/step.
        if let Some(cap) = cfg.writeback_cap
            && accepted_pairs.len() > cap
        {
            eprintln!(
                "[rubric] round {round} writeback-cap: {} accepted -> training first {cap}",
                accepted_pairs.len()
            );
            accepted_pairs.truncate(cap);
        }

        // Offload engines to host RAM for the 27B CE forward+backward.
        // `share_frozen_base`: the autograd student's frozen base ALIASES the
        // rollout engine's resident base bytes — freeing them would crash the
        // CE forward. Skip rollout offload when sharing; judge still offloads.
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
            // Base stays resident (aliased); scratch + KV are dead during CE.
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

        // Reload engines. `share_frozen_base`: rollout base was never offloaded
        // (aliased by student), so skip its reload — only re-acquire the KV pool.
        eprintln!("[rubric] round {round} phase-D: reloading engines");
        if !cfg.share_frozen_base {
            student.reload_engine_weights()?;
        } else {
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
