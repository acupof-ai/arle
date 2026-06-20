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
use crate::rubric::{Rubric, Verdict};

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
        Ok(rubric.parse_verdict(&output.text))
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
}
