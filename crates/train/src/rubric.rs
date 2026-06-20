//! Rubric-OPD judge primitives: render a judge prompt for a strong teacher
//! (DeepSeek-V4-Flash), and parse its structured verdict. Pure host logic — no
//! engine/GPU — so it unit-tests on CPU.
//!
//! Route + rationale: [`docs/plans/2026-06-21-opd-ceiling-27b-dense.md`]. The
//! student generates on-policy rollouts; the judge (Flash) scores each against a
//! rubric at the *text* level (vocab-agnostic, sidestepping cross-tokenizer KD);
//! accepted rollouts are written back as the student's own training targets (RFT).
//!
//! Per the 2026 rubric-reward literature (Rubrics-as-Rewards 2507.17746, step-wise
//! rubric rewards 2605.17291): split **Factual** criteria (correctness of
//! intermediate/final results) from **Process** criteria (valid reasoning steps) to
//! resist reward-hacking, and require all Factual criteria for acceptance. A judge
//! output that cannot be parsed is **never accepted and never silently bucketed**
//! (CLAUDE.md §0 case-as-fact: a timeout/parse-fail is not a pass and not a fail-class).

use serde::{Deserialize, Serialize};

/// Whether a criterion gates acceptance (Factual) or is advisory/logged (Process).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CriterionKind {
    /// Correctness of an intermediate or final result. **All Factual criteria must
    /// pass for the rollout to be accepted.**
    Factual,
    /// Quality of the reasoning process. Logged and reportable, but does not by
    /// itself gate acceptance (avoids penalizing correct answers with terse steps).
    Process,
}

/// One rubric criterion. `key` is the machine key the judge must emit in its JSON
/// verdict; `description` is the human-readable instruction shown to the judge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Criterion {
    pub key: String,
    pub description: String,
    pub kind: CriterionKind,
}

impl Criterion {
    pub fn factual(key: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            description: description.into(),
            kind: CriterionKind::Factual,
        }
    }
    pub fn process(key: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            description: description.into(),
            kind: CriterionKind::Process,
        }
    }
}

/// A task rubric: a short task name plus its criteria.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rubric {
    pub task: String,
    pub criteria: Vec<Criterion>,
}

/// Parsed judge verdict. `parse_error` is set when the judge output could not be
/// mapped to every criterion key; such a rollout is never accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// Per-criterion (key, passed). Empty when `parse_error`.
    pub passed: Vec<(String, bool)>,
    /// All Factual criteria passed and the verdict parsed cleanly.
    pub accepted: bool,
    /// The judge output could not be parsed into every criterion key.
    pub parse_error: bool,
}

impl Rubric {
    /// Render the prompt sent to the judge model. Asks for a single JSON object
    /// keyed by each criterion `key` with boolean values, emitted as the final line.
    pub fn judge_prompt(&self, problem: &str, rollout: &str) -> String {
        let mut s = String::new();
        s.push_str("You are a strict grader for ");
        s.push_str(&self.task);
        s.push_str(". Evaluate the SOLUTION against each criterion below.\n\n");
        s.push_str("PROBLEM:\n");
        s.push_str(problem);
        s.push_str("\n\nSOLUTION:\n");
        s.push_str(rollout);
        s.push_str("\n\nCRITERIA:\n");
        for c in &self.criteria {
            let tag = match c.kind {
                CriterionKind::Factual => "factual",
                CriterionKind::Process => "process",
            };
            s.push_str("- ");
            s.push_str(&c.key);
            s.push_str(" (");
            s.push_str(tag);
            s.push_str("): ");
            s.push_str(&c.description);
            s.push('\n');
        }
        s.push_str(
            "\nReason briefly, then on the FINAL line output ONLY a JSON object mapping \
             every criterion key to true or false, e.g. {",
        );
        for (i, c) in self.criteria.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push('"');
            s.push_str(&c.key);
            s.push_str("\": true");
        }
        s.push_str("}\n");
        s
    }

    /// Parse a judge output into a [`Verdict`]. Extracts the last balanced `{...}`
    /// JSON object, requires a boolean for every criterion key; any missing key or
    /// unparseable JSON yields `parse_error = true` (never accepted).
    pub fn parse_verdict(&self, judge_output: &str) -> Verdict {
        let reject = Verdict {
            passed: Vec::new(),
            accepted: false,
            parse_error: true,
        };
        let Some(obj) = last_json_object(judge_output) else {
            return reject;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&obj) else {
            return reject;
        };
        let Some(map) = value.as_object() else {
            return reject;
        };
        let mut passed = Vec::with_capacity(self.criteria.len());
        for c in &self.criteria {
            match map.get(&c.key).and_then(|v| v.as_bool()) {
                Some(b) => passed.push((c.key.clone(), b)),
                None => return reject, // missing/malformed key -> never accepted
            }
        }
        let accepted = self
            .criteria
            .iter()
            .zip(&passed)
            .filter(|(c, _)| c.kind == CriterionKind::Factual)
            .all(|(_, (_, ok))| *ok);
        Verdict {
            passed,
            accepted,
            parse_error: false,
        }
    }
}

/// Extract the last balanced top-level `{...}` substring (the judge may emit prose
/// before its JSON verdict). Returns `None` if no balanced object is found.
fn last_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let end = text.rfind('}')?;
    let mut depth = 0i32;
    let mut i = end as isize;
    while i >= 0 {
        match bytes[i as usize] {
            b'}' => depth += 1,
            b'{' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[i as usize..=end].to_string());
                }
            }
            _ => {}
        }
        i -= 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn math_rubric() -> Rubric {
        Rubric {
            task: "math reasoning".to_string(),
            criteria: vec![
                Criterion::factual("answer_correct", "The final boxed answer is correct."),
                Criterion::process("steps_valid", "Each step follows from the previous."),
            ],
        }
    }

    #[test]
    fn judge_prompt_contains_problem_rollout_and_keys() {
        let r = math_rubric();
        let p = r.judge_prompt("2+2=?", "The answer is \\boxed{4}.");
        assert!(p.contains("2+2=?"));
        assert!(p.contains("\\boxed{4}"));
        assert!(p.contains("answer_correct"));
        assert!(p.contains("steps_valid"));
        assert!(p.contains("math reasoning"));
    }

    #[test]
    fn parse_clean_json_accepts_when_factual_passes() {
        let r = math_rubric();
        let v = r.parse_verdict(r#"{"answer_correct": true, "steps_valid": true}"#);
        assert!(!v.parse_error);
        assert!(v.accepted);
        assert_eq!(
            v.passed,
            vec![
                ("answer_correct".to_string(), true),
                ("steps_valid".to_string(), true),
            ]
        );
    }

    #[test]
    fn parse_prose_wrapped_json_and_factual_fail_rejects() {
        let r = math_rubric();
        let out = "Let me check. The answer 5 is wrong.\nVerdict:\n{\"answer_correct\": false, \"steps_valid\": true}";
        let v = r.parse_verdict(out);
        assert!(!v.parse_error);
        assert!(!v.accepted); // factual failed
    }

    #[test]
    fn process_fail_alone_does_not_block_acceptance() {
        let r = math_rubric();
        let v = r.parse_verdict(r#"{"answer_correct": true, "steps_valid": false}"#);
        assert!(!v.parse_error);
        assert!(v.accepted); // only Process failed; Factual gates acceptance
    }

    #[test]
    fn unparseable_or_missing_key_never_accepted() {
        let r = math_rubric();
        let garbage = r.parse_verdict("no json here, the model timed out");
        assert!(garbage.parse_error);
        assert!(!garbage.accepted);
        let missing = r.parse_verdict(r#"{"steps_valid": true}"#);
        assert!(missing.parse_error); // factual key absent -> reject, not silent pass
        assert!(!missing.accepted);
    }

    #[test]
    fn picks_last_json_object_when_multiple() {
        let r = math_rubric();
        let out = "draft {\"answer_correct\": false, \"steps_valid\": false} final {\"answer_correct\": true, \"steps_valid\": true}";
        let v = r.parse_verdict(out);
        assert!(v.accepted);
    }
}
