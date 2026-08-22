//! Rubric-OPD judge primitives: render a judge prompt for a strong teacher
//! (DeepSeek-V4-Flash), and parse its structured verdict. Pure host logic — no
//! engine/GPU — so it unit-tests on CPU.
//!
//! The judge scores at the *text* level (vocab-agnostic, sidestepping
//! cross-tokenizer KD); accepted rollouts become the student's own training
//! targets (RFT).
//!
//! Per the 2026 rubric-reward literature (Rubrics-as-Rewards 2507.17746,
//! step-wise rubric rewards 2605.17291): split **Factual** criteria from
//! **Process** criteria to resist reward-hacking, and require all Factual
//! criteria for acceptance. A judge output that cannot be parsed is **never
//! accepted and never silently bucketed** (a timeout/parse-fail is not a pass
//! and not a fail-class).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CriterionKind {
    /// Correctness of an intermediate or final result. All Factual criteria must pass.
    Factual,
    /// Quality of the reasoning process. Logged but does not gate acceptance.
    Process,
}

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rubric {
    pub task: String,
    pub criteria: Vec<Criterion>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    pub passed: Vec<(String, bool)>,
    pub accepted: bool,
    pub parse_error: bool,
}

impl Verdict {
    /// Never accepted — a judge timeout/garbage is neither a pass nor a fail-class.
    pub fn parse_error() -> Self {
        Self {
            passed: Vec::new(),
            accepted: false,
            parse_error: true,
        }
    }
}

impl Rubric {
    pub fn judge_prompt(&self, problem: &str, rollout: &str) -> String {
        let criteria = self
            .criteria
            .iter()
            .map(|c| {
                let tag = match c.kind {
                    CriterionKind::Factual => "factual",
                    CriterionKind::Process => "process",
                };
                format!("- {} ({tag}): {}\n", c.key, c.description)
            })
            .collect::<String>();
        let example = self
            .criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}\"{}\"): true", if i > 0 { ", " } else { "" }, c.key))
            .collect::<String>();
        format!(
            "You are a strict grader for {task}. Evaluate the SOLUTION against each criterion \
             below.\n\nPROBLEM:\n{problem}\n\nSOLUTION:\n{rollout}\n\nCRITERIA:\n{criteria}\n\
             Reason briefly, then on the FINAL line output ONLY a JSON object mapping every \
             criterion key to true or false, e.g. {{{example}}}\n",
            task = self.task
        )
    }

    pub fn solve_prompt(&self, problem: &str) -> String {
        let requirements = self
            .criteria
            .iter()
            .filter(|c| c.kind == CriterionKind::Factual)
            .map(|c| format!("- {}\n", c.description))
            .collect::<String>();
        format!(
            "You are an expert solving a {task} problem. Produce a correct, complete, \
             self-contained solution.\n\nPROBLEM:\n{problem}\n\nThe solution MUST \
             satisfy:\n{requirements}\nThink step by step, then state the final answer.\n",
            task = self.task
        )
    }

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
        let Some(passed) = self
            .criteria
            .iter()
            .map(|c| {
                map.get(&c.key)
                    .and_then(|v| v.as_bool())
                    .map(|b| (c.key.clone(), b))
            })
            .collect::<Option<Vec<_>>>()
        else {
            return reject; // missing/malformed key -> never accepted
        };
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub accepted: Vec<usize>,
    pub distinct_accepted: usize,
    pub parse_errors: usize,
}

pub fn select(rollouts: &[String], verdicts: &[Verdict]) -> Selection {
    assert_eq!(
        rollouts.len(),
        verdicts.len(),
        "select: rollouts/verdicts length mismatch"
    );
    let mut accepted = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut distinct_accepted = 0usize;
    let mut parse_errors = 0usize;
    for (i, v) in verdicts.iter().enumerate() {
        if v.parse_error {
            parse_errors += 1;
        }
        if v.accepted {
            accepted.push(i);
            if seen.insert(rollouts[i].as_str()) {
                distinct_accepted += 1;
            }
        }
    }
    Selection {
        accepted,
        distinct_accepted,
        parse_errors,
    }
}

fn extract_last_braced(text: &str, marker: &str) -> Option<String> {
    if marker.is_empty() {
        return None;
    }
    let mut last: Option<String> = None;
    let mut start = 0usize;
    loop {
        let Some(rel) = text[start..].find(marker) else {
            return last;
        };
        let pos = start + rel;
        let body_start = pos + marker.len();
        let mut depth = 1i32;
        let mut out = String::new();
        let mut iter = text[body_start..].char_indices();
        for (_, ch) in iter.by_ref() {
            match ch {
                '{' => {
                    depth += 1;
                    out.push(ch);
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let candidate = out.trim();
                        if !candidate.is_empty() {
                            last = Some(candidate.to_string());
                        }
                        break;
                    }
                    out.push(ch);
                }
                _ => out.push(ch),
            }
        }
        start = body_start;
    }
}

fn normalize_math_answer(answer: &str) -> String {
    let mut s = answer.trim().to_string();
    if let Some(boxed) = extract_last_braced(&s, "\\boxed{") {
        s = boxed;
    }
    s = s.replace("\\$", "");
    s = s.trim_matches('$').to_string();
    for (old, new) in [
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\:", ""),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
    ] {
        s = s.replace(old, new);
    }
    s = unwrap_text_macro(&s);
    s = s.replace(',', "");
    s = s.chars().filter(|c| !c.is_whitespace()).collect();
    s = s.trim_end_matches('.').to_string();
    s.to_lowercase()
}

fn unwrap_text_macro(s: &str) -> String {
    let pat = "\\text{";
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find(pat) {
        let body_start = pos + pat.len();
        if let Some(close_rel) = rest[body_start..].find('}') {
            let body = &rest[body_start..body_start + close_rel];
            if body.contains('{') {
                out.push_str(&rest[..body_start]);
                rest = &rest[body_start..];
                continue;
            }
            out.push_str(&rest[..pos]);
            out.push_str(body);
            rest = &rest[body_start + close_rel + 1..];
        } else {
            break;
        }
    }
    out.push_str(rest);
    out
}

pub fn select_by_self_consistency(rollouts: &[String]) -> Selection {
    let answers: Vec<Option<String>> = rollouts
        .iter()
        .map(|r| {
            let norm = normalize_math_answer(r);
            (!norm.is_empty()).then_some(norm)
        })
        .collect();

    let parse_errors = answers.iter().filter(|a| a.is_none()).count();

    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut order: Vec<&str> = Vec::new();
    for a in answers.iter().flatten() {
        let key = a.as_str();
        if !counts.contains_key(key) {
            order.push(key);
        }
        *counts.entry(key).or_insert(0) += 1;
    }
    let mut majority: Option<&str> = None;
    let mut best = 0usize;
    for &k in &order {
        let c = counts[k];
        if c > best {
            best = c;
            majority = Some(k);
        }
    }
    let Some(majority) = majority.map(str::to_string) else {
        return Selection {
            accepted: Vec::new(),
            distinct_accepted: 0,
            parse_errors,
        };
    };

    let mut accepted = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut distinct_accepted = 0usize;
    for (i, a) in answers.iter().enumerate() {
        if a.as_deref() == Some(majority.as_str()) {
            accepted.push(i);
            if seen.insert(rollouts[i].as_str()) {
                distinct_accepted += 1;
            }
        }
    }
    Selection {
        accepted,
        distinct_accepted,
        parse_errors,
    }
}

pub fn math_rubric() -> Rubric {
    Rubric {
        task: "math reasoning".to_string(),
        criteria: vec![
            Criterion::factual(
                "answer_correct",
                "The final \\boxed{} answer is mathematically correct for the problem.",
            ),
            Criterion::process(
                "steps_valid",
                "Each reasoning step follows logically from the previous; no unjustified leaps.",
            ),
        ],
    }
}

pub fn bfcl_agentic_rubric() -> Rubric {
    Rubric {
        task: "agentic tool-use".to_string(),
        criteria: vec![
            Criterion::factual(
                "call_correct",
                "If the query is answerable with the given tools, the response calls the correct \
                 tool(s) with correct arguments; if NOT answerable, it abstains and emits no tool call.",
            ),
            Criterion::process(
                "reasons_before_acting",
                "The response reasons about tool relevance before deciding to call or abstain.",
            ),
        ],
    }
}
