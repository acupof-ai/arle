//! Math-reasoning rollout harness: K concurrent non-streaming `/v1/messages`
//! requests per task against the in-process serve, boxed-answer grading.
//! Device-neutral (HTTP + tokenizer only); the training loop lives in the CLI.

use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::cc_convert::CcWindow;

const MATH_SYSTEM_PROMPT: &str =
    "Solve the problem. Show your reasoning, then put your final answer in \\boxed{}.";

#[derive(Debug, Clone, Deserialize)]
pub struct MathTask {
    pub text: String,
    pub answer: String,
    #[serde(default)]
    pub source: Option<String>,
}

pub fn load_tasks(path: &Path, limit: Option<usize>) -> Result<Vec<MathTask>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut tasks = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let task = serde_json::from_str::<MathTask>(&line)
            .with_context(|| format!("parse {} line {}", path.display(), idx + 1))?;
        tasks.push(task);
        if let Some(n) = limit
            && tasks.len() >= n
        {
            break;
        }
    }
    ensure!(!tasks.is_empty(), "no tasks in {}", path.display());
    Ok(tasks)
}

/// LAST `\boxed{...}` with brace nesting; None when absent or unbalanced.
pub fn last_boxed(text: &str) -> Option<String> {
    const TAG: &str = "\\boxed{";
    let open = text.rfind(TAG)?;
    let close = matching_brace(text, open + TAG.len() - 1)?;
    Some(text[open + TAG.len()..close].to_owned())
}

/// One normalization pass: unwrap text-font wrappers, strip spacing/left/right
/// commands, rewrite dfrac/tfrac. Shrinks or rewrites one-way, so the caller
/// can iterate to a fixpoint.
fn normalize_pass(s: &str) -> String {
    strip_commands(&unwrap_wrappers(s))
}

const WRAPPERS: &[&str] = &[
    "\\text{",
    "\\textbf{",
    "\\mathrm{",
    "\\mbox{",
    "\\operatorname{",
    "\\mathbf{",
];

fn unwrap_wrappers(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(rel) = rest.find('\\') {
        let Some(w) = WRAPPERS.iter().find(|w| rest[rel..].starts_with(**w)) else {
            out.push_str(&rest[..=rel]);
            rest = &rest[rel + 1..];
            continue;
        };
        let open = rel + w.len() - 1;
        match matching_brace(rest, open) {
            Some(close) => {
                out.push_str(&rest[..rel]);
                out.push_str(&rest[open + 1..close]);
                rest = &rest[close + 1..];
            }
            // Unbalanced: keep the backslash and rescan past it.
            None => {
                out.push_str(&rest[..=rel]);
                rest = &rest[rel + 1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn matching_brace(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (idx, ch) in s[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + idx);
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_commands(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while !rest.is_empty() {
        let ch = rest.chars().next().unwrap();
        if ch == '$' || ch.is_whitespace() {
            rest = &rest[ch.len_utf8()..];
            continue;
        }
        if ch == '\\' {
            if let Some(next) = rest[1..].chars().next()
                && matches!(next, '!' | ',' | ';' | ':' | ' ')
            {
                rest = &rest[1 + next.len_utf8()..];
                continue;
            }
            if let Some(after) =
                strip_bare_command(rest, "\\left").or_else(|| strip_bare_command(rest, "\\right"))
            {
                rest = after;
                continue;
            }
            if let Some((from, to)) = [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac")]
                .into_iter()
                .find(|(from, _)| strip_bare_command(rest, from).is_some())
            {
                out.push_str(to);
                rest = &rest[from.len()..];
                continue;
            }
        }
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

/// `\cmd` stripped only at a command boundary (`\leftarrow` keeps its letters).
fn strip_bare_command<'a>(rest: &'a str, cmd: &str) -> Option<&'a str> {
    if rest.starts_with(cmd)
        && rest[cmd.len()..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_alphabetic())
    {
        Some(&rest[cmd.len()..])
    } else {
        None
    }
}

/// Shared by the grader and the corpus washer: wrapper-unwrap + spacing strip
/// to fixpoint, then thousands-commas when numeric, else lowercase.
pub fn canonicalize(s: &str) -> String {
    let mut current = s.to_owned();
    loop {
        let next = normalize_pass(&current);
        if next == current {
            return finalize(&next);
        }
        current = next;
    }
}

fn finalize(s: &str) -> String {
    if is_numeric(s) {
        s.replace(',', "")
    } else {
        s.to_lowercase()
    }
}

fn is_numeric(s: &str) -> bool {
    let mut t = s;
    if let Some(rest) = t.strip_prefix(['+', '-']) {
        t = rest;
    }
    let mut digits = 0usize;
    let mut dot = false;
    for ch in t.chars() {
        match ch {
            '0'..='9' => digits += 1,
            ',' => {}
            '.' if !dot => dot = true,
            _ => return false,
        }
    }
    digits > 0
}

pub fn grade(gold: &str, completion: &str) -> bool {
    last_boxed(completion).is_some_and(|ans| canonicalize(&ans) == canonicalize(gold))
}

pub struct MathHarness {
    pub base_url: String,
    pub model_id: String,
    pub dump_dir: PathBuf,
    pub tokenizer: tokenizers::Tokenizer,
    pub max_tokens: usize,
    pub agent: ureq::Agent,
}

#[derive(Debug, Clone)]
pub struct MathSample {
    pub sample: usize,
    pub model_tag: String,
    pub t_start_ms: u64,
    pub t_end_ms: u64,
    /// Thinking + answer blocks concatenated (the graded completion).
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub think_tokens: u64,
    pub answer_tokens: u64,
    pub timed_out: bool,
    pub capped: bool,
}

#[derive(Debug)]
pub struct GroupRollout {
    pub samples: Vec<MathSample>,
}

static GROUP_NONCE: AtomicU64 = AtomicU64::new(0);

pub fn next_nonce() -> u64 {
    GROUP_NONCE.fetch_add(1, Ordering::Relaxed)
}

/// `<model>#g<nonce>s<sample>` — distinct per sample for dump attribution.
fn sample_model(model_id: &str, nonce: u64, sample: usize) -> String {
    format!("{model_id}#g{nonce}s{sample}")
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

impl MathHarness {
    /// Shared HTTP agent with connection pooling; built once per harness so
    /// the K rollout samples and eval tasks reuse one TCP pool. Lives in the
    /// train crate because `ureq` is not a cli dependency.
    pub fn build_agent(timeout_secs: u64) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
    }

    pub fn run_group(
        &self,
        task: &MathTask,
        k: usize,
        temperature: f32,
        nonce: u64,
    ) -> GroupRollout {
        let k = k.max(1);
        let samples = std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::sync_channel(k);
            for sample in 0..k {
                let tx = tx.clone();
                scope.spawn(move || {
                    let _ = tx.send(self.run_sample(task, nonce, sample, temperature));
                });
            }
            drop(tx);
            let mut out = Vec::with_capacity(k);
            while let Ok(s) = rx.recv() {
                out.push(s);
            }
            out.sort_by_key(|s| s.sample);
            out
        });
        GroupRollout { samples }
    }

    fn run_sample(
        &self,
        task: &MathTask,
        nonce: u64,
        sample: usize,
        temperature: f32,
    ) -> MathSample {
        let model_tag = sample_model(&self.model_id, nonce, sample);
        // Qwen templates default thinking OFF; the explicit field turns it on,
        // and budget == max_tokens so the thinking cap never binds first.
        let body = serde_json::json!({
            "model": model_tag,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            // top_k=-1/top_p=1.0: sidecar logprobs must be full-softmax, not the serve's filtered default (GSPO IS ratio).
            "top_k": -1,
            "top_p": 1.0,
            "system": MATH_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": task.text}],
            "thinking": {"type": "enabled", "budget_tokens": self.max_tokens},
        });
        let agent = &self.agent;
        let t_start_ms = epoch_ms();
        let outcome = agent
            .post(&format!("{}/v1/messages", self.base_url))
            .send_json(body);
        let t_end_ms = epoch_ms();
        match outcome {
            Ok(resp) => match resp.into_json::<serde_json::Value>() {
                Ok(v) => {
                    let (think, answer) = split_blocks(v.get("content"));
                    let input_tokens = v["usage"]["input_tokens"].as_u64().unwrap_or(0);
                    let output_tokens = v["usage"]["output_tokens"].as_u64().unwrap_or(0);
                    let stop = v["stop_reason"].as_str().unwrap_or("");
                    let capped = stop == "max_tokens" || output_tokens as usize >= self.max_tokens;
                    let count = |text: &str| {
                        self.tokenizer
                            .encode(text, false)
                            .map(|e| e.get_ids().len() as u64)
                            .unwrap_or(0)
                    };
                    MathSample {
                        sample,
                        model_tag,
                        t_start_ms,
                        t_end_ms,
                        text: format!("{think}{answer}"),
                        input_tokens,
                        output_tokens,
                        think_tokens: count(&think),
                        answer_tokens: count(&answer),
                        timed_out: false,
                        capped,
                    }
                }
                Err(err) => {
                    eprintln!("[math-harness] sample {sample} decode failed: {err}");
                    failed_sample(sample, model_tag, t_start_ms, t_end_ms, false)
                }
            },
            Err(err) => {
                let timed_out = is_timeout(&err);
                eprintln!("[math-harness] sample {sample} request failed: {err}");
                failed_sample(sample, model_tag, t_start_ms, t_end_ms, timed_out)
            }
        }
    }
}

fn failed_sample(
    sample: usize,
    model_tag: String,
    t_start_ms: u64,
    t_end_ms: u64,
    timed_out: bool,
) -> MathSample {
    MathSample {
        sample,
        model_tag,
        t_start_ms,
        t_end_ms,
        text: String::new(),
        input_tokens: 0,
        output_tokens: 0,
        think_tokens: 0,
        answer_tokens: 0,
        timed_out,
        capped: false,
    }
}

/// The non-streaming response emits `thinking` blocks before `text` blocks
/// when the request enabled thinking; concatenated, they are the completion.
fn split_blocks(content: Option<&serde_json::Value>) -> (String, String) {
    let Some(blocks) = content.and_then(|c| c.as_array()) else {
        return (String::new(), String::new());
    };
    let mut think = String::new();
    let mut answer = String::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("thinking") => think.push_str(block["thinking"].as_str().unwrap_or("")),
            Some("text") => answer.push_str(block["text"].as_str().unwrap_or("")),
            _ => {}
        }
    }
    (think, answer)
}

fn is_timeout(err: &ureq::Error) -> bool {
    let ureq::Error::Transport(transport) = err else {
        return false;
    };
    let mut source = transport.source();
    while let Some(e) = source {
        if e.downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::TimedOut)
        {
            return true;
        }
        source = e.source();
    }
    false
}

pub fn to_windows(task_id: &str, rollout: &GroupRollout) -> Vec<CcWindow> {
    rollout
        .samples
        .iter()
        .map(|s| CcWindow {
            label: format!("{task_id}#{}", s.sample),
            t_start_ms: s.t_start_ms,
            t_end_ms: s.t_end_ms,
            reward: 0.0,
            errored: s.timed_out,
            model: Some(s.model_tag.clone()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_boxed_nested_and_last() {
        assert_eq!(last_boxed(r"\boxed{42}").as_deref(), Some("42"));
        assert_eq!(last_boxed(r"a \boxed{1} b \boxed{2}").as_deref(), Some("2"));
        assert_eq!(
            last_boxed(r"\boxed{\frac{1}{2}}").as_deref(),
            Some(r"\frac{1}{2}")
        );
        assert_eq!(last_boxed(r"\boxed{\boxed{x}}").as_deref(), Some("x"));
        assert_eq!(last_boxed("no box here"), None);
        assert_eq!(last_boxed(r"\boxed{unbalanced"), None);
    }

    #[test]
    fn canonicalize_numeric() {
        assert_eq!(canonicalize("1,000"), "1000");
        assert_eq!(canonicalize(" 1,000 "), "1000");
        assert_eq!(canonicalize("-1,000.5"), "-1000.5");
        assert_eq!(canonicalize("$1.5$"), "1.5");
        assert_eq!(canonicalize("3.14"), "3.14");
        assert_eq!(canonicalize("1.2.3"), "1.2.3");
    }

    #[test]
    fn canonicalize_spacing_and_case() {
        assert_eq!(canonicalize(r"1\,000"), "1000");
        assert_eq!(canonicalize("X + Y"), "x+y");
        assert_eq!(canonicalize(r"x\! y\; z\:"), "xyz");
    }

    #[test]
    fn canonicalize_wrappers() {
        assert_eq!(canonicalize(r"\textbf{(C)}19"), "(c)19");
        assert_eq!(canonicalize(r"180\text{minutes}"), "180minutes");
        assert_eq!(canonicalize(r"\text{\textbf{x}}"), "x");
        assert_eq!(canonicalize(r"\mathrm{sin}\left(x\right)"), "sin(x)");
        assert_eq!(canonicalize(r"\dfrac{1}{2}"), r"\frac{1}{2}");
        assert_eq!(canonicalize(r"\tfrac{a}{b}"), r"\frac{a}{b}");
        // \leftarrow is not \left.
        assert_eq!(canonicalize(r"\leftarrow"), r"\leftarrow");
    }

    #[test]
    fn grade_boxed() {
        assert!(grade("42", r"the answer is \boxed{42}."));
        assert!(grade("1000", r"got \boxed{1,000}"));
        assert!(grade(r"\frac{1}{2}", r"result: \boxed{\dfrac{1}{2}}"));
        assert!(grade("ABC", r"\boxed{abc}"));
        assert!(grade(r"(C)19", r"\boxed{\textbf{(C)}19}"));
        assert!(!grade("42", "no boxed answer"));
        assert!(!grade("42", r"\boxed{43}"));
    }
}
