//! SWE-bench-Pro task loading for agent-based OPD.
//!
//! Loads SWE-bench-Pro records from a JSONL file and normalizes their
//! JSON-shape quirks; the cc harness ([`crate::cc_harness`]) builds the
//! rollout prompt from these fields.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SweTask {
    pub instance_id: String,
    pub problem_statement: String,
    pub repo: String,
    pub base_commit: String,
    /// Hidden tests — applied at scoring, not given to the agent.
    pub test_patch: String,

    /// May be a JSON array or a JSON-string of a JSON array. Use `fail_to_pass`.
    #[serde(default)]
    pub fail_to_pass: serde_json::Value,

    #[serde(default)]
    pub before_repo_set_cmd: Option<String>,
}

fn normalize_string_list(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        serde_json::Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.starts_with('[')
                && let Ok(serde_json::Value::Array(items)) =
                    serde_json::from_str::<serde_json::Value>(trimmed)
            {
                return items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect();
            }
            if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_owned()]
            }
        }
        _ => Vec::new(),
    }
}

impl SweTask {
    pub fn fail_to_pass(&self) -> Vec<String> {
        normalize_string_list(&self.fail_to_pass)
    }
}

pub fn load_swe_tasks(path: &Path) -> Result<Vec<SweTask>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open SWE task file {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut tasks = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("failed to read {} line {}", path.display(), idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let task: SweTask = serde_json::from_str(&line).with_context(|| {
            format!(
                "invalid SWE task JSON at {} line {}",
                path.display(),
                idx + 1
            )
        })?;
        tasks.push(task);
    }
    Ok(tasks)
}
