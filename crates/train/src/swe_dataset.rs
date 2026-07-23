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

/// One SWE-bench-Pro task record. Only the fields the OPD rollout needs are
/// decoded; any extra columns in the JSONL are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct SweTask {
    /// Unique task id, e.g. `"ansible__ansible-12345"`.
    pub instance_id: String,
    /// The issue text the agent must fix.
    pub problem_statement: String,
    /// Repository slug, e.g. `"ansible/ansible"`.
    pub repo: String,
    /// Git sha the repo is checked out at.
    pub base_commit: String,
    /// Git diff adding/modifying the hidden tests — applied at scoring time,
    /// NOT given to the agent.
    pub test_patch: String,

    /// Tests that must flip from fail to pass. In SWE-bench-Pro this is often a
    /// JSON-string containing a JSON array, sometimes already a JSON array.
    /// Use [`SweTask::fail_to_pass`] to normalize.
    #[serde(default)]
    pub fail_to_pass: serde_json::Value,

    /// Setup commands run in the sandbox before tests; may be absent.
    #[serde(default)]
    pub before_repo_set_cmd: Option<String>,
}

/// Normalize a `serde_json::Value` that may be a JSON array of strings, a
/// JSON-string containing a JSON array, or a single bare string into a
/// `Vec<String>`. Anything else (null, unparseable) yields an empty vec.
fn normalize_string_list(value: &serde_json::Value) -> Vec<String> {
    match value {
        // Already a JSON array: collect the string elements.
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        // A string: it may itself be a JSON array literal (the common
        // SWE-bench-Pro shape), otherwise treat it as a single entry.
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
    /// `fail_to_pass` normalized to a `Vec<String>` regardless of JSON shape.
    pub fn fail_to_pass(&self) -> Vec<String> {
        normalize_string_list(&self.fail_to_pass)
    }
}

/// Load all tasks from a JSONL file (one JSON object per line; blank lines are
/// skipped). Each non-blank line must parse as a [`SweTask`].
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

#[cfg(test)]
mod tests {
    use super::*;

    fn task_from_line(line: &str) -> SweTask {
        serde_json::from_str(line).expect("valid SweTask JSON")
    }

    #[test]
    fn parses_string_or_array_fail_to_pass() {
        // Case 1: fail_to_pass is a JSON-string containing a JSON array.
        let line = r#"{
            "instance_id": "ansible__ansible-1",
            "problem_statement": "fix the bug",
            "repo": "ansible/ansible",
            "base_commit": "abc123",
            "test_patch": "diff --git a b",
            "fail_to_pass": "[\"a::b\", \"c::d\"]"
        }"#;
        let task = task_from_line(line);
        assert_eq!(task.fail_to_pass(), vec!["a::b", "c::d"]);

        // Case 2: fail_to_pass is already a JSON array.
        let line = r#"{
            "instance_id": "ansible__ansible-2",
            "problem_statement": "fix the bug",
            "repo": "ansible/ansible",
            "base_commit": "abc123",
            "test_patch": "diff --git a b",
            "fail_to_pass": ["a::b", "c::d"]
        }"#;
        let task = task_from_line(line);
        assert_eq!(task.fail_to_pass(), vec!["a::b", "c::d"]);
    }

    #[test]
    fn loads_jsonl_skipping_blanks() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"instance_id":"a__b-1","problem_statement":"p","repo":"a/b","base_commit":"c","test_patch":"d","fail_to_pass":"[\"t::1\"]"}}"#
        )
        .unwrap();
        writeln!(file).unwrap(); // blank line
        writeln!(
            file,
            r#"{{"instance_id":"a__b-2","problem_statement":"p","repo":"a/b","base_commit":"c","test_patch":"d","fail_to_pass":["t::2"]}}"#
        )
        .unwrap();
        file.flush().unwrap();

        let tasks = load_swe_tasks(file.path()).unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].instance_id, "a__b-1");
        assert_eq!(tasks[0].fail_to_pass(), vec!["t::1"]);
        assert_eq!(tasks[1].fail_to_pass(), vec!["t::2"]);
    }
}
