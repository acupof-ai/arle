//! SWE-bench-Pro task loading + agent-prompt construction for agent-based OPD.
//!
//! Loads SWE-bench-Pro records from a JSONL file and builds the initial agent
//! prompt (system + first user message) for a coding rollout. The repo
//! environment / file tree is gathered at runtime through the sandbox executor
//! (a later module), so the prompts here are static guidance plus the task's
//! repo name and problem statement only.

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

    /// Test files to run. Optional; tolerate string, array, or absent.
    /// Use [`SweTask::selected_test_files`] to normalize.
    #[serde(default)]
    pub selected_test_files_to_run: serde_json::Value,

    /// Setup commands run in the sandbox before tests; may be absent.
    #[serde(default)]
    pub before_repo_set_cmd: Option<String>,

    /// Optional extra requirements text; may be absent.
    #[serde(default)]
    pub requirements: Option<String>,
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

    /// `selected_test_files_to_run` normalized to a `Vec<String>`.
    pub fn selected_test_files(&self) -> Vec<String> {
        normalize_string_list(&self.selected_test_files_to_run)
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

const PROMPT_CHAR_BUDGET: usize = 3000;

/// Truncate `text` to at most `budget` chars (on a char boundary), appending an
/// elision marker when truncated.
fn truncate_chars(text: &str, budget: usize) -> String {
    if text.chars().count() <= budget {
        return text.to_owned();
    }
    let mut out: String = text.chars().take(budget).collect();
    out.push_str("\n…[truncated]");
    out
}

/// Static system prompt for the coding agent: autonomous agent operating in the
/// task's repo (cwd = repo root), the four tools available, and the workflow.
///
/// Deliberately excludes any environment / repo-tree detail — that is gathered
/// at runtime by the sandbox executor.
///
/// `think` selects the Qwen3.x thinking soft-switch: `false` appends
/// `/no_think` (tool calls with no deliberation — the cheap default);
/// `true` leaves thinking enabled (the 2026-06-20 think-on precedent:
/// enable thinking EVERYWHERE, no `<think>`-span masking).
pub fn agent_system_prompt(task: &SweTask, think: bool) -> String {
    let closing = if think {
        "Think the problem through, then act with tool calls."
    } else {
        "Respond directly with a tool call — do not deliberate at length. /no_think"
    };
    format!(
        "You are an autonomous coding agent fixing a bug in the `{repo}` \
repository. Your working directory is the repo root (cwd = repo root); the \
repo is already checked out and ready to edit.

You have exactly four tools:
- `read`: read a file (use it to inspect source before editing).
- `write`: create or overwrite a file with new contents.
- `replace`: replace an exact string in a file — prefer this for edits, it is \
the smallest correct change.
- `bash`: run a shell command (use it to explore the tree, e.g. `ls`, `grep`, \
`find`).

All paths are RELATIVE to the repo root (cwd). Use `lib/ansible/...`, \
`test/...` — never absolute paths like `/repo` or `/work`.

Workflow:
1. Explore briefly with `bash` and `read` (a few steps — locate the relevant \
files and understand the failure).
2. Make the fix: you MUST apply at least one edit with `replace` (or `write`) \
— the SMALLEST correct change that resolves the issue. Do not finish without \
editing a file; do not refactor unrelated code.
3. Do NOT run the hidden tests — they are not present and will be applied at \
scoring time. Do not try to write or guess the tests.
If you have not edited a file within your first 5 tool calls, STOP exploring \
and apply your best minimal fix NOW — an imperfect edit beats no edit.
4. Only after you have edited a file, reply with a short final message stating \
what you changed and why.

Keep the change minimal and correct.

{closing}",
        repo = task.repo,
    )
}

/// The first user message: the problem statement, plus requirements when
/// present, each truncated to a sane char budget (~3000 chars).
pub fn agent_user_prompt(task: &SweTask, think: bool) -> String {
    let mut prompt = String::new();
    prompt.push_str("Problem statement:\n");
    prompt.push_str(&truncate_chars(&task.problem_statement, PROMPT_CHAR_BUDGET));

    if let Some(requirements) = task.requirements.as_deref() {
        let requirements = requirements.trim();
        if !requirements.is_empty() {
            prompt.push_str("\n\nAdditional requirements:\n");
            prompt.push_str(&truncate_chars(requirements, PROMPT_CHAR_BUDGET));
        }
    }

    // Qwen3.x soft-switch: with `think` off the student emits tool calls
    // directly instead of an empty <think> block that strips to nothing
    // (decoded EmptyNoProgress root cause). Scoped to agent-OPD, not the
    // chat renderer, so the local `arle run` agent keeps thinking available.
    if !think {
        prompt.push_str("\n\n/no_think");
    }
    prompt
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
    fn selected_test_files_tolerates_shapes() {
        // Absent field -> empty.
        let line = r#"{
            "instance_id": "x__y-1",
            "problem_statement": "p",
            "repo": "x/y",
            "base_commit": "c",
            "test_patch": "d"
        }"#;
        let task = task_from_line(line);
        assert!(task.selected_test_files().is_empty());

        // Single string.
        let line = r#"{
            "instance_id": "x__y-2",
            "problem_statement": "p",
            "repo": "x/y",
            "base_commit": "c",
            "test_patch": "d",
            "selected_test_files_to_run": "tests/test_foo.py"
        }"#;
        let task = task_from_line(line);
        assert_eq!(task.selected_test_files(), vec!["tests/test_foo.py"]);

        // JSON array.
        let line = r#"{
            "instance_id": "x__y-3",
            "problem_statement": "p",
            "repo": "x/y",
            "base_commit": "c",
            "test_patch": "d",
            "selected_test_files_to_run": ["tests/a.py", "tests/b.py"]
        }"#;
        let task = task_from_line(line);
        assert_eq!(task.selected_test_files(), vec!["tests/a.py", "tests/b.py"]);
    }

    #[test]
    fn prompts_mention_repo_and_tools() {
        let line = r#"{
            "instance_id": "ansible__ansible-3",
            "problem_statement": "the foo module crashes on empty input",
            "repo": "ansible/ansible",
            "base_commit": "abc123",
            "test_patch": "diff",
            "fail_to_pass": "[\"t::a\"]",
            "requirements": "must keep backward compat"
        }"#;
        let task = task_from_line(line);

        let system = agent_system_prompt(&task, false);
        assert!(system.contains("ansible/ansible"));
        for tool in ["`read`", "`write`", "`replace`", "`bash`"] {
            assert!(system.contains(tool), "system prompt missing {tool}");
        }
        assert!(system.contains("cwd = repo root"));
        assert!(system.contains("/no_think"));

        let user = agent_user_prompt(&task, false);
        assert!(user.contains("the foo module crashes on empty input"));
        assert!(user.contains("must keep backward compat"));
        assert!(user.contains("/no_think"));

        // Think mode drops the soft-switch from BOTH prompts.
        assert!(!agent_system_prompt(&task, true).contains("/no_think"));
        assert!(!agent_user_prompt(&task, true).contains("/no_think"));
    }

    #[test]
    fn user_prompt_truncates_long_statement() {
        let long = "x".repeat(PROMPT_CHAR_BUDGET + 500);
        let line = format!(
            r#"{{
                "instance_id": "x__y-4",
                "problem_statement": "{long}",
                "repo": "x/y",
                "base_commit": "c",
                "test_patch": "d"
            }}"#
        );
        let task = task_from_line(&line);
        let user = agent_user_prompt(&task, false);
        assert!(user.contains("[truncated]"));
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
