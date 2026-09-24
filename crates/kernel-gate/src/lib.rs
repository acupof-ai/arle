//! Operator registry (`operators/registry.toml`) schema, validation, and the
//! parity-gate list the GPU batch runs.
//!
//! A registry has three row kinds: `[[semantic]]` (what an operator computes and
//! how its correctness is gated), `[[implementation]]` (one provider/kernel for a
//! semantic, with its legality and fallback), and `[policy."<semantic id>"]`
//! (numeric tolerances and fallback status). Unknown fields are rejected.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Where the GPU batch's standalone parity examples live, relative to the root.
pub const DEFAULT_GATE_PREFIX: &str = "crates/infer-cuda/examples/";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    #[serde(default)]
    pub semantic: Vec<Semantic>,
    #[serde(default)]
    pub implementation: Vec<Implementation>,
    #[serde(default)]
    pub policy: BTreeMap<String, Policy>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Semantic {
    pub id: String,
    pub kind: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub reference: String,
    /// `none`, `vendor-trusted: <paths>`, or `;`-separated repo-relative paths.
    #[serde(default)]
    pub correctness_gate: String,
    /// `production` when the gate exercises the production geometry.
    #[serde(default)]
    pub gate_scope: String,
    /// Which production shape/path the gate does not cover.
    #[serde(default)]
    pub gate_gap: String,
    /// `manual: <who runs it and how>` for operator-run gates.
    #[serde(default)]
    pub gate_invoke: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Implementation {
    pub id: String,
    pub semantic: String,
    pub provider: String,
    pub source: String,
    pub legality: String,
    pub fallback: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub numeric_tolerance_mode: Option<String>,
    pub numeric_abs_tolerance: Option<f64>,
    pub numeric_rel_tolerance: Option<f64>,
    pub fallback_min_m: Option<i64>,
    pub fallback_status: Option<String>,
    pub router: Option<String>,
}

impl Registry {
    pub fn parse(text: &str) -> Result<Self> {
        toml::from_str(text).context("registry does not match the schema")
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read registry {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parse registry {}", path.display()))
    }
}

impl Semantic {
    /// Repo-relative files named by `correctness_gate`. `none` names nothing;
    /// `vendor-trusted: a; b` names `a` and `b`.
    pub fn gate_paths(&self) -> Vec<&str> {
        let gate = self.correctness_gate.trim();
        if gate == "none" {
            return Vec::new();
        }
        let gate = gate.strip_prefix("vendor-trusted:").map_or(gate, str::trim);
        gate.split(';')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect()
    }

    fn is_concrete_gate(&self) -> bool {
        let gate = self.correctness_gate.trim();
        gate != "none" && !gate.starts_with("vendor-trusted:")
    }
}

/// Validate the gate rules; returns one message per violation (empty = pass).
///
/// - every semantic names a `correctness_gate`;
/// - every path it names exists under `root`;
/// - a concrete gate (not `none` / `vendor-trusted:`) states its coverage with
///   exactly one of `gate_scope = "production"` or a non-empty `gate_gap`;
/// - a `gate_invoke` uses the form `manual: <who runs it and how>`.
pub fn check(registry: &Registry, root: &Path, label: &str) -> Vec<String> {
    let mut errors = Vec::new();
    for entry in &registry.semantic {
        let sid = &entry.id;
        if entry.correctness_gate.trim().is_empty() {
            errors.push(format!("{label}: semantic {sid} has no correctness_gate"));
            continue;
        }
        let paths = entry.gate_paths();
        for part in &paths {
            if !root.join(part).exists() {
                errors.push(format!(
                    "{label}: semantic {sid} gate path does not exist: {part}"
                ));
            }
        }
        if !paths.is_empty() && entry.is_concrete_gate() {
            let production = entry.gate_scope.trim() == "production";
            let gap = !entry.gate_gap.trim().is_empty();
            if !production && !gap {
                errors.push(format!(
                    "{label}: semantic {sid} has a concrete gate path but neither \
                     gate_scope=\"production\" nor a non-empty gate_gap — state coverage explicitly"
                ));
            } else if production && gap {
                errors.push(format!(
                    "{label}: semantic {sid} sets both gate_scope=production and gate_gap; pick one"
                ));
            }
        }
        let invoke = entry.gate_invoke.trim();
        if !invoke.is_empty() && !invoke.starts_with("manual:") {
            let head: String = invoke.chars().take(40).collect();
            errors.push(format!(
                "{label}: semantic {sid} gate_invoke must use the form \
                 \"manual: <who runs it and how>\", got {head:?}"
            ));
        }
    }
    errors
}

/// One standalone parity example the GPU batch builds and runs.
#[derive(Debug, PartialEq, Eq)]
pub struct ParityGate {
    /// Example (and binary) name: the file stem.
    pub name: String,
    /// The example header (first 60 lines) names SM90.
    pub sm90: bool,
    /// The example reads `INFER_DSV4_MODEL_PATH` (multi-rank model gate).
    pub model: bool,
}

impl ParityGate {
    /// Comma list from `{sm90, model}`, in that order.
    pub fn flags(&self) -> String {
        let mut flags = Vec::new();
        if self.sm90 {
            flags.push("sm90");
        }
        if self.model {
            flags.push("model");
        }
        flags.join(",")
    }
}

/// Every `<prefix><name>.rs` occurrence in `text` (`name` = `[A-Za-z0-9_]+`).
fn example_paths<'a>(text: &'a str, prefix: &str) -> Vec<&'a str> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(offset) = text[from..].find(prefix) {
        let start = from + offset;
        let stem_start = start + prefix.len();
        let stem_len = text[stem_start..]
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        let end = stem_start + stem_len;
        if stem_len > 0 && text[end..].starts_with(".rs") {
            found.push(&text[start..end + 3]);
            from = end + 3;
        } else {
            from = start + 1;
        }
    }
    found
}

/// Parity examples named by any `correctness_gate`, sorted by path (bytewise)
/// and deduplicated. A named example that does not exist is an error: skipping
/// it would shorten the batch with no row and no error.
pub fn parity_gates(registry: &Registry, root: &Path, prefix: &str) -> Result<Vec<ParityGate>> {
    let mut paths: Vec<&str> = registry
        .semantic
        .iter()
        .flat_map(|entry| example_paths(&entry.correctness_gate, prefix))
        .collect();
    paths.sort_unstable();
    paths.dedup();

    let missing: Vec<&str> = paths
        .iter()
        .copied()
        .filter(|rel| !root.join(rel).is_file())
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "registry names correctness_gate examples that do not exist: {}",
        missing.join(", ")
    );

    paths
        .into_iter()
        .map(|rel| {
            let src = root.join(rel);
            let bytes = std::fs::read(&src).with_context(|| format!("read {}", src.display()))?;
            let text = String::from_utf8_lossy(&bytes);
            let header = text
                .split('\n')
                .take(60)
                .collect::<Vec<_>>()
                .join("\n")
                .to_ascii_lowercase();
            let name = rel
                .rsplit('/')
                .next()
                .and_then(|file| file.strip_suffix(".rs"))
                .unwrap_or(rel)
                .to_string();
            Ok(ParityGate {
                name,
                sm90: header.contains("sm90") || header.contains("sm_90"),
                model: text.contains("INFER_DSV4_MODEL_PATH"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn real_registry_passes_check() {
        let root = repo_root();
        let registry = Registry::load(&root.join("operators/registry.toml")).unwrap();
        assert!(!registry.semantic.is_empty());
        let errors = check(&registry, &root, "operators/registry.toml");
        assert!(errors.is_empty(), "{errors:#?}");
        let gates = parity_gates(&registry, &root, DEFAULT_GATE_PREFIX).unwrap();
        assert!(!gates.is_empty());
    }

    #[test]
    fn nonexistent_gate_path_fails_check() {
        let registry = Registry::parse(
            r#"
[[semantic]]
id = "demo.op"
kind = "primitive"
inputs = ["x"]
outputs = ["y"]
reference = "cuda.demo.op"
correctness_gate = "crates/infer-cuda/examples/no_such_gate_parity.rs"
gate_scope = "production"
"#,
        )
        .unwrap();
        let root = repo_root();
        let errors = check(&registry, &root, "registry.toml");
        assert_eq!(
            errors,
            ["registry.toml: semantic demo.op gate path does not exist: \
                 crates/infer-cuda/examples/no_such_gate_parity.rs"]
        );
        let err = parity_gates(&registry, &root, DEFAULT_GATE_PREFIX).unwrap_err();
        assert!(err.to_string().contains("no_such_gate_parity.rs"), "{err}");
    }
}
