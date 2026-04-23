//! HuggingFace Hub model search with nucleo-powered fuzzy matching.
//!
//! Searches both official and community repos (mlx-community, TheBloke, etc.)
//! with a 5-second timeout fallback.

use anyhow::Result;
use nucleo_matcher::pattern::{Atom, AtomKind, CaseMatching, Normalization};
use nucleo_matcher::{Config, Matcher, Utf32Str};
use serde::Deserialize;

const HF_API_BASE: &str = "https://huggingface.co/api/models";
const SEARCH_LIMIT: usize = 30;
const TIMEOUT_SECS: u64 = 5;

/// A model returned from the HuggingFace API.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct HfSearchResult {
    #[serde(rename = "modelId")]
    pub(crate) model_id: String,
    #[serde(default)]
    pub(crate) downloads: u64,
    #[serde(default)]
    pub(crate) likes: u64,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) tags: Vec<String>,
}

impl HfSearchResult {
    /// Format for display in the picker.
    pub(crate) fn display_line(&self) -> String {
        let dl = format_count(self.downloads);
        let lk = format_count(self.likes);
        format!("{}  (dl:{dl}  lk:{lk})", self.model_id)
    }
}

/// Search HuggingFace for text-generation models matching the query.
///
/// Returns up to `SEARCH_LIMIT` results sorted by downloads. Falls back
/// gracefully on network errors.
pub(crate) fn search_hf_models(query: &str) -> Result<Vec<HfSearchResult>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
        .build()?;

    let url = format!(
        "{HF_API_BASE}?search={}&filter=text-generation&sort=downloads&direction=-1&limit={SEARCH_LIMIT}",
        urlenccode(query)
    );

    let response = client.get(&url).send()?;
    let results: Vec<HfSearchResult> = response.json()?;
    Ok(results)
}

/// Fuzzy-filter a list of search results using nucleo-matcher.
///
/// Returns results sorted by match score (best first). Items that don't
/// match at all are excluded.
#[allow(dead_code)]
pub(crate) fn fuzzy_filter(
    results: &[HfSearchResult],
    pattern: &str,
) -> Vec<(u16, HfSearchResult)> {
    if pattern.is_empty() {
        return results
            .iter()
            .enumerate()
            .map(|(i, r)| (i as u16, r.clone()))
            .collect();
    }

    let mut matcher = Matcher::new(Config::DEFAULT);
    let atom = Atom::new(
        pattern,
        CaseMatching::Ignore,
        Normalization::Smart,
        AtomKind::Fuzzy,
        false,
    );

    let mut scored: Vec<(u16, HfSearchResult)> = results
        .iter()
        .filter_map(|r| {
            let mut buf = Vec::new();
            let haystack = Utf32Str::new(&r.model_id, &mut buf);
            let score = atom.score(haystack, &mut matcher)?;
            Some((score, r.clone()))
        })
        .collect();

    scored.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    scored
}

fn urlenccode(s: &str) -> String {
    s.replace(' ', "+").replace('/', "%2F").replace(':', "%3A")
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_filter_empty_pattern_returns_all() {
        let results = vec![
            HfSearchResult {
                model_id: "Qwen/Qwen3-4B".to_string(),
                downloads: 1000,
                likes: 50,
                tags: vec![],
            },
            HfSearchResult {
                model_id: "meta-llama/Llama-3-8B".to_string(),
                downloads: 2000,
                likes: 100,
                tags: vec![],
            },
        ];
        let filtered = fuzzy_filter(&results, "");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn fuzzy_filter_narrows_results() {
        let results = vec![
            HfSearchResult {
                model_id: "Qwen/Qwen3-4B".to_string(),
                downloads: 1000,
                likes: 50,
                tags: vec![],
            },
            HfSearchResult {
                model_id: "meta-llama/Llama-3-8B".to_string(),
                downloads: 2000,
                likes: 100,
                tags: vec![],
            },
        ];
        let filtered = fuzzy_filter(&results, "qwen");
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].1.model_id.contains("Qwen"));
    }

    #[test]
    fn format_count_scales() {
        assert_eq!(format_count(500), "500");
        assert_eq!(format_count(1500), "1.5K");
        assert_eq!(format_count(2_500_000), "2.5M");
    }

    #[test]
    fn display_line_non_empty() {
        let r = HfSearchResult {
            model_id: "Qwen/Qwen3-4B".to_string(),
            downloads: 1000,
            likes: 50,
            tags: vec![],
        };
        let line = r.display_line();
        assert!(line.contains("Qwen/Qwen3-4B"));
    }
}
