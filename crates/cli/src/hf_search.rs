//! HuggingFace Hub model search.
//!
//! Searches both official and community repos (mlx-community, TheBloke, etc.)
//! with a 5-second timeout fallback.

use anyhow::Result;
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
