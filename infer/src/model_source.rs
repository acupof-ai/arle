use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::{
    gguf::{GgufFile, try_open as try_open_gguf},
    hf_hub,
    tokenizer::Tokenizer,
};

pub struct ResolvedModelSource {
    resolved_path: PathBuf,
    model_root: PathBuf,
    gguf: Option<GgufFile>,
    runtime_assets_dir: Option<PathBuf>,
}

impl ResolvedModelSource {
    pub fn resolve(model_source: &str) -> Result<Self> {
        let resolved_path = hf_hub::resolve_model_path(model_source)
            .with_context(|| format!("failed to resolve model '{model_source}'"))?;
        let model_root = model_root_from_path(&resolved_path);
        let gguf = try_open_gguf(resolved_path.to_str().unwrap_or(model_source));
        let runtime_assets_dir = gguf
            .as_ref()
            .map(|gguf| resolve_gguf_runtime_assets_dir(&model_root, gguf))
            .transpose()?
            .flatten();

        Ok(Self {
            resolved_path,
            model_root,
            gguf,
            runtime_assets_dir,
        })
    }

    pub fn resolved_path(&self) -> &Path {
        &self.resolved_path
    }

    pub fn model_root(&self) -> &Path {
        &self.model_root
    }

    pub fn gguf(&self) -> Option<&GgufFile> {
        self.gguf.as_ref()
    }

    pub fn runtime_assets_dir(&self) -> Option<&Path> {
        self.runtime_assets_dir.as_deref()
    }

    pub fn config_dir(&self) -> Option<&Path> {
        self.preferred_dir_with("config.json")
    }

    pub fn tokenizer_dir(&self) -> Option<&Path> {
        self.preferred_dir_with("tokenizer.json")
    }

    pub fn load_tokenizer(&self) -> Result<Tokenizer> {
        if let Some(dir) = self.tokenizer_dir() {
            return Tokenizer::from_file(dir.to_str().unwrap_or("."))
                .with_context(|| format!("failed to load tokenizer from {}", dir.display()));
        }

        if let Some(gguf) = self.gguf()
            && let Some(json_str) = gguf.extract_tokenizer_json()
        {
            let tok_path = self.model_root.join("_gguf_tokenizer.json");
            std::fs::write(&tok_path, json_str).with_context(|| {
                format!(
                    "failed to write extracted tokenizer to {}",
                    tok_path.display()
                )
            })?;
            return Tokenizer::from_file(tok_path.to_str().unwrap_or("."))
                .with_context(|| format!("failed to load tokenizer from {}", tok_path.display()));
        }

        anyhow::bail!(
            "model is missing tokenizer.json and does not embed tokenizer.huggingface.json"
        )
    }

    fn preferred_dir_with(&self, filename: &str) -> Option<&Path> {
        self.runtime_assets_dir
            .as_deref()
            .filter(|dir| dir.join(filename).exists())
            .or_else(|| {
                self.model_root
                    .join(filename)
                    .exists()
                    .then_some(self.model_root.as_path())
            })
    }
}

fn model_root_from_path(path: &Path) -> PathBuf {
    if path.is_file() {
        path.parent().unwrap_or(Path::new(".")).to_path_buf()
    } else {
        path.to_path_buf()
    }
}

fn gguf_base_model_repo_id(gguf: &GgufFile) -> Option<String> {
    gguf.meta_str("general.base_model.0.repo_url")
        .and_then(|url| url.strip_prefix("https://huggingface.co/"))
        .map(|repo_id| repo_id.trim_end_matches('/').to_string())
}

fn resolve_gguf_runtime_assets_dir(model_root: &Path, gguf: &GgufFile) -> Result<Option<PathBuf>> {
    let has_local_config = model_root.join("config.json").exists();
    let has_local_tokenizer = model_root.join("tokenizer.json").exists();
    if has_local_config && has_local_tokenizer {
        return Ok(Some(model_root.to_path_buf()));
    }

    if let Some(repo_id) = gguf_base_model_repo_id(gguf) {
        let assets_dir = hf_hub::download_runtime_assets_from_hub(&repo_id).with_context(|| {
            format!("failed to resolve runtime assets for GGUF base model '{repo_id}'")
        })?;
        return Ok(Some(assets_dir));
    }

    if has_local_config {
        return Ok(Some(model_root.to_path_buf()));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::ResolvedModelSource;
    use tempfile::tempdir;

    #[test]
    fn prefers_runtime_assets_over_partial_local_gguf_sidecars() {
        let root = tempdir().unwrap();
        let runtime = tempdir().unwrap();
        std::fs::write(root.path().join("config.json"), "{}").unwrap();
        std::fs::write(runtime.path().join("config.json"), "{}").unwrap();
        std::fs::write(runtime.path().join("tokenizer.json"), "{}").unwrap();

        let source = ResolvedModelSource {
            resolved_path: root.path().to_path_buf(),
            model_root: root.path().to_path_buf(),
            gguf: None,
            runtime_assets_dir: Some(runtime.path().to_path_buf()),
        };

        assert_eq!(source.config_dir(), Some(runtime.path()));
        assert_eq!(source.tokenizer_dir(), Some(runtime.path()));
    }

    #[test]
    fn falls_back_to_local_runtime_files_when_no_sidecar_dir_exists() {
        let root = tempdir().unwrap();
        std::fs::write(root.path().join("config.json"), "{}").unwrap();
        std::fs::write(root.path().join("tokenizer.json"), "{}").unwrap();

        let source = ResolvedModelSource {
            resolved_path: root.path().to_path_buf(),
            model_root: root.path().to_path_buf(),
            gguf: None,
            runtime_assets_dir: None,
        };

        assert_eq!(source.config_dir(), Some(root.path()));
        assert_eq!(source.tokenizer_dir(), Some(root.path()));
    }
}
