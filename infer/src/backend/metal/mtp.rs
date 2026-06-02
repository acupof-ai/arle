use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

use crate::gguf::GgufFile;

const SAFETENSORS_HEADER_LIMIT_BYTES: u64 = 64 * 1024 * 1024;
const MTP_EXAMPLE_LIMIT: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalMtpMode {
    Auto,
    Explicit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalMtpOptions {
    pub mode: MetalMtpMode,
}

impl MetalMtpOptions {
    pub const fn auto() -> Self {
        Self {
            mode: MetalMtpMode::Auto,
        }
    }

    pub const fn explicit() -> Self {
        Self {
            mode: MetalMtpMode::Explicit,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalMtpTensorSource {
    SafetensorsIndex(PathBuf),
    SafetensorsHeader,
    Gguf,
    NotFound,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalMtpProbe {
    pub tensor_count: usize,
    pub examples: Vec<String>,
    pub source: MetalMtpTensorSource,
}

impl MetalMtpProbe {
    pub fn has_tensors(&self) -> bool {
        self.tensor_count > 0
    }

    pub fn examples_label(&self) -> String {
        if self.examples.is_empty() {
            "none".to_string()
        } else {
            self.examples.join(", ")
        }
    }
}

pub(super) fn probe_mtp_tensors(
    model_root: &Path,
    gguf: Option<&GgufFile>,
) -> Result<MetalMtpProbe> {
    if let Some(gguf) = gguf {
        return Ok(probe_gguf(gguf));
    }

    if let Some(probe) = probe_safetensors_index(model_root)? {
        return Ok(probe);
    }

    let mut all_matches = Vec::new();
    for shard in collect_safetensors_files(model_root)? {
        all_matches.extend(probe_safetensors_header(&shard)?);
    }
    Ok(probe_from_matches(
        all_matches,
        MetalMtpTensorSource::SafetensorsHeader,
    ))
}

fn probe_gguf(gguf: &GgufFile) -> MetalMtpProbe {
    let mut matches = Vec::new();
    for (key, value) in &gguf.metadata {
        if is_mtp_metadata_key(key)
            && value
                .as_u32()
                .or_else(|| value.as_str().and_then(|raw| raw.parse::<u32>().ok()))
                .is_some_and(|n| n > 0)
        {
            matches.push(format!("{key}={}", value_label(key, value)));
        }
    }
    matches.extend(
        gguf.tensors
            .keys()
            .filter(|name| is_mtp_tensor_name(name))
            .cloned(),
    );
    probe_from_matches(matches, MetalMtpTensorSource::Gguf)
}

fn probe_safetensors_index(model_root: &Path) -> Result<Option<MetalMtpProbe>> {
    let index_path = model_root.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&index_path)
        .with_context(|| format!("reading {}", index_path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", index_path.display()))?;
    let Some(weight_map) = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(Some(MetalMtpProbe {
            tensor_count: 0,
            examples: Vec::new(),
            source: MetalMtpTensorSource::SafetensorsIndex(index_path),
        }));
    };
    let matches = weight_map
        .keys()
        .filter(|name| is_mtp_tensor_name(name))
        .cloned()
        .collect::<Vec<_>>();
    Ok(Some(probe_from_matches(
        matches,
        MetalMtpTensorSource::SafetensorsIndex(index_path),
    )))
}

fn probe_safetensors_header(path: &Path) -> Result<Vec<String>> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .with_context(|| format!("reading safetensors header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    anyhow::ensure!(
        header_len <= SAFETENSORS_HEADER_LIMIT_BYTES,
        "safetensors header in {} is too large: {} bytes",
        path.display(),
        header_len
    );
    file.seek(SeekFrom::Start(8))
        .with_context(|| format!("seeking safetensors header in {}", path.display()))?;
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("reading safetensors header from {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("parsing safetensors header from {}", path.display()))?;
    let Some(root) = value.as_object() else {
        return Ok(Vec::new());
    };
    Ok(root
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .filter(|name| is_mtp_tensor_name(name))
        .cloned()
        .collect())
}

fn collect_safetensors_files(model_root: &Path) -> Result<Vec<PathBuf>> {
    if !model_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut files = std::fs::read_dir(model_root)
        .with_context(|| format!("reading {}", model_root.display()))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext == "safetensors")
        })
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn probe_from_matches(mut matches: Vec<String>, source: MetalMtpTensorSource) -> MetalMtpProbe {
    matches.sort();
    matches.dedup();
    let tensor_count = matches.len();
    matches.truncate(MTP_EXAMPLE_LIMIT);
    MetalMtpProbe {
        tensor_count,
        examples: matches,
        source: if tensor_count == 0 {
            match source {
                MetalMtpTensorSource::SafetensorsIndex(path) => {
                    MetalMtpTensorSource::SafetensorsIndex(path)
                }
                _ => MetalMtpTensorSource::NotFound,
            }
        } else {
            source
        },
    }
}

fn is_mtp_tensor_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("mtp.")
        || lower.contains(".mtp.")
        || lower.contains("nextn")
        || lower.contains("next_n")
        || lower.contains(".eh_proj.")
        || lower.contains(".enorm.")
        || lower.contains(".hnorm.")
}

fn is_mtp_metadata_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    lower.ends_with(".nextn_predict_layers")
        || lower.contains("nextn_predict_layers")
        || lower.contains("next_n_predict_layers")
        || lower.contains("mtp_num_hidden_layers")
}

fn value_label(key: &str, value: &crate::gguf::GgufValue) -> String {
    value
        .as_u32()
        .map(|v| v.to_string())
        .or_else(|| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| format!("{key}:present"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_mtp_from_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embed_tokens.weight":"a.safetensors","mtp.fc.weight":"b.safetensors","mtp.layers.0.self_attn.q_proj.weight":"b.safetensors","model.layers.0.nextn.weight":"b.safetensors"}}"#,
        )
        .expect("write index");

        let probe = probe_mtp_tensors(dir.path(), None).expect("probe");
        assert_eq!(probe.tensor_count, 3);
        assert_eq!(
            probe.examples,
            vec![
                "model.layers.0.nextn.weight".to_string(),
                "mtp.fc.weight".to_string(),
                "mtp.layers.0.self_attn.q_proj.weight".to_string(),
            ]
        );
        assert!(matches!(
            probe.source,
            MetalMtpTensorSource::SafetensorsIndex(_)
        ));
    }

    #[test]
    fn explicit_no_index_no_shards_is_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let probe = probe_mtp_tensors(dir.path(), None).expect("probe");
        assert_eq!(probe.tensor_count, 0);
        assert_eq!(probe.source, MetalMtpTensorSource::NotFound);
    }

    #[test]
    fn detects_mtp_from_gguf_metadata_and_tensor_names() {
        let dir = tempfile::tempdir().expect("tempdir");
        let gguf_path = dir.path().join("mtp.gguf");
        write_minimal_gguf(
            &gguf_path,
            &[("qwen35.nextn_predict_layers", 2)],
            &[
                "blk.48.nextn.eh_proj.weight",
                "blk.48.nextn.enorm.weight",
                "blk.48.attn_q.weight",
            ],
        );
        let gguf = GgufFile::open(gguf_path.to_str().expect("path")).expect("gguf");

        let probe = probe_mtp_tensors(dir.path(), Some(&gguf)).expect("probe");
        assert_eq!(probe.tensor_count, 3);
        assert_eq!(
            probe.examples,
            vec![
                "blk.48.nextn.eh_proj.weight".to_string(),
                "blk.48.nextn.enorm.weight".to_string(),
                "qwen35.nextn_predict_layers=2".to_string(),
            ]
        );
        assert_eq!(probe.source, MetalMtpTensorSource::Gguf);
    }

    #[test]
    fn gguf_without_mtp_signals_is_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let gguf_path = dir.path().join("plain.gguf");
        write_minimal_gguf(&gguf_path, &[("qwen35.nextn_predict_layers", 0)], &[]);
        let gguf = GgufFile::open(gguf_path.to_str().expect("path")).expect("gguf");

        let probe = probe_mtp_tensors(dir.path(), Some(&gguf)).expect("probe");
        assert_eq!(probe.tensor_count, 0);
        assert_eq!(probe.source, MetalMtpTensorSource::NotFound);
    }

    fn write_minimal_gguf(path: &Path, metadata: &[(&str, u32)], tensors: &[&str]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            push_string(&mut bytes, key);
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for tensor in tensors {
            push_string(&mut bytes, tensor);
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write gguf");
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
