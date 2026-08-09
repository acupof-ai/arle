//! Trainer checkpoint helpers.
//!
//! Live trainer state uses the v2 directory layout:
//! - `trainer_state.json`
//! - `optimizer.safetensors`

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::Path;

use autograd::TensorId;
use autograd::adamw_state::{AdamWParamState, AdamWState};
use safetensors::{Dtype, SafeTensors, serialize_to_file};
use serde::{Deserialize, Serialize};

pub const TRAINER_STATE_CODEC_VERSION: u32 = 2;
pub const TRAINER_STATE_FILENAME: &str = "trainer_state.json";
pub const OPTIMIZER_STATE_FILENAME: &str = "optimizer.safetensors";

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("bad magic: expected {expected:?}, got {actual:?}")]
    BadMagic { expected: [u8; 8], actual: [u8; 8] },
    #[error("parameter count mismatch: file has {file}, model has {model}")]
    ParamCount { file: usize, model: usize },
    #[error("shape mismatch at param {index}: file {file:?}, model {model:?}")]
    ShapeMismatch {
        index: usize,
        file: Vec<usize>,
        model: Vec<usize>,
    },
    #[error("missing tensor id {0}")]
    MissingTensor(TensorId),
    #[error("trainer state json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing optim tensor pair for param '{0}' (need both .m and .v)")]
    MissingMomentPair(String),
    #[error("trainer state v2 codec version mismatch: expected 2, got {0}")]
    VersionMismatch(u32),
    #[error("safetensors: {0}")]
    Safetensors(String),
}

pub type Result<T> = std::result::Result<T, CheckpointError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerStateDoc {
    pub step: u64,
    pub optim_schema: String,
    pub schedule_name: String,
    pub schedule_params: serde_json::Value,
    pub grad_accum_current: u64,
    pub rng_seed: u64,
    pub codec_version: u32,
}

/// Save trainer state + AdamW moments to `<dir>/trainer_state.json` and
/// `<dir>/optimizer.safetensors`.
pub fn save_trainer_state_v2(
    dir: &Path,
    state: &TrainerStateDoc,
    optim: &AdamWState,
) -> std::result::Result<(), CheckpointError> {
    std::fs::create_dir_all(dir)?;

    let json = serde_json::to_string_pretty(state)?;
    std::fs::write(dir.join(TRAINER_STATE_FILENAME), json)?;

    let mut tensors: Vec<(String, OptimTensorView)> = Vec::with_capacity(optim.params.len() * 2);
    for param in &optim.params {
        let expected_len: usize = if param.shape.is_empty() {
            1
        } else {
            param.shape.iter().product()
        };
        if param.m.len() != expected_len || param.v.len() != expected_len {
            return Err(CheckpointError::Safetensors(format!(
                "AdamW moment length mismatch for '{}' during save: shape {:?} => {} elems, m {} v {}",
                param.name,
                param.shape,
                expected_len,
                param.m.len(),
                param.v.len(),
            )));
        }
        tensors.push((
            format!("{}.m", param.name),
            OptimTensorView::from_f32(param.shape.clone(), &param.m),
        ));
        tensors.push((
            format!("{}.v", param.name),
            OptimTensorView::from_f32(param.shape.clone(), &param.v),
        ));
    }

    let mut metadata: HashMap<String, String> = HashMap::new();
    metadata.insert("step".to_string(), optim.step.to_string());
    metadata.insert(
        "skipped_export".to_string(),
        optim.skipped_export.to_string(),
    );

    serialize_to_file(tensors, Some(metadata), &dir.join(OPTIMIZER_STATE_FILENAME))
        .map_err(|err| CheckpointError::Safetensors(err.to_string()))?;

    Ok(())
}

pub fn load_trainer_state_v2(
    dir: &Path,
) -> std::result::Result<(TrainerStateDoc, AdamWState), CheckpointError> {
    let json_path = dir.join(TRAINER_STATE_FILENAME);
    let json_bytes = std::fs::read(&json_path)?;
    let state: TrainerStateDoc = serde_json::from_slice(&json_bytes)?;
    if state.codec_version != TRAINER_STATE_CODEC_VERSION {
        return Err(CheckpointError::VersionMismatch(state.codec_version));
    }

    let optim_path = dir.join(OPTIMIZER_STATE_FILENAME);
    let optim_bytes = std::fs::read(&optim_path)?;
    let (_, header_metadata) = SafeTensors::read_metadata(&optim_bytes)
        .map_err(|err| CheckpointError::Safetensors(err.to_string()))?;
    let st = SafeTensors::deserialize(&optim_bytes)
        .map_err(|err| CheckpointError::Safetensors(err.to_string()))?;

    let mut order: Vec<String> = Vec::new();
    let mut groups: BTreeMap<String, MomentPair> = BTreeMap::new();
    for (key, view) in st.tensors() {
        let (base, which) = split_moment_key(&key).ok_or_else(|| {
            CheckpointError::Safetensors(format!(
                "optimizer.safetensors: tensor '{key}' has no .m/.v suffix"
            ))
        })?;
        let entry = groups.entry(base.to_string()).or_insert_with(|| {
            order.push(base.to_string());
            MomentPair::default()
        });
        let shape = view.shape().to_vec();
        let data = optim_tensor_view_to_f32(&view)
            .map_err(|err| CheckpointError::Safetensors(err.to_string()))?;
        match which {
            Moment::M => entry.m = Some((shape, data)),
            Moment::V => entry.v = Some((shape, data)),
        }
    }

    let params = order
        .iter()
        .map(|name| {
            let pair = groups.get(name).expect("just inserted");
            let (m_shape, m_data) = pair
                .m
                .clone()
                .ok_or_else(|| CheckpointError::MissingMomentPair(name.clone()))?;
            let (v_shape, v_data) = pair
                .v
                .clone()
                .ok_or_else(|| CheckpointError::MissingMomentPair(name.clone()))?;
            if m_shape != v_shape {
                return Err(CheckpointError::Safetensors(format!(
                    "optimizer.safetensors: '{name}' .m shape {m_shape:?} != .v shape {v_shape:?}"
                )));
            }
            Ok(AdamWParamState {
                name: name.clone(),
                m: m_data,
                v: v_data,
                shape: m_shape,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let metadata: &Option<HashMap<String, String>> = header_metadata.metadata();
    let step = metadata
        .as_ref()
        .and_then(|m: &HashMap<String, String>| m.get("step"))
        .and_then(|s: &String| s.parse::<u64>().ok())
        .unwrap_or(state.step);
    let skipped_export = metadata
        .as_ref()
        .and_then(|m: &HashMap<String, String>| m.get("skipped_export"))
        .and_then(|s: &String| s.parse::<usize>().ok())
        .unwrap_or(0);

    Ok((
        state,
        AdamWState {
            step,
            params,
            skipped_export,
        },
    ))
}

#[derive(Default)]
struct MomentPair {
    m: Option<(Vec<usize>, Vec<f32>)>,
    v: Option<(Vec<usize>, Vec<f32>)>,
}

enum Moment {
    M,
    V,
}

fn split_moment_key(key: &str) -> Option<(&str, Moment)> {
    if let Some(base) = key.strip_suffix(".m") {
        Some((base, Moment::M))
    } else {
        key.strip_suffix(".v").map(|base| (base, Moment::V))
    }
}

fn optim_tensor_view_to_f32(
    view: &safetensors::tensor::TensorView<'_>,
) -> std::result::Result<Vec<f32>, String> {
    match view.dtype() {
        Dtype::F32 => Ok(view
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        dtype => Err(format!("optimizer.safetensors: unsupported dtype {dtype}")),
    }
}

struct OptimTensorView {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl OptimTensorView {
    fn from_f32(shape: Vec<usize>, values: &[f32]) -> Self {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self { shape, bytes }
    }
}

impl safetensors::View for OptimTensorView {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.bytes.as_slice())
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

/// Atomically refresh a `latest` symlink inside `parent` pointing at
/// `target_basename`. The symlink is relative (basename only) so the tree
/// stays rsync-safe. Refuses to overwrite a non-symlink at `latest`.
/// Atomic via `.latest.tmp` + `rename` (no missing-`latest` window).
#[cfg(unix)]
pub fn write_latest_symlink(parent: &Path, target_basename: &str) -> io::Result<()> {
    use std::os::unix::fs::symlink;

    if target_basename.contains('/') || target_basename.contains('\\') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("write_latest_symlink: target must be a basename, got {target_basename:?}"),
        ));
    }

    let link = parent.join("latest");
    match std::fs::symlink_metadata(&link) {
        Ok(meta) if meta.file_type().is_symlink() => {}
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "write_latest_symlink: refusing to overwrite non-symlink at {}",
                    link.display()
                ),
            ));
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }

    let tmp = parent.join(".latest.tmp");
    match std::fs::symlink_metadata(&tmp) {
        Ok(_) => std::fs::remove_file(&tmp)?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }

    symlink(target_basename, &tmp)?;
    if let Err(err) = std::fs::rename(&tmp, &link) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn write_latest_symlink(_parent: &Path, _target_basename: &str) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod latest_symlink_tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[cfg(unix)]
    #[test]
    fn writes_latest_symlink_when_absent() {
        let dir = tempdir().expect("tempdir");
        let step_dir = dir.path().join("step_000001");
        fs::create_dir_all(&step_dir).unwrap();
        fs::write(step_dir.join("config.json"), "{}").unwrap();

        write_latest_symlink(dir.path(), "step_000001").expect("write latest");

        let link = dir.path().join("latest");
        let meta = fs::symlink_metadata(&link).expect("latest exists");
        assert!(meta.file_type().is_symlink(), "latest must be a symlink");
        let resolved = fs::canonicalize(&link).expect("resolve latest");
        let expected = fs::canonicalize(&step_dir).expect("resolve step");
        assert_eq!(resolved, expected, "latest must point at step_000001");
        assert!(
            link.join("config.json").exists(),
            "config.json must resolve through the symlink"
        );
    }

    #[cfg(unix)]
    #[test]
    fn refreshes_latest_symlink_to_new_target() {
        let dir = tempdir().expect("tempdir");
        let step1 = dir.path().join("step_000001");
        let step2 = dir.path().join("step_000002");
        fs::create_dir_all(&step1).unwrap();
        fs::create_dir_all(&step2).unwrap();
        fs::write(step1.join("marker.txt"), "one").unwrap();
        fs::write(step2.join("marker.txt"), "two").unwrap();

        write_latest_symlink(dir.path(), "step_000001").unwrap();
        write_latest_symlink(dir.path(), "step_000002").unwrap();

        let link = dir.path().join("latest");
        let marker = fs::read_to_string(link.join("marker.txt")).expect("read marker");
        assert_eq!(marker, "two", "latest must now point at step_000002");
    }

    #[cfg(unix)]
    #[test]
    fn refuses_to_overwrite_regular_file_at_latest() {
        let dir = tempdir().expect("tempdir");
        fs::write(dir.path().join("latest"), "user-data").expect("pre-existing file");

        let err = write_latest_symlink(dir.path(), "step_000001")
            .expect_err("must refuse to clobber regular file");
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);

        let surviving = fs::read_to_string(dir.path().join("latest")).unwrap();
        assert_eq!(surviving, "user-data");
    }

    #[test]
    fn rejects_basename_with_path_separator() {
        let dir = tempdir().expect("tempdir");
        let err = write_latest_symlink(dir.path(), "../etc/passwd")
            .expect_err("must reject path-like basename");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
