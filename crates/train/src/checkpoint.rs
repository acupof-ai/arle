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

/// Codec version for the v2 directory checkpoint layout.
pub const TRAINER_STATE_CODEC_VERSION: u32 = 2;

/// Filename for the trainer scalar/schedule JSON under a v2 checkpoint dir.
pub const TRAINER_STATE_FILENAME: &str = "trainer_state.json";

/// Filename for the AdamW moments safetensors file under a v2 checkpoint dir.
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

// ---------------------------------------------------------------------------
// Checkpoint Codec v2 — directory layout (trainer_state.json + optimizer.safetensors)
// ---------------------------------------------------------------------------

/// Trainer-side scalar/schedule state persisted alongside optimizer moments.
///
/// Serialized pretty to `<dir>/trainer_state.json`. Kept open-schema via
/// `schedule_params: serde_json::Value` so schedule implementations can own
/// their own parameter shape without a matching enum in this crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerStateDoc {
    pub step: u64,
    /// Schema tag for the optimizer state. Today: `"adamw-v1"`.
    pub optim_schema: String,
    /// `"constant"` | `"linear-warmup"` | `"cosine-with-warmup"`.
    pub schedule_name: String,
    pub schedule_params: serde_json::Value,
    pub grad_accum_current: u64,
    pub rng_seed: u64,
    /// On-disk codec version. Must equal [`TRAINER_STATE_CODEC_VERSION`].
    pub codec_version: u32,
}

/// Save a v2 directory checkpoint (scalar trainer state + AdamW moments).
///
/// Writes:
/// - `<dir>/trainer_state.json` (pretty-printed JSON)
/// - `<dir>/optimizer.safetensors` (each param → two tensors `"{name}.m"`,
///   `"{name}.v"`, f32; top-level metadata carries `step` + `skipped_export`)
///
/// Creates `dir` (and any missing parents) on demand.
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

/// Load a v2 directory checkpoint written by [`save_trainer_state_v2`].
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
    // `SafeTensors::deserialize` gives us tensor views; the top-level
    // metadata lives on `Metadata` and has to be fetched via `read_metadata`.
    let (_, header_metadata) = SafeTensors::read_metadata(&optim_bytes)
        .map_err(|err| CheckpointError::Safetensors(err.to_string()))?;
    let st = SafeTensors::deserialize(&optim_bytes)
        .map_err(|err| CheckpointError::Safetensors(err.to_string()))?;

    // Group by base name (strip .m/.v suffix), preserving first-seen order so
    // saved-order round-trips deterministically.
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

    let mut params = Vec::with_capacity(order.len());
    for name in &order {
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
        params.push(AdamWParamState {
            name: name.clone(),
            m: m_data,
            v: v_data,
            shape: m_shape,
        });
    }

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

/// Thin `safetensors::View` impl that borrows moment data as little-endian f32
/// bytes without a heap-allocated detour.
struct OptimTensorView {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl OptimTensorView {
    fn from_f32(shape: Vec<usize>, values: &[f32]) -> Self {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
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

/// Atomically refresh a `latest` symlink inside `parent` pointing at a
/// sibling directory named `target_basename`. Used by every training
/// binary's save-checkpoint path so downstream tooling (e.g.
/// `infer --model-path <out>/latest`) can reference "the most recent
/// checkpoint" without knowing the step number. Docs/plan:
/// `docs/plans/train-eval-infer-dx-v1.md` Phase DX-1.
///
/// `target_basename` must be a basename (e.g. `"step_000100"`), not a
/// full path — the symlink is relative so the whole `<parent>/` tree
/// remains copyable / rsync-safe.
///
/// Refuses to overwrite a regular file or directory at `<parent>/latest`
/// (only an existing symlink is replaced). This guards against a user
/// accidentally stashing their final checkpoint under that exact name.
///
/// The update is atomic from a reader's perspective: we create the new
/// symlink under `.latest.tmp` first, then `rename()` it onto `latest`.
/// POSIX `rename` on the same directory is atomic, so readers either see
/// the old target or the new one — never a missing/half-applied pointer.
/// Codex review 2026-04-20 on 0da212f (Medium): the previous
/// remove-then-create sequence exposed a brief window where `latest`
/// did not exist. An `infer --model-path <out>/latest` call that landed
/// in that window would fail even though a checkpoint was ready.
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
    // Refuse to overwrite a non-symlink (file or directory) at `<parent>/latest`.
    // We only want to atomically swap an existing symlink or fill an empty slot.
    match std::fs::symlink_metadata(&link) {
        Ok(meta) if meta.file_type().is_symlink() => { /* swap via rename below */ }
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "write_latest_symlink: refusing to overwrite non-symlink at {}",
                    link.display()
                ),
            ));
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => { /* fresh install */ }
        Err(e) => return Err(e),
    }

    // Clean up any leftover tmp from a prior crashed call.
    let tmp = parent.join(".latest.tmp");
    match std::fs::symlink_metadata(&tmp) {
        Ok(_) => std::fs::remove_file(&tmp)?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }

    symlink(target_basename, &tmp)?;
    // POSIX rename is atomic on the same filesystem and will replace an
    // existing symlink at `link` without an intermediate "missing" state.
    if let Err(err) = std::fs::rename(&tmp, &link) {
        // Best-effort cleanup so we don't leave `.latest.tmp` behind.
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn write_latest_symlink(_parent: &Path, _target_basename: &str) -> io::Result<()> {
    // Non-unix targets (currently unsupported by this workspace per
    // CLAUDE.md support matrix) — no-op rather than erroring. Callers
    // who need a portable marker should fall back to reading the
    // lexicographically-largest `step_*` entry.
    Ok(())
}

/// Publish `<parent>/latest` pointing at `<parent>/<target_basename>`,
/// but **only after** `model.safetensors` has landed in the target dir.
///
/// This is the "publish-last" rule from
/// `docs/experience/wins/2026-04-20-phase-dx1-latest-symlink.md`:
/// `latest` is a contract that its target directory is complete. Per the
/// codex review on 0da212f (Medium #1), publishing before the final
/// artifact exposes readers to an incomplete step dir.
///
/// Returns an error if `<parent>/<target_basename>/model.safetensors`
/// does not exist yet. Both binary save hooks (`pretrain_qwen3::save_checkpoint`
/// and `train_sft::save_checkpoint_via_registry`) call this instead of
/// raw `write_latest_symlink` so a future refactor that drops the
/// ordering-dependent weight write first would fail a targeted unit
/// test (see `latest_symlink_tests::publish_after_weights_*`).
pub fn publish_latest_after_weights(parent: &Path, target_basename: &str) -> io::Result<()> {
    let weights = parent.join(target_basename).join("model.safetensors");
    if !weights.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "publish_latest_after_weights: {} missing — refusing to publish \
                 `latest` before the final artifact lands (publish-last contract)",
                weights.display()
            ),
        ));
    }
    write_latest_symlink(parent, target_basename)
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
        // Downstream tooling treats `<out>/latest` as a model dir; the
        // symlink must transparently expose the step dir's files.
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

        // User data must survive.
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

    // --- publish_latest_after_weights: publish-last contract ------------------
    // These pin the "no `latest` without model.safetensors" invariant so a
    // future refactor of the binary save hooks that moves the symlink above
    // the weight write — or drops the weight write — fails here instead of
    // silently exposing an incomplete step dir. Paired with the call-site
    // swaps in `pretrain_qwen3::save_checkpoint` and
    // `train_sft::save_checkpoint_via_registry` (codex review 2026-04-20 on
    // 8bde810, Low — restore end-to-end coverage the `latest`-refactor
    // dropped from `test_trainer_loop.rs:278`).

    #[cfg(unix)]
    #[test]
    fn publish_after_weights_writes_symlink_when_weights_present() {
        let dir = tempdir().expect("tempdir");
        let step_dir = dir.path().join("step_000001");
        fs::create_dir_all(&step_dir).unwrap();
        // Minimal staged checkpoint — `model.safetensors` is the gate.
        fs::write(step_dir.join("config.json"), "{}").unwrap();
        fs::write(step_dir.join("model.safetensors"), b"stub-bytes").unwrap();

        publish_latest_after_weights(dir.path(), "step_000001")
            .expect("publish should succeed with weights present");

        let link = dir.path().join("latest");
        let meta = fs::symlink_metadata(&link).expect("latest exists");
        assert!(meta.file_type().is_symlink(), "latest must be a symlink");
        let resolved = fs::canonicalize(&link).expect("resolve latest");
        let expected = fs::canonicalize(&step_dir).expect("resolve step");
        assert_eq!(resolved, expected, "latest must point at step_000001");
    }

    #[test]
    fn publish_after_weights_refuses_when_weights_missing() {
        let dir = tempdir().expect("tempdir");
        let step_dir = dir.path().join("step_000001");
        fs::create_dir_all(&step_dir).unwrap();
        // config.json is present but model.safetensors is NOT — simulates
        // a refactor that publishes `latest` before `registry.save_from`.
        fs::write(step_dir.join("config.json"), "{}").unwrap();

        let err = publish_latest_after_weights(dir.path(), "step_000001")
            .expect_err("must refuse to publish without model.safetensors");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(
            msg.contains("model.safetensors"),
            "error must name the missing artifact: {msg}"
        );
        assert!(
            msg.contains("publish-last"),
            "error must cite the publish-last contract: {msg}"
        );

        // Critically: no `latest` should have been created.
        let link = dir.path().join("latest");
        assert!(
            !link.exists() && fs::symlink_metadata(&link).is_err(),
            "latest must not exist after a refused publish"
        );
    }

    #[cfg(unix)]
    #[test]
    fn publish_after_weights_refuses_when_weights_is_directory() {
        // Guard the `.is_file()` check: a directory at the expected path
        // must not count as "weights present". Catches a regression where
        // someone loosens the check to `.exists()`.
        let dir = tempdir().expect("tempdir");
        let step_dir = dir.path().join("step_000001");
        fs::create_dir_all(step_dir.join("model.safetensors")).unwrap();

        let err = publish_latest_after_weights(dir.path(), "step_000001")
            .expect_err("must refuse: model.safetensors is a directory");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}
