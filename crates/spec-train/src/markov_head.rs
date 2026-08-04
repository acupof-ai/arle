//! The DSpark Markov head artifact: `bias[v] = Σ_r w1[c][r]·w2[v][r]`, stored in
//! the draft loader's own tensor names so a head written here and one read off a
//! checkpoint are the same file.

use anyhow::{Context, Result, ensure};
use std::path::Path;

const MARKOV_W1: &str = "markov_head.markov_w1.weight";
const MARKOV_W2: &str = "markov_head.markov_w2.weight";

/// `w1`'s `[vocab, rank]` shape, read from the header alone. The serve needs it
/// before the engine loads, to size the head slot a DFlash backbone ships
/// without — guessing a rank would only turn a load into a size mismatch.
pub fn shape(path: &Path) -> Result<(usize, usize)> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read markov head {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| anyhow::anyhow!("parse markov head {}: {e}", path.display()))?;
    let shape = st
        .tensor(MARKOV_W1)
        .map_err(|e| anyhow::anyhow!("{} missing {MARKOV_W1}: {e}", path.display()))?
        .shape()
        .to_vec();
    match shape[..] {
        [vocab, rank] => Ok((vocab, rank)),
        _ => anyhow::bail!(
            "{} {MARKOV_W1}: expected [vocab, rank], got {shape:?}",
            path.display()
        ),
    }
}

/// Read a head as host f32 `(w1, w2)`, ready for
/// `LoadedInferenceEngine::update_dspark_markov_weights`.
pub fn load(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read markov head {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| anyhow::anyhow!("parse markov head {}: {e}", path.display()))?;
    let read = |name: &str| -> Result<Vec<f32>> {
        let t = st
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("{} missing {name}: {e}", path.display()))?;
        ensure!(
            t.dtype() == safetensors::Dtype::BF16,
            "{} {name}: expected BF16, got {:?}",
            path.display(),
            t.dtype()
        );
        Ok(t.data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect())
    };
    Ok((read(MARKOV_W1)?, read(MARKOV_W2)?))
}

/// Write `(w1, w2)` as `[rows, rank]` bf16.
pub fn save(path: &Path, w1: &[f32], w2: &[f32], rank: usize) -> Result<()> {
    ensure!(rank > 0, "markov head rank must be positive");
    ensure!(
        w1.len() == w2.len() && w1.len().is_multiple_of(rank),
        "markov head shapes: w1={} w2={} rank={rank}",
        w1.len(),
        w2.len()
    );
    let rows = w1.len() / rank;
    let to_bf16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
            .collect()
    };
    let (w1_b, w2_b) = (to_bf16(w1), to_bf16(w2));
    let tensors = [(MARKOV_W1, &w1_b), (MARKOV_W2, &w2_b)].map(|(name, bytes)| {
        (
            name.to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![rows, rank], bytes)
                .expect("bf16 view over own buffer"),
        )
    });
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // Write-then-rename: `serialize_to_file` truncates first, so a failed save
    // would otherwise destroy the last good checkpoint.
    let tmp = path.with_extension("safetensors.tmp");
    safetensors::serialize_to_file(tensors, None, &tmp)
        .map_err(|e| anyhow::anyhow!("markov head save to {} failed: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}
