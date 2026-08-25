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
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f32())
            .collect())
    };
    Ok((read(MARKOV_W1)?, read(MARKOV_W2)?))
}
