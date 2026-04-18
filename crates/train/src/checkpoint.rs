//! Minimal f32 checkpoint format for TinyLM-sized models.
//!
//! Self-contained serialization — no postcard, no serde. Walks a
//! `Module`'s parameter list in declaration order and writes one
//! `(shape, data)` record per tensor. Load path matches the file
//! record against the live module's parameter shapes and overwrites
//! `.data` in place, so the caller's `TensorStore` keeps its tensor
//! ids stable.
//!
//! ## On-disk layout
//! ```text
//! magic: 8 bytes, literal "TLMCKP02"
//! config: 7 × u64 LE — vocab_size, d_model, n_layers, n_heads, d_head,
//!                     d_ff, max_seq_len
//! num_tensors: u64 LE
//! per tensor:
//!   ndim: u64 LE
//!   shape[ndim]: u64 LE each
//!   data_len: u64 LE  (== product(shape))
//!   data[data_len]: f32 LE
//! ```

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use autograd::{TensorId, TensorStore, module::Module};

use crate::model::TinyLMConfig;

const MAGIC: &[u8; 8] = b"TLMCKP02";

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
}

pub type Result<T> = std::result::Result<T, CheckpointError>;

pub fn save<M: Module>(
    model: &M,
    config: &TinyLMConfig,
    store: &TensorStore,
    path: impl AsRef<Path>,
) -> Result<()> {
    let params = model.parameters();
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    writer.write_all(MAGIC)?;
    write_u64(&mut writer, config.vocab_size as u64)?;
    write_u64(&mut writer, config.d_model as u64)?;
    write_u64(&mut writer, config.n_layers as u64)?;
    write_u64(&mut writer, config.n_heads as u64)?;
    write_u64(&mut writer, config.d_head as u64)?;
    write_u64(&mut writer, config.d_ff as u64)?;
    write_u64(&mut writer, config.max_seq_len as u64)?;
    write_u64(&mut writer, params.len() as u64)?;
    for id in &params {
        let tensor = store.get(*id).ok_or(CheckpointError::MissingTensor(*id))?;
        write_u64(&mut writer, tensor.shape.len() as u64)?;
        for dim in &tensor.shape {
            write_u64(&mut writer, *dim as u64)?;
        }
        write_u64(&mut writer, tensor.data.len() as u64)?;
        for value in &tensor.data {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
    Ok(())
}

pub fn read_config(path: impl AsRef<Path>) -> Result<TinyLMConfig> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    read_header(&mut reader).map(|(config, _)| config)
}

fn read_header<R: Read>(reader: &mut R) -> Result<(TinyLMConfig, usize)> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(CheckpointError::BadMagic {
            expected: *MAGIC,
            actual: magic,
        });
    }
    let config = TinyLMConfig {
        vocab_size: read_u64(reader)? as usize,
        d_model: read_u64(reader)? as usize,
        n_layers: read_u64(reader)? as usize,
        n_heads: read_u64(reader)? as usize,
        d_head: read_u64(reader)? as usize,
        d_ff: read_u64(reader)? as usize,
        max_seq_len: read_u64(reader)? as usize,
        lora: None,
    };
    let num_tensors = read_u64(reader)? as usize;
    Ok((config, num_tensors))
}

pub fn load<M: Module>(model: &M, store: &mut TensorStore, path: impl AsRef<Path>) -> Result<()> {
    let params = model.parameters();
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let (_config, num_tensors) = read_header(&mut reader)?;
    if num_tensors != params.len() {
        return Err(CheckpointError::ParamCount {
            file: num_tensors,
            model: params.len(),
        });
    }

    for (index, id) in params.iter().enumerate() {
        let ndim = read_u64(&mut reader)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64(&mut reader)? as usize);
        }
        let data_len = read_u64(&mut reader)? as usize;
        let mut data = vec![0.0_f32; data_len];
        let mut buf = [0u8; 4];
        for slot in data.iter_mut() {
            reader.read_exact(&mut buf)?;
            *slot = f32::from_le_bytes(buf);
        }

        let tensor = store
            .get_mut(*id)
            .ok_or(CheckpointError::MissingTensor(*id))?;
        if tensor.shape != shape {
            return Err(CheckpointError::ShapeMismatch {
                index,
                file: shape,
                model: tensor.shape.clone(),
            });
        }
        if tensor.data.len() != data.len() {
            return Err(CheckpointError::ShapeMismatch {
                index,
                file: shape,
                model: tensor.shape.clone(),
            });
        }
        tensor.data = data;
    }

    Ok(())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
