use super::*;

/// Host image of one whole DSv4 slot for capacity spill: the per-layer
/// attention carry (SW ring, compressor/indexer, FlashMLA metadata, DSA) plus
/// the slot's FP8 KV band pages from each layer's shared pool.
pub(crate) struct Dsv4SlotImage {
    pub(crate) seq_len: usize,
    pub(crate) layers: Vec<crate::attention::Dsv4LayerAttentionImage>,
    pub(crate) kv_pages: Vec<Vec<u8>>,
}

impl Dsv4SlotImage {
    pub(crate) fn dram_bytes(&self) -> usize {
        let mut bytes = 8 + 8;
        for layer in &self.layers {
            bytes += layer.sw_window_cache.len() * 2;
            if let Some(c) = &layer.compressor {
                bytes += compressor_image_bytes(c);
            }
            if let Some(c) = &layer.indexer {
                bytes += compressor_image_bytes(c);
            }
            if layer.flashmla.is_some() {
                bytes += 1 + 8;
            }
            if let Some(d) = &layer.dsa_official {
                bytes += d.rotated_keys.len() * 2 + 8;
            }
        }
        for pages in &self.kv_pages {
            bytes += pages.len();
        }
        bytes
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.dram_bytes() + 256);
        buf.extend_from_slice(&(self.seq_len as u64).to_le_bytes());
        buf.extend_from_slice(&(self.layers.len() as u64).to_le_bytes());
        for layer in &self.layers {
            write_bf16_vec(&mut buf, &layer.sw_window_cache);
            write_compressor(&mut buf, layer.compressor.as_ref());
            write_compressor(&mut buf, layer.indexer.as_ref());
            write_flashmla(&mut buf, layer.flashmla.as_ref());
            write_dsa(&mut buf, layer.dsa_official.as_ref());
        }
        buf.extend_from_slice(&(self.kv_pages.len() as u64).to_le_bytes());
        for pages in &self.kv_pages {
            buf.extend_from_slice(&(pages.len() as u64).to_le_bytes());
            buf.extend_from_slice(pages);
        }
        buf
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let seq_len = read_u64(bytes, &mut pos)? as usize;
        let num_layers = read_u64(bytes, &mut pos)? as usize;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let sw_window_cache = read_bf16_vec(bytes, &mut pos)?;
            let compressor = read_compressor(bytes, &mut pos)?;
            let indexer = read_compressor(bytes, &mut pos)?;
            let flashmla = read_flashmla(bytes, &mut pos)?;
            let dsa_official = read_dsa(bytes, &mut pos)?;
            layers.push(crate::attention::Dsv4LayerAttentionImage {
                sw_window_cache,
                compressor,
                indexer,
                flashmla,
                dsa_official,
            });
        }
        let num_kv = read_u64(bytes, &mut pos)? as usize;
        let mut kv_pages = Vec::with_capacity(num_kv);
        for _ in 0..num_kv {
            kv_pages.push(read_bytes(bytes, &mut pos)?);
        }
        Ok(Dsv4SlotImage {
            seq_len,
            layers,
            kv_pages,
        })
    }
}

pub(super) fn compressor_image_bytes(c: &crate::attention::Dsv4CompressorImage) -> usize {
    c.pending_kv.len() * 2
        + c.pending_score.len() * 2
        + c.prev_overlap_kv.len() * 2
        + c.prev_overlap_score.len() * 2
        + c.compressed.len() * 2
        + c.fp32_pending_kv.len() * 4
        + c.fp32_pending_score.len() * 4
        + c.fp32_prev_kv.len() * 4
        + c.fp32_prev_score.len() * 4
        + 8
        + 1
}

pub(super) fn write_bytes(buf: &mut Vec<u8>, data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buf.extend_from_slice(data);
}

pub(super) fn write_bf16_vec(buf: &mut Vec<u8>, v: &[half::bf16]) {
    // SAFETY: bf16 is #[repr(transparent)] over u16; byte view is valid.
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    write_bytes(buf, bytes);
}

pub(super) fn write_f32_vec(buf: &mut Vec<u8>, v: &[f32]) {
    // SAFETY: f32 has no padding bytes; byte view is valid.
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    write_bytes(buf, bytes);
}

pub(super) fn write_compressor(
    buf: &mut Vec<u8>,
    c: Option<&crate::attention::Dsv4CompressorImage>,
) {
    match c {
        None => buf.push(0),
        Some(c) => {
            buf.push(1);
            write_bf16_vec(buf, &c.pending_kv);
            write_bf16_vec(buf, &c.pending_score);
            write_bf16_vec(buf, &c.prev_overlap_kv);
            write_bf16_vec(buf, &c.prev_overlap_score);
            write_bf16_vec(buf, &c.compressed);
            buf.extend_from_slice(&(c.compressed_seq_len as u64).to_le_bytes());
            write_f32_vec(buf, &c.fp32_pending_kv);
            write_f32_vec(buf, &c.fp32_pending_score);
            write_f32_vec(buf, &c.fp32_prev_kv);
            write_f32_vec(buf, &c.fp32_prev_score);
            buf.push(c.fp32_carry_stale as u8);
        }
    }
}

pub(super) fn write_flashmla(buf: &mut Vec<u8>, f: Option<&crate::attention::Dsv4FlashMlaImage>) {
    match f {
        None => buf.push(0),
        Some(f) => {
            buf.push(1);
            buf.push(f.fp8_kv_sw_bootstrapped as u8);
            buf.extend_from_slice(&(f.fp8_kv_comp_packed_rows as u64).to_le_bytes());
        }
    }
}

pub(super) fn write_dsa(buf: &mut Vec<u8>, d: Option<&crate::attention::Dsv4DsaImage>) {
    match d {
        None => buf.push(0),
        Some(d) => {
            buf.push(1);
            write_bf16_vec(buf, &d.rotated_keys);
            buf.extend_from_slice(&(d.packed_rows as u64).to_le_bytes());
        }
    }
}

pub(super) fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let end = pos
        .checked_add(8)
        .ok_or_else(|| anyhow!("dsv4 slot image u64 overflow"))?;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| anyhow!("dsv4 slot image truncated u64 at {pos}"))?;
    *pos = end;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

pub(super) fn read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8> {
    let end = pos
        .checked_add(1)
        .ok_or_else(|| anyhow!("dsv4 slot image u8 overflow"))?;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| anyhow!("dsv4 slot image truncated u8 at {pos}"))?;
    *pos = end;
    Ok(slice[0])
}

pub(super) fn read_bytes(bytes: &[u8], pos: &mut usize) -> Result<Vec<u8>> {
    let len = read_u64(bytes, pos)? as usize;
    let end = pos
        .checked_add(len)
        .ok_or_else(|| anyhow!("dsv4 slot image bytes overflow"))?;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| anyhow!("dsv4 slot image truncated bytes at {pos}"))?;
    *pos = end;
    Ok(slice.to_vec())
}

pub(super) fn read_bf16_vec(bytes: &[u8], pos: &mut usize) -> Result<Vec<half::bf16>> {
    let len = read_u64(bytes, pos)? as usize;
    ensure!(
        len.is_multiple_of(2),
        "dsv4 slot image bf16 vec odd length {len}"
    );
    let end = pos
        .checked_add(len)
        .ok_or_else(|| anyhow!("dsv4 slot image bf16 overflow"))?;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| anyhow!("dsv4 slot image truncated bf16 at {pos}"))?;
    *pos = end;
    let n = len / 2;
    let mut out = Vec::with_capacity(n);
    // SAFETY: bf16 is #[repr(transparent)] over u16; slice is exactly n*2 bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(slice.as_ptr(), out.as_mut_ptr() as *mut u8, len);
        out.set_len(n);
    }
    Ok(out)
}

pub(super) fn read_f32_vec(bytes: &[u8], pos: &mut usize) -> Result<Vec<f32>> {
    let len = read_u64(bytes, pos)? as usize;
    ensure!(
        len.is_multiple_of(4),
        "dsv4 slot image f32 vec odd length {len}"
    );
    let end = pos
        .checked_add(len)
        .ok_or_else(|| anyhow!("dsv4 slot image f32 overflow"))?;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| anyhow!("dsv4 slot image truncated f32 at {pos}"))?;
    *pos = end;
    let n = len / 4;
    let mut out = Vec::with_capacity(n);
    // SAFETY: f32 has no padding; slice is exactly n*4 bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(slice.as_ptr(), out.as_mut_ptr() as *mut u8, len);
        out.set_len(n);
    }
    Ok(out)
}

pub(super) fn read_compressor(
    bytes: &[u8],
    pos: &mut usize,
) -> Result<Option<crate::attention::Dsv4CompressorImage>> {
    let flag = read_u8(bytes, pos)?;
    if flag == 0 {
        return Ok(None);
    }
    let pending_kv = read_bf16_vec(bytes, pos)?;
    let pending_score = read_bf16_vec(bytes, pos)?;
    let prev_overlap_kv = read_bf16_vec(bytes, pos)?;
    let prev_overlap_score = read_bf16_vec(bytes, pos)?;
    let compressed = read_bf16_vec(bytes, pos)?;
    let compressed_seq_len = read_u64(bytes, pos)? as usize;
    let fp32_pending_kv = read_f32_vec(bytes, pos)?;
    let fp32_pending_score = read_f32_vec(bytes, pos)?;
    let fp32_prev_kv = read_f32_vec(bytes, pos)?;
    let fp32_prev_score = read_f32_vec(bytes, pos)?;
    let fp32_carry_stale = read_u8(bytes, pos)? != 0;
    Ok(Some(crate::attention::Dsv4CompressorImage {
        pending_kv,
        pending_score,
        prev_overlap_kv,
        prev_overlap_score,
        compressed,
        compressed_seq_len,
        fp32_pending_kv,
        fp32_pending_score,
        fp32_prev_kv,
        fp32_prev_score,
        fp32_carry_stale,
    }))
}

pub(super) fn read_flashmla(
    bytes: &[u8],
    pos: &mut usize,
) -> Result<Option<crate::attention::Dsv4FlashMlaImage>> {
    let flag = read_u8(bytes, pos)?;
    if flag == 0 {
        return Ok(None);
    }
    let fp8_kv_sw_bootstrapped = read_u8(bytes, pos)? != 0;
    let fp8_kv_comp_packed_rows = read_u64(bytes, pos)? as usize;
    Ok(Some(crate::attention::Dsv4FlashMlaImage {
        fp8_kv_sw_bootstrapped,
        fp8_kv_comp_packed_rows,
    }))
}

pub(super) fn read_dsa(
    bytes: &[u8],
    pos: &mut usize,
) -> Result<Option<crate::attention::Dsv4DsaImage>> {
    let flag = read_u8(bytes, pos)?;
    if flag == 0 {
        return Ok(None);
    }
    let rotated_keys = read_bf16_vec(bytes, pos)?;
    let packed_rows = read_u64(bytes, pos)? as usize;
    Ok(Some(crate::attention::Dsv4DsaImage {
        rotated_keys,
        packed_rows,
    }))
}
