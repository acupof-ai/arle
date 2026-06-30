//! T2 SSD persistence tier for Metal KV pages and prefix snapshots.

use super::*;

#[cfg(feature = "metal")]
pub struct MetalPageStore {
    pub(super) pages: HashMap<u32, MetalPageBlock>,
    pub(super) prefixes: HashMap<Vec<u64>, MetalPrefixSnapshot>,
    pub(super) next_logical_id: u64,
    pub(super) ssd: Option<MetalSsdTier>,
}

#[cfg(feature = "metal")]
impl Default for MetalPageStore {
    fn default() -> Self {
        Self {
            pages: HashMap::new(),
            prefixes: HashMap::new(),
            next_logical_id: 1,
            ssd: None,
        }
    }
}

#[cfg(feature = "metal")]
#[derive(Clone)]
pub struct MetalPageBlock {
    pub(super) logical_id: u64,
    pub(super) owner: Option<MetalPageOwner>,
    pub(super) kv_flat: Vec<mlx::MlxArray>,
}

#[cfg(feature = "metal")]
#[derive(Clone, Copy)]
pub struct MetalPageOwner {
    pub(super) slot: usize,
    pub(super) slot_epoch: u64,
    pub(super) page_idx: usize,
}

#[cfg(feature = "metal")]
#[derive(Clone)]
pub struct MetalPrefixSnapshot {
    pub(super) cache_len: usize,
    pub(super) gdr_flat: Vec<mlx::MlxArray>,
}

#[cfg(feature = "metal")]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MetalDiskKey {
    Page(u64),
    Prefix(Vec<u64>),
}

#[cfg(feature = "metal")]
pub struct MetalDiskRecord {
    pub(super) bytes: usize,
    pub(super) stamp: u64,
}

#[cfg(feature = "metal")]
pub struct MetalSsdTier {
    pub(super) root: PathBuf,
    pub(super) budget_bytes: usize,
    pub(super) bytes_per_page: usize,
    pub(super) capacity_pages: usize,
    pub(super) used_bytes: usize,
    pub(super) clock: u64,
    pub(super) records: HashMap<MetalDiskKey, MetalDiskRecord>,
    pub(super) lru: BTreeSet<(u64, MetalDiskKey)>,
    pub(super) tier_to_logical: HashMap<u64, u64>,
    pub(super) read_scratch: Vec<u8>,
}

#[cfg(feature = "metal")]
impl Drop for MetalSsdTier {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[cfg(feature = "metal")]
impl MetalSsdTier {
    pub(super) fn new(root: PathBuf, budget_bytes: usize, bytes_per_page: usize) -> Option<Self> {
        let capacity_pages = budget_bytes.checked_div(bytes_per_page.max(1)).unwrap_or(0);
        if capacity_pages == 0 {
            return None;
        }
        let root = metal_t2_namespace(root);
        if let Err(err) = std::fs::create_dir_all(&root) {
            log::warn!(
                "Metal KV T2 namespace creation failed under {}: {err}",
                root.display()
            );
            return None;
        }
        Some(Self {
            root,
            budget_bytes,
            bytes_per_page,
            capacity_pages,
            used_bytes: 0,
            clock: 0,
            records: HashMap::new(),
            lru: BTreeSet::new(),
            tier_to_logical: HashMap::new(),
            read_scratch: Vec::new(),
        })
    }

pub(super)     fn has_prefix(&self, key: &[u64]) -> bool {
        self.records
            .contains_key(&MetalDiskKey::Prefix(key.to_vec()))
    }

pub(super)     fn write_page(&mut self, block: &MetalPageBlock) -> bool {
        match encode_metal_t2_page(block) {
            Ok(bytes) => self.write_record(MetalDiskKey::Page(block.logical_id), &bytes),
            Err(err) => {
                log::warn!(
                    "Metal KV T2 page encode failed for logical page {}: {err:#}",
                    block.logical_id
                );
                false
            }
        }
    }

pub(super)     fn write_prefix(&mut self, key: &[u64], snapshot: &MetalPrefixSnapshot) -> bool {
        match encode_metal_t2_prefix(key, snapshot) {
            Ok(bytes) => self.write_record(MetalDiskKey::Prefix(key.to_vec()), &bytes),
            Err(err) => {
                log::warn!("Metal KV T2 prefix encode failed for key {key:?}: {err:#}");
                false
            }
        }
    }

pub(super)     fn bind_tier_key(&mut self, tier_key: u64, block: &MetalPageBlock) -> bool {
        if !self
            .records
            .contains_key(&MetalDiskKey::Page(block.logical_id))
            && !self.write_page(block)
        {
            return false;
        }
        self.tier_to_logical.insert(tier_key, block.logical_id);
        true
    }

pub(super)     fn read_tier_page(&mut self, tier_key: u64) -> anyhow::Result<MetalPageBlock> {
        let logical_id = *self
            .tier_to_logical
            .get(&tier_key)
            .ok_or_else(|| anyhow::anyhow!("Metal KV T2 missing tier key {tier_key}"))?;
        self.read_page(logical_id)
    }

pub(super)     fn logical_id_for_tier_key(&self, tier_key: u64) -> Option<u64> {
        self.tier_to_logical.get(&tier_key).copied()
    }

pub(super)     fn read_page(&mut self, logical_id: u64) -> anyhow::Result<MetalPageBlock> {
        let key = MetalDiskKey::Page(logical_id);
        self.read_record(&key)?;
        decode_metal_t2_page(&self.read_scratch, logical_id)
    }

pub(super)     fn read_prefix(&mut self, key: &[u64]) -> anyhow::Result<MetalPrefixSnapshot> {
        let disk_key = MetalDiskKey::Prefix(key.to_vec());
        self.read_record(&disk_key)?;
        decode_metal_t2_prefix(&self.read_scratch, key)
    }

pub(super)     fn drop_tier_entries(&mut self, keys: &[u64]) {
        for key in keys {
            self.tier_to_logical.remove(key);
        }
    }

pub(super)     fn read_record(&mut self, key: &MetalDiskKey) -> anyhow::Result<()> {
        let fingerprint = metal_t2_fingerprint(key);
        kv_native_sys::read_block_into_sharded(&self.root, fingerprint, &mut self.read_scratch)
            .map_err(|err| anyhow::anyhow!("Metal KV T2 read for {key:?}: {err}"))?;
        self.touch_record(key);
        Ok(())
    }

pub(super)     fn write_record(&mut self, key: MetalDiskKey, bytes: &[u8]) -> bool {
        if bytes.len() > self.budget_bytes {
            log::warn!(
                "Metal KV T2 record {key:?} has {} bytes, exceeding budget {}",
                bytes.len(),
                self.budget_bytes
            );
            return false;
        }
        let existing = self.records.get(&key).map_or(0, |record| record.bytes);
        while self
            .used_bytes
            .saturating_sub(existing)
            .saturating_add(bytes.len())
            > self.budget_bytes
        {
            if !self.evict_one_excluding(&key) {
                return false;
            }
        }
        let fingerprint = metal_t2_fingerprint(&key);
        if let Err(err) = kv_native_sys::write_block_cache_sharded(&self.root, fingerprint, bytes) {
            log::warn!(
                "Metal KV T2 write failed for {key:?} under {}: {err}",
                self.root.display()
            );
            return false;
        }
        let stamp = self.next_stamp();
        if let Some(old) = self.records.insert(
            key.clone(),
            MetalDiskRecord {
                bytes: bytes.len(),
                stamp,
            },
        ) {
            self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
            self.lru.remove(&(old.stamp, key.clone()));
        }
        self.used_bytes = self.used_bytes.saturating_add(bytes.len());
        self.lru.insert((stamp, key));
        true
    }

pub(super)     fn touch_record(&mut self, key: &MetalDiskKey) {
        let Some(old_stamp) = self.records.get(key).map(|record| record.stamp) else {
            return;
        };
        let stamp = self.next_stamp();
        self.lru.remove(&(old_stamp, key.clone()));
        if let Some(record) = self.records.get_mut(key) {
            record.stamp = stamp;
        }
        self.lru.insert((stamp, key.clone()));
    }

pub(super)     fn evict_one_excluding(&mut self, excluded: &MetalDiskKey) -> bool {
        let candidate = self
            .lru
            .iter()
            .find(|(_, key)| key != excluded && !self.is_pinned(key))
            .cloned();
        let Some((stamp, key)) = candidate else {
            return false;
        };
        self.lru.remove(&(stamp, key.clone()));
        if let Some(record) = self.records.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(record.bytes);
            let _ =
                kv_native_sys::remove_block_sharded(&self.root, metal_t2_fingerprint(&key), true);
        }
        true
    }

pub(super)     fn is_pinned(&self, key: &MetalDiskKey) -> bool {
        match key {
            MetalDiskKey::Page(logical_id) => self
                .tier_to_logical
                .values()
                .any(|pinned| pinned == logical_id),
            MetalDiskKey::Prefix(_) => false,
        }
    }

pub(super)     fn next_stamp(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }
}

#[cfg(feature = "metal")]
pub fn metal_t2_namespace(root: PathBuf) -> PathBuf {
    let counter = METAL_T2_NAMESPACE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    root.join(format!(
        "arle-metal-kv-tier-{}-{nanos}-{counter}",
        std::process::id()
    ))
}

#[cfg(feature = "metal")]
pub fn metal_t2_fingerprint(key: &MetalDiskKey) -> [u8; 16] {
pub(super)     fn mix(hash: &mut u64, bytes: &[u8]) {
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    let mut h1 = 0xcbf2_9ce4_8422_2325_u64;
    let mut h2 = 0x9e37_79b9_7f4a_7c15_u64;
    match key {
        MetalDiskKey::Page(logical_id) => {
            mix(&mut h1, &[METAL_T2_PAGE_RECORD]);
            mix(&mut h1, &logical_id.to_le_bytes());
            mix(&mut h2, &logical_id.to_be_bytes());
            mix(&mut h2, &[METAL_T2_PAGE_RECORD]);
        }
        MetalDiskKey::Prefix(ids) => {
            mix(&mut h1, &[METAL_T2_PREFIX_RECORD]);
            mix(&mut h1, &(ids.len() as u64).to_le_bytes());
            mix(&mut h2, &(ids.len() as u64).to_be_bytes());
            for id in ids {
                mix(&mut h1, &id.to_le_bytes());
                mix(&mut h2, &id.to_be_bytes());
            }
            mix(&mut h2, &[METAL_T2_PREFIX_RECORD]);
        }
    }
    let mut f = [0u8; 16];
    f[..8].copy_from_slice(&h1.to_le_bytes());
    f[8..].copy_from_slice(&h2.to_le_bytes());
    f
}

#[cfg(feature = "metal")]
pub fn encode_metal_t2_page(block: &MetalPageBlock) -> anyhow::Result<Vec<u8>> {
    encode_metal_t2_record(METAL_T2_PAGE_RECORD, &[block.logical_id], 0, &block.kv_flat)
}

#[cfg(feature = "metal")]
pub fn encode_metal_t2_prefix(
    key: &[u64],
    snapshot: &MetalPrefixSnapshot,
) -> anyhow::Result<Vec<u8>> {
    encode_metal_t2_record(
        METAL_T2_PREFIX_RECORD,
        key,
        snapshot.cache_len,
        &snapshot.gdr_flat,
    )
}

#[cfg(feature = "metal")]
pub fn encode_metal_t2_record(
    kind: u8,
    ids: &[u64],
    cache_len: usize,
    arrays: &[mlx::MlxArray],
) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(METAL_T2_MAGIC);
    out.push(METAL_T2_VERSION);
    out.push(kind);
    put_u32(&mut out, usize_to_u32(ids.len())?);
    for id in ids {
        put_u64(&mut out, *id);
    }
    put_u64(&mut out, usize_to_u64(cache_len)?);
    put_u32(&mut out, usize_to_u32(arrays.len())?);
    for array in arrays {
        let dtype = array.dtype();
        let shape = array.shape().to_vec();
        let bytes = array.export_bytes();
        anyhow::ensure!(
            bytes.len() == expected_array_nbytes(&shape, dtype)?,
            "Metal KV T2 encode byte size mismatch for shape={shape:?} dtype={dtype:?}"
        );
        put_i32(&mut out, dtype.to_raw());
        put_u32(&mut out, usize_to_u32(shape.len())?);
        for dim in &shape {
            put_i32(&mut out, *dim);
        }
        put_u64(&mut out, usize_to_u64(bytes.len())?);
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

#[cfg(feature = "metal")]
pub fn decode_metal_t2_page(bytes: &[u8], logical_id: u64) -> anyhow::Result<MetalPageBlock> {
    let record = decode_metal_t2_record(bytes, METAL_T2_PAGE_RECORD)?;
    anyhow::ensure!(
        record.ids == [logical_id],
        "Metal KV T2 page logical id mismatch: requested={logical_id}, record={:?}",
        record.ids
    );
    Ok(MetalPageBlock {
        logical_id,
        owner: None,
        kv_flat: record.arrays,
    })
}

#[cfg(feature = "metal")]
pub fn decode_metal_t2_prefix(bytes: &[u8], key: &[u64]) -> anyhow::Result<MetalPrefixSnapshot> {
    let record = decode_metal_t2_record(bytes, METAL_T2_PREFIX_RECORD)?;
    anyhow::ensure!(
        record.ids == key,
        "Metal KV T2 prefix key mismatch: requested={key:?}, record={:?}",
        record.ids
    );
    Ok(MetalPrefixSnapshot {
        cache_len: record.cache_len,
        gdr_flat: record.arrays,
    })
}

#[cfg(feature = "metal")]
pub struct DecodedMetalT2Record {
    pub(super) ids: Vec<u64>,
    pub(super) cache_len: usize,
    pub(super) arrays: Vec<mlx::MlxArray>,
}

#[cfg(feature = "metal")]
pub fn decode_metal_t2_record(
    bytes: &[u8],
    expected_kind: u8,
) -> anyhow::Result<DecodedMetalT2Record> {
    let mut cursor = MetalT2Cursor { bytes, offset: 0 };
    anyhow::ensure!(
        cursor.take(METAL_T2_MAGIC.len())? == METAL_T2_MAGIC,
        "Metal KV T2 record magic mismatch"
    );
    let version = cursor.u8()?;
    anyhow::ensure!(
        version == METAL_T2_VERSION,
        "Metal KV T2 record version mismatch: {version}"
    );
    let kind = cursor.u8()?;
    anyhow::ensure!(
        kind == expected_kind,
        "Metal KV T2 record kind mismatch: expected={expected_kind}, got={kind}"
    );
    let id_count = cursor.u32()? as usize;
    let ids: Vec<u64> = (0..id_count)
        .map(|_| cursor.u64())
        .collect::<anyhow::Result<Vec<_>>>()?;
    let cache_len = usize::try_from(cursor.u64()?)
        .map_err(|_| anyhow::anyhow!("Metal KV T2 cache_len exceeds usize"))?;
    let array_count = cursor.u32()? as usize;
    let arrays: Vec<mlx::MlxArray> = (0..array_count)
        .map(|_| {
            let dtype_raw = cursor.i32()?;
            let dtype = mlx::Dtype::from_raw(dtype_raw)
                .ok_or_else(|| anyhow::anyhow!("Metal KV T2 unknown dtype {dtype_raw}"))?;
            let ndim = cursor.u32()? as usize;
            let shape: Vec<i32> = (0..ndim)
                .map(|_| cursor.i32())
                .collect::<anyhow::Result<Vec<_>>>()?;
            let byte_len = usize::try_from(cursor.u64()?)
                .map_err(|_| anyhow::anyhow!("Metal KV T2 array byte_len exceeds usize"))?;
            anyhow::ensure!(
                byte_len == expected_array_nbytes(&shape, dtype)?,
                "Metal KV T2 array byte size mismatch for shape={shape:?} dtype={dtype:?}"
            );
            let data = cursor.take(byte_len)?;
            Ok(mlx::MlxArray::from_bytes(data, &shape, dtype))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(
        cursor.offset == bytes.len(),
        "Metal KV T2 record has {} trailing bytes",
        bytes.len().saturating_sub(cursor.offset)
    );
    Ok(DecodedMetalT2Record {
        ids,
        cache_len,
        arrays,
    })
}

#[cfg(feature = "metal")]
pub struct MetalT2Cursor<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) offset: usize,
}

#[cfg(feature = "metal")]
impl<'a> MetalT2Cursor<'a> {
    pub(super) fn take(&mut self, len: usize) -> anyhow::Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("Metal KV T2 cursor overflow"))?;
        anyhow::ensure!(
            end <= self.bytes.len(),
            "Metal KV T2 truncated record: need {len} bytes at offset {}",
            self.offset
        );
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

pub(super)     fn u8(&mut self) -> anyhow::Result<u8> {
        Ok(self.take(1)?[0])
    }

pub(super)     fn u32(&mut self) -> anyhow::Result<u32> {
        let mut raw = [0u8; 4];
        raw.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(raw))
    }

pub(super)     fn i32(&mut self) -> anyhow::Result<i32> {
        let mut raw = [0u8; 4];
        raw.copy_from_slice(self.take(4)?);
        Ok(i32::from_le_bytes(raw))
    }

pub(super)     fn u64(&mut self) -> anyhow::Result<u64> {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(raw))
    }
}

#[cfg(feature = "metal")]
pub fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "metal")]
pub fn put_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "metal")]
pub fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "metal")]
pub fn usize_to_u32(value: usize) -> anyhow::Result<u32> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("value {value} exceeds u32::MAX"))
}

#[cfg(feature = "metal")]
pub fn usize_to_u64(value: usize) -> anyhow::Result<u64> {
    u64::try_from(value).map_err(|_| anyhow::anyhow!("value {value} exceeds u64::MAX"))
}

#[cfg(feature = "metal")]
pub fn expected_array_nbytes(shape: &[i32], dtype: mlx::Dtype) -> anyhow::Result<usize> {
    let mut elements = 1usize;
    for dim in shape {
        anyhow::ensure!(*dim >= 0, "negative MLX array dimension {dim}");
        elements = elements
            .checked_mul(*dim as usize)
            .ok_or_else(|| anyhow::anyhow!("MLX array shape overflows usize: {shape:?}"))?;
    }
    elements
        .checked_mul(dtype_size(dtype))
        .ok_or_else(|| anyhow::anyhow!("MLX array byte size overflows usize: {shape:?}"))
}

#[cfg(feature = "metal")]
pub fn dtype_size(dtype: mlx::Dtype) -> usize {
    match dtype {
        mlx::Dtype::Bool | mlx::Dtype::Uint8 | mlx::Dtype::Int8 => 1,
        mlx::Dtype::Uint16 | mlx::Dtype::Int16 | mlx::Dtype::Float16 | mlx::Dtype::Bfloat16 => 2,
        mlx::Dtype::Uint32 | mlx::Dtype::Int32 | mlx::Dtype::Float32 => 4,
        mlx::Dtype::Uint64 | mlx::Dtype::Int64 | mlx::Dtype::Float64 | mlx::Dtype::Complex64 => 8,
    }
}

#[cfg(feature = "metal")]
impl MetalPageStore {
    pub(super) fn set_ssd(&mut self, root: PathBuf, budget_bytes: usize, bytes_per_page: usize) -> bool {
        let Some(ssd) = MetalSsdTier::new(root, budget_bytes, bytes_per_page) else {
            return false;
        };
        eprintln!(
            "[infer-metal] KV T2 SSD tier: root={}, budget_bytes={}, capacity_pages={}",
            ssd.root.display(),
            ssd.budget_bytes,
            ssd.capacity_pages
        );
        self.ssd = Some(ssd);
        true
    }

pub(super)     fn kv_tier_capacity_pages(&self) -> usize {
        self.ssd.as_ref().map_or(0, |ssd| ssd.capacity_pages)
    }

pub(super)     fn kv_tier_page_bytes(&self) -> usize {
        self.ssd.as_ref().map_or(0, |ssd| ssd.bytes_per_page)
    }

pub(super)     fn kv_tier_disk_pages(&self) -> usize {
        self.ssd.as_ref().map_or(0, |ssd| ssd.tier_to_logical.len())
    }

pub(super)     fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        self.ssd.as_ref().and_then(|ssd| {
            ssd.tier_to_logical
                .contains_key(&key)
                .then_some(infer_seam::KvTierLocation::Disk)
        })
    }

    /// Largest leading block count for which Metal has a complete restore
    /// image. Demoted keys are checked before promotion, so an unusable prefix
    /// tail is never read back from T2.
pub(super)     fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        (1..=blocks.len())
            .rev()
            .find(|&k| {
                self.logical_key_for_prefix_blocks(&blocks[..k])
                    .is_some_and(|key| self.prefix_available(&key))
            })
            .unwrap_or(0)
    }

pub(super)     fn release_pages(&mut self, pages: &[u32]) {
        if pages.is_empty() {
            return;
        }
        let released_logical_ids: Vec<u64> = pages
            .iter()
            .filter_map(|page| self.pages.remove(page).map(|b| b.logical_id))
            .collect();
        if released_logical_ids.is_empty() {
            return;
        }
        self.prefixes.retain(|key, _| {
            !released_logical_ids
                .iter()
                .any(|logical_id| key.contains(logical_id))
        });
    }

pub(super)     fn next_logical_id(&mut self) -> u64 {
        let id = self.next_logical_id.max(1);
        self.next_logical_id = id.saturating_add(1);
        id
    }

pub(super)     fn logical_key_for_pages(&self, pages: &[u32]) -> Option<Vec<u64>> {
        pages
            .iter()
            .map(|page| self.pages.get(page).map(|b| b.logical_id))
            .collect()
    }

pub(super)     fn logical_key_for_prefix_blocks(&self, blocks: &[PrefixBlock]) -> Option<Vec<u64>> {
        blocks
            .iter()
            .map(|block| match *block {
                PrefixBlock::ResidentPage(page) => Some(self.pages.get(&page)?.logical_id),
                PrefixBlock::DemotedKey(tier_key) => {
                    self.ssd.as_ref()?.logical_id_for_tier_key(tier_key)
                }
            })
            .collect()
    }

pub(super)     fn prefix_available(&self, key: &[u64]) -> bool {
        self.prefixes.contains_key(key) || self.ssd.as_ref().is_some_and(|ssd| ssd.has_prefix(key))
    }

pub(super)     fn ensure_prefix_snapshot_resident(&mut self, key: &[u64]) -> anyhow::Result<()> {
        if self.prefixes.contains_key(key) {
            return Ok(());
        }
        let ssd = self
            .ssd
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Metal prefix snapshot {key:?} is not resident"))?;
        let snapshot = ssd.read_prefix(key)?;
        self.prefixes.insert(key.to_vec(), snapshot);
        Ok(())
    }

pub(super)     fn demote_prefix_pages(&mut self, entries: &[(u32, u64)]) -> anyhow::Result<usize> {
        let Some(ssd) = self.ssd.as_mut() else {
            return Ok(0);
        };
        let mut accepted = 0usize;
        for &(page, tier_key) in entries {
            let Some(block) = self.pages.get(&page) else {
                break;
            };
            if !ssd.bind_tier_key(tier_key, block) {
                break;
            }
            accepted = accepted.saturating_add(1);
        }
        Ok(accepted)
    }

pub(super)     fn promote_prefix_pages(&mut self, entries: &[(u64, u32)]) -> anyhow::Result<()> {
        let ssd = self
            .ssd
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Metal KV T2 store is not configured"))?;
        for &(tier_key, dst_page) in entries {
            let block = ssd.read_tier_page(tier_key)?;
            if let Some(old) = self.pages.insert(dst_page, block) {
                let old_id = old.logical_id;
                self.prefixes.retain(|key, _| !key.contains(&old_id));
            }
        }
        Ok(())
    }

pub(super)     fn drop_kv_tier_entries(&mut self, keys: &[u64]) {
        if let Some(ssd) = self.ssd.as_mut() {
            ssd.drop_tier_entries(keys);
        }
    }

pub(super)     fn publish_slot(&mut self, slot: &MetalSlotState, kv: &dyn KvPool) -> anyhow::Result<()> {
        let page_size = kv.page_size().max(1);
        let full_pages = slot.cache_len / page_size;
        if full_pages == 0 {
            return Ok(());
        }

        let page_ids = kv.page_indices(slot.slot);
        let publish_pages = full_pages.min(page_ids.len());
        let mut overwritten_logical_ids = Vec::new();
        for (page_idx, page_id) in page_ids.iter().take(publish_pages).enumerate() {
            let start = page_idx * page_size;
            let end = start + page_size;
            let kv_flat = slot
                .kv_flat
                .iter()
                .map(|array| slice_kv_tokens(array, start, end))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let owner = MetalPageOwner {
                slot: slot.slot,
                slot_epoch: slot.slot_epoch,
                page_idx,
            };
            let logical_id = if self
                .pages
                .get(page_id)
                .and_then(|block| block.owner)
                .is_some_and(|old| {
                    old.slot == owner.slot
                        && old.slot_epoch == owner.slot_epoch
                        && old.page_idx == owner.page_idx
                }) {
                self.pages
                    .get(page_id)
                    .expect("page checked above")
                    .logical_id
            } else {
                self.next_logical_id()
            };
            // Host page ids may be reused after the seam frees a slot. Overwrite
            // with the current slot's contents; retained/shared pages cannot be
            // reallocated by the host pool, so this does not corrupt live reuse.
            let block = MetalPageBlock {
                logical_id,
                owner: Some(owner),
                kv_flat,
            };
            if let Some(old) = self.pages.insert(*page_id, block) {
                if old.logical_id != logical_id {
                    overwritten_logical_ids.push(old.logical_id);
                }
            }
            if let (Some(ssd), Some(block)) = (self.ssd.as_mut(), self.pages.get(page_id)) {
                ssd.write_page(block);
            }
        }
        if !overwritten_logical_ids.is_empty()
            && let Some(live_key) = self.logical_key_for_pages(&page_ids[..publish_pages])
        {
            // Alias hazard: overwriting a logical page means this page id was
            // recycled to a new occupant. Any surviving prefix containing the old
            // logical id would pair NEW K/V with a STALE restore snapshot. Keep
            // only exact prefixes of the live occupant's logical page list.
            self.prefixes.retain(|key, _| {
                !overwritten_logical_ids
                    .iter()
                    .any(|logical_id| key.contains(logical_id))
                    || (key.len() <= live_key.len() && live_key[..key.len()] == key[..])
            });
        }

        // A reusable prefix boundary is valid only when the page-id prefix and
        // every prefix-wide side state describe the same token length. Publish
        // that restore image only at exact page boundaries.
        if slot.cache_len.is_multiple_of(page_size) && publish_pages == full_pages {
            if let Some(key) = self.logical_key_for_pages(&page_ids[..full_pages]) {
                let snapshot = MetalPrefixSnapshot {
                    cache_len: slot.cache_len,
                    gdr_flat: slot.gdr_flat.clone(),
                };
                self.prefixes.insert(key.clone(), snapshot);
                if let (Some(ssd), Some(snapshot)) = (self.ssd.as_mut(), self.prefixes.get(&key)) {
                    ssd.write_prefix(&key, snapshot);
                }
            }
        }

        Ok(())
    }

pub(super)     fn materialize_slot_from_prefix(
        &mut self,
        slot: usize,
        slot_epoch: u64,
        kv: &dyn KvPool,
        prefix_tokens: usize,
        capacity_tokens: usize,
    ) -> anyhow::Result<MetalSlotState> {
        let page_size = kv.page_size().max(1);
        anyhow::ensure!(
            prefix_tokens.is_multiple_of(page_size),
            "Metal prefix attach requires page-aligned prefix: prefix_tokens={}, page_size={}",
            prefix_tokens,
            page_size
        );
        let prefix_pages = prefix_tokens / page_size;
        let slot_pages = kv.page_indices(slot);
        anyhow::ensure!(
            slot_pages.len() >= prefix_pages,
            "Metal prefix attach for slot {slot} needs {prefix_pages} pages, host slot has {}",
            slot_pages.len()
        );
        let key = self
            .logical_key_for_pages(&slot_pages[..prefix_pages])
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Metal prefix attach missing resident logical pages for slot {slot}, prefix_tokens={prefix_tokens}"
                )
            })?;
        self.ensure_prefix_snapshot_resident(&key)?;
        let snapshot = self.prefixes.get(&key).ok_or_else(|| {
            anyhow::anyhow!(
                "Metal prefix attach missing GDR snapshot for slot {slot}, prefix_tokens={prefix_tokens}, key={key:?}"
            )
        })?;
        anyhow::ensure!(
            snapshot.cache_len == prefix_tokens,
            "Metal prefix snapshot length mismatch for slot {slot}: requested={}, snapshot={}",
            prefix_tokens,
            snapshot.cache_len
        );

        let first_page = key
            .first()
            .ok_or_else(|| anyhow::anyhow!("Metal prefix attach got empty logical key"))?;
        let first_physical_page = slot_pages
            .first()
            .ok_or_else(|| anyhow::anyhow!("Metal prefix attach got empty page key"))?;
        let first_block = self.pages.get(first_physical_page).ok_or_else(|| {
            anyhow::anyhow!(
                "Metal prefix attach missing K/V page {first_physical_page} for slot {slot}, logical={first_page}"
            )
        })?;

        let capacity = round_up_capacity(capacity_tokens.max(prefix_tokens)) as usize;
        let kv_flat = (0..first_block.kv_flat.len())
            .map(|array_idx| -> anyhow::Result<_> {
                let page_arrays = slot_pages[..prefix_pages]
                    .iter()
                    .map(|page| -> anyhow::Result<_> {
                        let block = self.pages.get(page).ok_or_else(|| {
                            anyhow::anyhow!(
                                "Metal prefix attach missing K/V page {page} for slot {slot}"
                            )
                        })?;
                        block.kv_flat.get(array_idx).cloned().ok_or_else(|| {
                            anyhow::anyhow!(
                                "Metal prefix attach K/V page {page} is missing array index {array_idx}"
                            )
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let prefix_array = concatenate_or_single(page_arrays);
                let shape = prefix_array.shape().to_vec();
                anyhow::ensure!(
                    shape.len() == 4 && shape[2] as usize == prefix_tokens,
                    "Metal prefix K/V materialization shape mismatch for slot {slot}: shape={shape:?}, prefix_tokens={prefix_tokens}"
                );
                Ok(if capacity > prefix_tokens {
                    let mut zero_shape = shape;
                    zero_shape[2] = usize_to_i32(capacity - prefix_tokens)?;
                    let zeros = mlx::zeros(&zero_shape, prefix_array.dtype());
                    mlx::concatenate_axis(&[prefix_array, zeros], 2)
                } else {
                    prefix_array
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(MetalSlotState::from_arrays(
            slot,
            slot_epoch,
            prefix_tokens,
            kv_flat,
            snapshot.gdr_flat.clone(),
        ))
    }
}
