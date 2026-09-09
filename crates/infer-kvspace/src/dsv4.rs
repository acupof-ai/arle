//! DSv4 prefix-page content codec and capture lifecycle. Host-only: the
//! codec is byte math over `Vec<u8>` / `Vec<bf16>`, and the pending-page
//! state machine tracks confirm/cancel/repair across the radix dedup that
//! happens between capture and publish.

use anyhow::{Result, ensure};

/// One layer's share of a per-page entry; empty vec = section absent. Boundary
/// sections (`overlap_*`, `idx_overlap_*`, `ring`) only exist when the forward
/// ended exactly at the page end — an overshot boundary is unrecoverable.
///
/// The FP8 compressed band is NOT captured: it is written only on the decode
/// lane, so a prefill-time capture holds zeros and restoring it corrupts every
/// warm decode; the first post-restore decode's bulk pack rebuilds it from the
/// restored staging.
#[derive(Default, Debug, PartialEq)]
pub struct Dsv4LayerPageState {
    /// Main compressor bf16 staging rows. Indexer staging is NOT captured — its
    /// only reader drains the delta `[packed_rows, seq_len)`, empty at a boundary.
    pub staging: Vec<half::bf16>,
    pub dsa_data: Vec<u8>,
    pub dsa_scale: Vec<u8>,
    pub overlap_kv: Vec<half::bf16>,
    pub overlap_score: Vec<half::bf16>,
    pub idx_overlap_kv: Vec<half::bf16>,
    pub idx_overlap_score: Vec<half::bf16>,
    /// Full bf16 SW ring (the FP8 ring region is rebuilt from it by the
    /// bootstrap, so bf16 stays the single source).
    pub ring: Vec<half::bf16>,
    /// Frontier-tail sections: the sub-page tail `[matched_len, finish_len)` the
    /// radix match can't cover, present only on the frontier entry when a finish
    /// landed off a page boundary. `pending_*` = the incomplete compress block's
    /// raw rows (`finish_len % ratio` tokens × width).
    pub pending_kv: Vec<half::bf16>,
    pub pending_score: Vec<half::bf16>,
    /// #165: without it an off-ratio restore left the prior occupant's rows in
    /// the indexer's bf16 pending.
    pub idx_pending_kv: Vec<half::bf16>,
    pub idx_pending_score: Vec<half::bf16>,
    pub tail_staging: Vec<half::bf16>,
    /// Tail-page DSA rows; no cache-page straddle (starts at a 16-row multiple,
    /// < 16 rows).
    pub tail_dsa_data: Vec<u8>,
    pub tail_dsa_scale: Vec<u8>,
}

#[derive(Debug)]
pub struct Dsv4PrefixPageEntry {
    /// KV content is position-dependent, so a restore must see the same index —
    /// a mismatch means the host page id was recycled into different content.
    pub page_index: u32,
    /// Boundary sections present (forward ended exactly at this page's end).
    pub boundary: bool,
    pub layers: Vec<Dsv4LayerPageState>,
}

// Doubles as format version: bumped on layout change so stale entries fail-close
// at the header instead of misparsing positional sections.
const ENTRY_MAGIC: &[u8; 4] = b"DSP2";

fn push_bytes(buf: &mut Vec<u8>, v: &[u8]) {
    buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
    buf.extend_from_slice(v);
}

fn push_bf16(buf: &mut Vec<u8>, v: &[half::bf16]) {
    let byte_len = v.len() * 2;
    buf.extend_from_slice(&(byte_len as u32).to_le_bytes());
    // SAFETY: half::bf16 is #[repr(transparent)] over u16; byte view is valid.
    let raw = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, byte_len) };
    buf.extend_from_slice(raw);
}

fn read_bytes(pos: &mut usize, bytes: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        *pos + 4 <= bytes.len(),
        "prefix-state entry truncated at len"
    );
    let len = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap()) as usize;
    *pos += 4;
    ensure!(
        *pos + len <= bytes.len(),
        "prefix-state entry truncated at section"
    );
    let v = bytes[*pos..*pos + len].to_vec();
    *pos += len;
    Ok(v)
}

fn read_bf16(pos: &mut usize, bytes: &[u8]) -> Result<Vec<half::bf16>> {
    let raw = read_bytes(pos, bytes)?;
    ensure!(
        raw.len().is_multiple_of(2),
        "prefix-state bf16 section has odd byte length {}",
        raw.len()
    );
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::bf16::from_le_bytes(*c))
        .collect())
}

impl Dsv4PrefixPageEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.host_bytes() + 64);
        buf.extend_from_slice(ENTRY_MAGIC);
        buf.extend_from_slice(&self.page_index.to_le_bytes());
        buf.push(u8::from(self.boundary));
        buf.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            push_bf16(&mut buf, &layer.staging);
            push_bytes(&mut buf, &layer.dsa_data);
            push_bytes(&mut buf, &layer.dsa_scale);
            push_bf16(&mut buf, &layer.overlap_kv);
            push_bf16(&mut buf, &layer.overlap_score);
            push_bf16(&mut buf, &layer.idx_overlap_kv);
            push_bf16(&mut buf, &layer.idx_overlap_score);
            push_bf16(&mut buf, &layer.ring);
            push_bf16(&mut buf, &layer.pending_kv);
            push_bf16(&mut buf, &layer.pending_score);
            push_bf16(&mut buf, &layer.idx_pending_kv);
            push_bf16(&mut buf, &layer.idx_pending_score);
            push_bf16(&mut buf, &layer.tail_staging);
            push_bytes(&mut buf, &layer.tail_dsa_data);
            push_bytes(&mut buf, &layer.tail_dsa_scale);
        }
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= 13 && &bytes[..4] == ENTRY_MAGIC,
            "bad prefix-state entry header"
        );
        let page_index = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let boundary = bytes[8] != 0;
        let n_layers = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
        let mut pos = 13usize;
        let layers = (0..n_layers)
            .map(|_| {
                Ok(Dsv4LayerPageState {
                    staging: read_bf16(&mut pos, bytes)?,
                    dsa_data: read_bytes(&mut pos, bytes)?,
                    dsa_scale: read_bytes(&mut pos, bytes)?,
                    overlap_kv: read_bf16(&mut pos, bytes)?,
                    overlap_score: read_bf16(&mut pos, bytes)?,
                    idx_overlap_kv: read_bf16(&mut pos, bytes)?,
                    idx_overlap_score: read_bf16(&mut pos, bytes)?,
                    ring: read_bf16(&mut pos, bytes)?,
                    pending_kv: read_bf16(&mut pos, bytes)?,
                    pending_score: read_bf16(&mut pos, bytes)?,
                    idx_pending_kv: read_bf16(&mut pos, bytes)?,
                    idx_pending_score: read_bf16(&mut pos, bytes)?,
                    tail_staging: read_bf16(&mut pos, bytes)?,
                    tail_dsa_data: read_bytes(&mut pos, bytes)?,
                    tail_dsa_scale: read_bytes(&mut pos, bytes)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(pos == bytes.len(), "prefix-state entry has trailing bytes");
        Ok(Self {
            page_index,
            boundary,
            layers,
        })
    }

    pub fn host_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.dsa_data.len()
                    + l.dsa_scale.len()
                    + l.tail_dsa_data.len()
                    + l.tail_dsa_scale.len()
                    + (l.staging.len()
                        + l.overlap_kv.len()
                        + l.overlap_score.len()
                        + l.idx_overlap_kv.len()
                        + l.idx_overlap_score.len()
                        + l.ring.len()
                        + l.pending_kv.len()
                        + l.pending_score.len()
                        + l.idx_pending_kv.len()
                        + l.idx_pending_score.len()
                        + l.tail_staging.len())
                        * 2
            })
            .sum()
    }
}

/// At most this many captures may be pending (fence not yet polled) per
/// executor — the queue depth the publish path budgets for.
pub const MAX_PENDING_PREFIX_CAPTURES: usize = 2;

/// A page captured from a slot's KV, not yet published to the content-keyed
/// pool. The radix dedup between capture and publish can retarget or cancel
/// it: `confirm`/`cancel_provisional`/`repair` track that, and publish skips
/// any page that did not survive.
pub struct PendingPrefixPage {
    source_page: u32,
    target_page: u32,
    confirmed: bool,
    cancelled: bool,
    entry: Dsv4PrefixPageEntry,
    frontier_tail: Option<Vec<u32>>,
}

impl PendingPrefixPage {
    pub fn new(
        source_page: u32,
        target_page: u32,
        entry: Dsv4PrefixPageEntry,
        frontier_tail: Option<Vec<u32>>,
    ) -> Self {
        Self {
            source_page,
            target_page,
            confirmed: false,
            cancelled: false,
            entry,
            frontier_tail,
        }
    }

    pub fn source_page(&self) -> u32 {
        self.source_page
    }

    pub fn target_page(&self) -> u32 {
        self.target_page
    }

    pub fn is_confirmed(&self) -> bool {
        self.confirmed
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    pub fn entry(&self) -> &Dsv4PrefixPageEntry {
        &self.entry
    }

    pub fn frontier_tail(&self) -> Option<&[u32]> {
        self.frontier_tail.as_deref()
    }

    pub fn confirm(&mut self, pages: &[u32]) {
        self.confirmed |= pages.contains(&self.target_page);
    }

    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    pub fn cancel_provisional(&mut self, pages: &[u32]) {
        self.cancelled |= !self.confirmed && pages.contains(&self.source_page);
    }

    pub fn repair(&mut self, canonical: u32, own: u32, canonical_exists: bool) {
        if self.source_page != own || self.cancelled {
            return;
        }
        if canonical_exists {
            self.cancelled = true;
        } else {
            self.target_page = canonical;
            self.confirmed = true;
        }
    }
}

/// A capture is publishable only while the slot's content epoch still matches
/// the one captured under.
pub fn capture_epoch_matches(captured: u64, current: Option<u64>) -> bool {
    current == Some(captured)
}

/// A captured page may not publish onto a target that already holds different
/// content.
pub fn rekey_target_conflicts(source_page: u32, target_page: u32, target_exists: bool) -> bool {
    source_page != target_page && target_exists
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn sample_entry() -> Dsv4PrefixPageEntry {
        Dsv4PrefixPageEntry {
            page_index: 7,
            boundary: true,
            layers: vec![
                Dsv4LayerPageState {
                    staging: vec![bf16::from_f32(1.0), bf16::from_f32(-2.5)],
                    dsa_data: vec![0xAB, 0xCD],
                    ring: vec![bf16::from_f32(3.0)],
                    ..Default::default()
                },
                Dsv4LayerPageState::default(),
            ],
        }
    }

    #[test]
    fn codec_round_trips_and_sizes_the_buffer() {
        let entry = sample_entry();
        let bytes = entry.to_bytes();
        // 4 magic + 4 page_index + 1 boundary + 4 n_layers + per layer 15
        // sections × 4-byte len prefixes + the payload itself.
        let expected = 13 + 2 * 15 * 4 + entry.host_bytes();
        assert_eq!(bytes.len(), expected);
        let back = Dsv4PrefixPageEntry::from_bytes(&bytes).unwrap();
        assert_eq!(back.page_index, 7);
        assert!(back.boundary);
        assert_eq!(back.layers.len(), 2);
        assert_eq!(back.layers[0].staging, entry.layers[0].staging);
        assert_eq!(back.layers[0].dsa_data, entry.layers[0].dsa_data);
        assert_eq!(back.layers[0].ring, entry.layers[0].ring);
        assert_eq!(back.layers[1], Dsv4LayerPageState::default());
    }

    #[test]
    fn codec_rejects_truncated_and_trailing_bytes() {
        let bytes = sample_entry().to_bytes();
        assert!(Dsv4PrefixPageEntry::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut bad = bytes.clone();
        bad[0] ^= 0xFF; // break the magic
        assert!(Dsv4PrefixPageEntry::from_bytes(&bad).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(Dsv4PrefixPageEntry::from_bytes(&trailing).is_err());
    }

    #[test]
    fn pending_page_lifecycle_tracks_dedup() {
        let entry = sample_entry();
        let mut page = PendingPrefixPage::new(10, 10, entry, None);
        assert!(!page.is_confirmed() && !page.is_cancelled());

        // Confirmation sticks; an unrelated page list does not.
        page.confirm(&[11]);
        assert!(!page.is_confirmed());
        page.confirm(&[10]);
        assert!(page.is_confirmed());

        // Provisional cancel fires only while unconfirmed and the SOURCE page
        // was freed.
        let mut provisional = PendingPrefixPage::new(10, 10, sample_entry(), None);
        provisional.cancel_provisional(&[10]);
        assert!(provisional.is_cancelled());
        let mut confirmed = PendingPrefixPage::new(10, 10, sample_entry(), None);
        confirmed.confirm(&[10]);
        confirmed.cancel_provisional(&[10]);
        assert!(!confirmed.is_cancelled());

        // Repair: canonical exists → cancel; otherwise adopt the canonical id
        // as confirmed. A page whose source was already retargeted is untouched.
        let mut adopt = PendingPrefixPage::new(10, 10, sample_entry(), None);
        adopt.repair(20, 10, false);
        assert_eq!(adopt.target_page(), 20);
        assert!(adopt.is_confirmed() && !adopt.is_cancelled());
        let mut clash = PendingPrefixPage::new(10, 10, sample_entry(), None);
        clash.repair(20, 10, true);
        assert!(clash.is_cancelled());
        let mut foreign = PendingPrefixPage::new(10, 10, sample_entry(), None);
        foreign.repair(20, 11, false);
        assert_eq!(foreign.target_page(), 10);
    }

    #[test]
    fn epoch_and_rekey_predicates() {
        assert!(capture_epoch_matches(5, Some(5)));
        assert!(!capture_epoch_matches(5, Some(6)));
        assert!(!capture_epoch_matches(5, None));
        assert!(rekey_target_conflicts(1, 2, true));
        assert!(!rekey_target_conflicts(1, 2, false));
        assert!(!rekey_target_conflicts(1, 1, true));
    }
}
