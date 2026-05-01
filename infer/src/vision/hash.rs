use serde::{Deserialize, Serialize};

/// Stable 128-bit multimodal content identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct MmHash(pub [u8; 16]);

impl MmHash {
    #[must_use]
    pub fn as_bytes(self) -> [u8; 16] {
        self.0
    }
}

/// Synthetic token sequence reserved for replacing non-text media in prefix IDs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct VisionPadTokens(pub [u32; 8]);

/// Runtime identity for one media item after canonical decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ImageIdentity {
    pub hash: MmHash,
    pub pad_tokens: VisionPadTokens,
}

/// Hash canonical media bytes with MIME/domain separation.
///
/// Callers should pass bytes after protocol-level decoding and image
/// canonicalization. The optional MIME hint is still part of the identity so
/// ambiguous byte streams do not collide across media families.
#[must_use]
pub fn hash_canonical_media(bytes: &[u8], mime_hint: Option<&str>) -> MmHash {
    let mut h = blake3::Hasher::new();
    h.update(b"arle-mm-v1\x00");
    h.update(b"mime\x00");
    if let Some(mime) = mime_hint {
        h.update(&(mime.len() as u64).to_le_bytes());
        h.update(mime.as_bytes());
    } else {
        h.update(&0u64.to_le_bytes());
    }
    h.update(b"bytes\x00");
    h.update(&(bytes.len() as u64).to_le_bytes());
    h.update(bytes);

    let full = h.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&full.as_bytes()[..16]);
    MmHash(out)
}

/// Derive high-bit synthetic tokens from a multimodal hash.
///
/// `RadixCache` matches online prefixes by token path, so the path placeholder
/// must carry the full 128-bit media identity. Each output token stores 16 hash
/// bits plus a lane tag in a high-bit reserved range, making the mapping
/// injective without colliding with normal tokenizer vocabularies.
#[must_use]
pub fn pad_tokens_from_hash(hash: MmHash) -> VisionPadTokens {
    let mut tokens = [0u32; 8];
    for (idx, chunk) in hash.0.chunks_exact(2).enumerate() {
        let payload = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        tokens[idx] = 0x8000_0000 | ((idx as u32) << 16) | payload;
    }
    VisionPadTokens(tokens)
}

#[must_use]
pub fn hash_media_sequence<'a>(
    media: impl IntoIterator<Item = (&'a [u8], Option<&'a str>)>,
) -> Vec<ImageIdentity> {
    media
        .into_iter()
        .map(|(bytes, mime_hint)| {
            let hash = hash_canonical_media(bytes, mime_hint);
            ImageIdentity {
                hash,
                pad_tokens: pad_tokens_from_hash(hash),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_hash_is_deterministic() {
        let a = hash_canonical_media(b"pixels", Some("image/png"));
        let b = hash_canonical_media(b"pixels", Some("image/png"));
        let c = hash_canonical_media(b"pixels!", Some("image/png"));

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn canonical_hash_binds_mime_hint() {
        let png = hash_canonical_media(b"same-bytes", Some("image/png"));
        let jpeg = hash_canonical_media(b"same-bytes", Some("image/jpeg"));
        let unknown = hash_canonical_media(b"same-bytes", None);

        assert_ne!(png, jpeg);
        assert_ne!(png, unknown);
    }

    #[test]
    fn pad_tokens_use_reserved_high_bit() {
        let hash = hash_canonical_media(b"pixels", Some("image/png"));
        let pad = pad_tokens_from_hash(hash);

        assert_ne!(pad.0, [0; 8]);
        assert!(pad.0.iter().all(|token| token & 0x8000_0000 == 0x8000_0000));
    }

    #[test]
    fn pad_tokens_are_injective_over_full_hash() {
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        b[4] = 1;
        let pad_a = pad_tokens_from_hash(MmHash(a));
        let pad_b = pad_tokens_from_hash(MmHash(b));

        a[1] = 0x80;
        let pad_c = pad_tokens_from_hash(MmHash(a));

        assert_eq!(pad_a.0[0], pad_b.0[0]);
        assert_ne!(pad_a, pad_b);
        assert_ne!(pad_a, pad_c);
    }

    #[test]
    fn sequence_hashes_preserve_order() {
        let ids = hash_media_sequence([
            (&b"a"[..], Some("image/png")),
            (&b"b"[..], Some("image/png")),
        ]);

        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0].hash, ids[1].hash);
        assert_ne!(ids[0].pad_tokens, ids[1].pad_tokens);
    }
}
