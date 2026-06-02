//! CRC32C with hardware acceleration where the target exposes it.
//!
//! CRC32C is used as a fast integrity check for KV-tier object reads. It is not
//! used as the content address: persisted identity still uses the wider
//! [`crate::types::BlockFingerprint`] hash.

use std::sync::OnceLock;

const CRC32C_POLY_REVERSED: u32 = 0x82f6_3b78;

pub fn checksum(bytes: &[u8]) -> u32 {
    extend(0, bytes)
}

pub fn extend(seed: u32, bytes: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("crc") {
            return unsafe { extend_aarch64(seed, bytes) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.2") {
            return unsafe { extend_x86_sse42(seed, bytes) };
        }
    }

    extend_slicing_by_8(seed, bytes)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn extend_aarch64(seed: u32, bytes: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd, __crc32cw};

    let mut crc = !seed;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("chunk size is 8"));
        crc = __crc32cd(crc, word);
    }
    let mut tail = chunks.remainder();
    if tail.len() >= 4 {
        let (word, rest) = tail.split_at(4);
        crc = __crc32cw(
            crc,
            u32::from_le_bytes(word.try_into().expect("chunk size is 4")),
        );
        tail = rest;
    }
    for &byte in tail {
        crc = __crc32cb(crc, byte);
    }
    !crc
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "sse4.2")]
unsafe fn extend_x86_sse42(seed: u32, bytes: &[u8]) -> u32 {
    use std::arch::x86::{_mm_crc32_u8, _mm_crc32_u32};

    let mut crc = !seed;
    let mut chunks = bytes.chunks_exact(4);
    for chunk in &mut chunks {
        let word = u32::from_le_bytes(chunk.try_into().expect("chunk size is 4"));
        crc = _mm_crc32_u32(crc, word);
    }
    for &byte in chunks.remainder() {
        crc = _mm_crc32_u8(crc, byte);
    }
    !crc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn extend_x86_sse42(seed: u32, bytes: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};

    let mut crc = u64::from(!seed);
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("chunk size is 8"));
        crc = _mm_crc32_u64(crc, word);
    }
    let mut crc32 = crc as u32;
    for &byte in chunks.remainder() {
        crc32 = _mm_crc32_u8(crc32, byte);
    }
    !crc32
}

fn extend_slicing_by_8(seed: u32, bytes: &[u8]) -> u32 {
    let tables = slicing_tables();
    let mut crc = !seed;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("chunk size is 8"));
        let low = (word as u32) ^ crc;
        let high = (word >> 32) as u32;
        crc = tables[7][(low & 0xff) as usize]
            ^ tables[6][((low >> 8) & 0xff) as usize]
            ^ tables[5][((low >> 16) & 0xff) as usize]
            ^ tables[4][((low >> 24) & 0xff) as usize]
            ^ tables[3][(high & 0xff) as usize]
            ^ tables[2][((high >> 8) & 0xff) as usize]
            ^ tables[1][((high >> 16) & 0xff) as usize]
            ^ tables[0][((high >> 24) & 0xff) as usize];
    }
    for &byte in chunks.remainder() {
        crc = tables[0][((crc ^ u32::from(byte)) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

fn slicing_tables() -> &'static [[u32; 256]; 8] {
    static TABLES: OnceLock<[[u32; 256]; 8]> = OnceLock::new();
    TABLES.get_or_init(make_slicing_tables)
}

fn make_slicing_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    for i in 0..256 {
        let mut crc = i as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 0 {
                crc >> 1
            } else {
                (crc >> 1) ^ CRC32C_POLY_REVERSED
            };
        }
        tables[0][i] = crc;
    }
    for table_idx in 1..8 {
        for i in 0..256 {
            let crc = tables[table_idx - 1][i];
            tables[table_idx][i] = tables[0][(crc & 0xff) as usize] ^ (crc >> 8);
        }
    }
    tables
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_matches_castagnoli_check_vector() {
        assert_eq!(checksum(b"123456789"), 0xe306_9283);
    }

    #[test]
    fn crc32c_extend_is_incremental() {
        let first = extend(0, b"1234");
        let second = extend(first, b"56789");
        assert_eq!(second, checksum(b"123456789"));
    }
}
