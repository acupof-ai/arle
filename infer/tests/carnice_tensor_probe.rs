//! Dump the first superblock of a real Q4_K_S tensor from Carnice and print
//! decoded (d, dmin, sub_scales, sub_mins, nibbles). Also dump the first few
//! dequantized values from the BF16 path so we can see if the magnitudes make
//! any sense at all.

#![cfg(feature = "cuda")]

use half::f16;
use infer::gguf::GgufFile;

fn model_path() -> String {
    std::env::var("INFER_CARNICE_PATH").unwrap_or_else(|_| "models/Carnice-27b-GGUF".to_string())
}

fn find_gguf(dir: &str) -> String {
    for entry in std::fs::read_dir(dir).expect("read dir") {
        let p = entry.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
            return p.to_string_lossy().into_owned();
        }
    }
    panic!("no .gguf file in {dir}");
}

#[test]
#[ignore]
fn probe_q4k_superblock() {
    let gguf_path = find_gguf(&model_path());
    let gguf = GgufFile::open(&gguf_path).expect("open gguf");

    let name = "blk.58.ffn_down.weight";
    let info = &gguf.tensors[name];
    println!("{name} dtype={:?} shape={:?}", info.dtype, info.shape);

    // blk.58.ffn_down in Carnice is actually Q6_K.
    let raw = gguf.read_tensor_q6k_packed(name).expect("packed");
    println!("packed bytes: {}", raw.len());

    // First superblock (row 0, superblock 0).
    let sb = &raw[0..144];
    let d = f16::from_le_bytes([sb[0], sb[1]]).to_f32();
    let dmin = f16::from_le_bytes([sb[2], sb[3]]).to_f32();
    println!("d={d:e}  dmin={dmin:e}");

    let scales_raw = &sb[4..16];
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for i in 0..4 {
        sc[i] = scales_raw[i] & 0x3F;
        mn[i] = scales_raw[i + 4] & 0x3F;
    }
    for i in 0..4 {
        sc[4 + i] = (scales_raw[i] >> 6) | ((scales_raw[8 + i] & 0x0F) << 2);
        mn[4 + i] = (scales_raw[i + 4] >> 6) | ((scales_raw[8 + i] >> 4) << 2);
    }
    println!("sub_scales={sc:?}  sub_mins={mn:?}");

    let qs = &sb[16..144];
    print!("first 32 nibbles: ");
    for i in 0..16 {
        print!("{:02x} ", qs[i]);
    }
    println!();

    // Dequant first 8 elements by hand
    println!("first 8 dequant values:");
    for i in 0..8 {
        let byte = qs[i / 2];
        let nib = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        let w = (nib as f32) * d * (sc[0] as f32) - dmin * (mn[0] as f32);
        println!("  element {i}: nib={nib} w={w:e}");
    }

    // Dequant via read_tensor_bf16 and dump first 8 elements of row 0.
    let bf16_data = gguf.read_tensor_bf16(name).expect("bf16 dequant");
    let ne0 = info.shape[0] as usize;
    println!("read_tensor_bf16: len={}  cols={ne0}", bf16_data.len());
    println!("row 0 first 8 from read_tensor_bf16:");
    for i in 0..8 {
        println!("  [{i}] = {}", bf16_data[i].to_f32());
    }

    // Sanity stats on row 0
    let row0: Vec<f32> = bf16_data[0..ne0].iter().map(|v| v.to_f32()).collect();
    let min = row0.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = row0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = row0.iter().sum::<f32>() / row0.len() as f32;
    let rms = (row0.iter().map(|v| v * v).sum::<f32>() / row0.len() as f32).sqrt();
    println!("row 0 stats: min={min:e}  max={max:e}  mean={mean:e}  rms={rms:e}");
}
