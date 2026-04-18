//! Audit dtype distribution in the Carnice-27b GGUF to see which tensors
//! actually end up on which loader path. Skipped unless the model is present.
//!
//!   INFER_CARNICE_PATH=/path/to/models/Carnice-27b-GGUF \
//!       cargo test --release --test carnice_dtype_audit -- --nocapture --ignored

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;

use infer::gguf::GgufFile;

fn model_path() -> String {
    std::env::var("INFER_CARNICE_PATH").unwrap_or_else(|_| "models/Carnice-27b-GGUF".to_string())
}

#[test]
#[ignore = "requires Carnice GGUF weights (set INFER_CARNICE_PATH)"]
fn audit_carnice_dtypes() {
    // Group (dtype, rank) → count, total bytes.
    #[derive(Default, Debug)]
    struct Stat {
        count: usize,
        bytes_packed: usize,
        elems: usize,
    }

    let dir = model_path();
    let mut gguf_path = String::new();
    for entry in std::fs::read_dir(&dir).expect("read dir") {
        let p = entry.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
            gguf_path = p.to_string_lossy().into_owned();
            break;
        }
    }
    assert!(!gguf_path.is_empty(), "no .gguf file in {dir}");
    println!("auditing {gguf_path}");

    let gguf = GgufFile::open(&gguf_path).expect("open gguf");
    println!("tensors: {}", gguf.tensors.len());

    let mut groups: BTreeMap<String, Stat> = BTreeMap::new();

    for (name, info) in &gguf.tensors {
        let key = format!("{:?} rank={}", info.dtype, info.shape.len());
        let s = groups.entry(key).or_default();
        s.count += 1;
        s.bytes_packed += info.size_bytes();
        s.elems += info.numel();
        // Log a sample name for the first occurrence of each dtype to understand
        // what goes where.
        if s.count == 1 {
            println!(
                "  first {:?} tensor: {name} shape={:?}",
                info.dtype, info.shape
            );
        }
    }

    let mut total_packed = 0usize;
    let mut total_elems = 0usize;
    println!("\n── by dtype × rank ──");
    for (k, s) in &groups {
        println!(
            "  {k:30} count={:<4} elems={:<12} packed={:.3} GiB",
            s.count,
            s.elems,
            s.bytes_packed as f64 / (1 << 30) as f64
        );
        total_packed += s.bytes_packed;
        total_elems += s.elems;
    }
    println!(
        "TOTAL: {} tensors, {} params, {:.3} GiB packed",
        gguf.tensors.len(),
        total_elems,
        total_packed as f64 / (1 << 30) as f64
    );

    // Compute what our loader WOULD put on GPU under the current branching.
    let mut gpu_packed = 0u64;
    let mut gpu_bf16_expanded = 0u64;
    for info in gguf.tensors.values() {
        use infer::gguf::GgmlType;
        let packed_bytes = info.size_bytes() as u64;
        let bf16_bytes = (info.numel() as u64) * 2;
        let is_2d = info.shape.len() == 2;
        match info.dtype {
            GgmlType::BF16 | GgmlType::F16 | GgmlType::F32 => {
                // Current loader uploads as-is (BF16 after cast).
                gpu_bf16_expanded += bf16_bytes;
            }
            GgmlType::Q8_0 if is_2d => {
                gpu_packed += packed_bytes;
            }
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K if is_2d => {
                gpu_packed += packed_bytes;
            }
            // Everything else falls through to read_tensor_bf16 → BF16 on GPU.
            _ => {
                gpu_bf16_expanded += bf16_bytes;
            }
        }
    }
    println!("\nEXPECTED GPU weight residency with current loader:");
    println!(
        "  packed (Q4_K/Q8_0/etc)   : {:.3} GiB",
        gpu_packed as f64 / (1 << 30) as f64
    );
    println!(
        "  bf16-expanded fallback   : {:.3} GiB",
        gpu_bf16_expanded as f64 / (1 << 30) as f64
    );
    println!(
        "  sum                      : {:.3} GiB",
        (gpu_packed + gpu_bf16_expanded) as f64 / (1 << 30) as f64
    );
}
