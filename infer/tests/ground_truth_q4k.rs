//! Cross-check our Q4_K dequant vs Python ground truth (llama.cpp canonical layout).
//! See tests/carnice_tensor_probe.rs for how the Python reference was computed.
//!
//! Enable with:
//!   cargo test --release --test ground_truth_q4k -- --nocapture --ignored

#![cfg(feature = "cuda")]

use infer::gguf::GgufFile;

fn path_4b() -> String {
    std::env::var("INFER_QWEN35_4B_GGUF_PATH")
        .unwrap_or_else(|_| "models/Qwen3.5-4B-GGUF-test/Qwen3.5-4B-Q4_K_M.gguf".to_string())
}

fn check(label: &str, gguf: &GgufFile, name: &str, expected: &[(usize, f32)]) {
    let bf16_vec = gguf.read_tensor_bf16(name).expect("read");
    println!("\n[{label}] {name}");
    for &(idx, want) in expected {
        let got = bf16_vec[idx].to_f32();
        let diff = (got - want).abs();
        println!("  [{idx}] got={got:.6} want={want:.6} diff={diff:.2e}");
        assert!(
            diff < 1e-3,
            "{label} mismatch at [{idx}]: got={got}, want={want}"
        );
    }
    println!("  ✓ matches llama.cpp ground truth");
}

#[test]
#[ignore = "requires Qwen3-4B Q4K GGUF weights (set INFER_QWEN3_Q4K_PATH)"]
fn dequant_matches_llama_cpp_ground_truth() {
    let gguf = GgufFile::open(&path_4b()).expect("open 4B gguf");

    // Q4_K — first superblock of blk.0.ffn_gate (Python-computed reference).
    check(
        "Q4_K",
        &gguf,
        "blk.0.ffn_gate.weight",
        &[
            (0, -0.005418),
            (1, 0.008783),
            (2, -0.090627),
            (32, -0.022258),
            (33, 0.007141),
            (34, 0.036541),
        ],
    );

    // Q6_K — first superblock of blk.0.ffn_down.
    check(
        "Q6_K",
        &gguf,
        "blk.0.ffn_down.weight",
        &[
            (0, 0.116526),
            (1, 0.166465),
            (2, -0.058263),
            (32, -0.025250),
            (33, 0.029459),
            (34, 0.037875),
        ],
    );

    // Q5_K — first superblock of blk.0.attn_qkv.
    check(
        "Q5_K",
        &gguf,
        "blk.0.attn_qkv.weight",
        &[
            (0, -0.001127),
            (1, -0.010327),
            (2, -0.005727),
            (32, 0.026377),
            (33, -0.009819),
            (34, -0.005098),
        ],
    );

    // Q8_0 — first two blocks of blk.0.ssm_alpha.
    check(
        "Q8_0",
        &gguf,
        "blk.0.ssm_alpha.weight",
        &[
            (0, -0.009217),
            (1, -0.005964),
            (2, -0.011928),
            (3, 0.013554),
            (4, -0.023855),
        ],
    );
}
