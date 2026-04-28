use super::*;

#[test]
fn test_name_mapping() {
    assert_eq!(
        map_gguf_name("token_embd.weight"),
        "model.embed_tokens.weight"
    );
    assert_eq!(map_gguf_name("output_norm.weight"), "model.norm.weight");
    assert_eq!(map_gguf_name("output.weight"), "lm_head.weight");
    assert_eq!(
        map_gguf_name("blk.0.attn_q.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        map_gguf_name("blk.12.ffn_gate.weight"),
        "model.layers.12.mlp.gate_proj.weight"
    );
    assert_eq!(
        map_gguf_name("blk.5.attn_norm.weight"),
        "model.layers.5.input_layernorm.weight"
    );
}

#[test]
fn test_dequant_q8_0_roundtrip() {
    // Manually construct a Q8_0 block: scale=0.5, values=[0,1,2,...,31]
    let scale_f16 = f16::from_f32(0.5);
    let scale_bytes = scale_f16.to_le_bytes();
    let mut block = Vec::new();
    block.extend_from_slice(&scale_bytes);
    for i in 0..32u8 {
        block.push(i);
    }

    let result = dequant_q8_0(&block, 32);
    assert_eq!(result.len(), 32);
    // First element: 0 * 0.5 = 0.0
    assert_eq!(result[0], bf16::from_f32(0.0));
    // Last element: 31 * 0.5 = 15.5
    assert!((result[31].to_f32() - 15.5).abs() < 0.1);
}

#[test]
fn test_dequant_q4_0_roundtrip() {
    // Q4_0 block: scale=1.0, 16 bytes of packed nibbles
    let scale_f16 = f16::from_f32(1.0);
    let scale_bytes = scale_f16.to_le_bytes();
    let mut block = Vec::new();
    block.extend_from_slice(&scale_bytes);
    // Pack: lo=8 (→0 after -8 offset), hi=9 (→1 after -8)
    block.extend(std::iter::repeat_n(0x98u8, 16)); // hi=9, lo=8 × 16 bytes

    let result = dequant_q4_0(&block, 32);
    assert_eq!(result.len(), 32);
    // lo = (8 - 8) * 1.0 = 0.0
    assert_eq!(result[0], bf16::from_f32(0.0));
    // hi = (9 - 8) * 1.0 = 1.0
    assert!((result[1].to_f32() - 1.0).abs() < 0.1);
}

#[test]
fn parse_only_gguf_quant_types_fail_explicitly_in_dequant_path() {
    for dtype in [
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_1,
        GgmlType::Q2_K,
        GgmlType::Q8_K,
    ] {
        let err = dequant_to_bf16(&[], dtype, 0).unwrap_err().to_string();
        assert!(
            err.contains("Dequant not yet implemented"),
            "unexpected error for {dtype:?}: {err}"
        );
    }
}

#[test]
fn test_decode_scale_min_k4_matches_ggml_layout() {
    let mut scales = [0u8; 12];
    scales[0] = 0b1000_0000;
    scales[1] = 0b0010_1010;
    scales[4] = 0b1100_0000;
    scales[5] = 0b0001_0001;
    scales[8] = 0x75;

    assert_eq!(decode_scale_min_k4(&scales, 1), (42, 17));
    assert_eq!(decode_scale_min_k4(&scales, 4), (37, 55));
}

#[test]
#[ignore = "requires downloaded GGUF model"]
fn test_parse_real_gguf() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
    );
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping: {path} not found");
        return;
    }

    let gguf = GgufFile::open(path).expect("Failed to parse GGUF");
    assert!(gguf.version >= 2);
    assert!(!gguf.tensors.is_empty());

    // Check architecture metadata
    let arch = gguf.architecture().expect("Missing architecture metadata");
    assert!(
        arch == "qwen3" || arch == "qwen2" || arch == "llama",
        "Unexpected architecture: {arch}"
    );

    // Check tensor count
    eprintln!("Tensors: {}", gguf.tensors.len());
    eprintln!("Architecture: {arch}");

    // Verify name mapping works for first few tensors
    for (name, info) in gguf.tensors.iter().take(5) {
        let hf_name = map_gguf_name(name);
        eprintln!("  {name} → {hf_name} ({:?}, {:?})", info.dtype, info.shape);
    }

    // Try reading and dequantizing one tensor
    let first_name = gguf.tensors.keys().next().unwrap();
    let bf16_data = gguf
        .read_tensor_bf16(first_name)
        .expect("Failed to dequant");
    let info = &gguf.tensors[first_name];
    assert_eq!(bf16_data.len(), info.numel());
    eprintln!(
        "Dequantized '{}': {} elements, first={}, last={}",
        first_name,
        bf16_data.len(),
        bf16_data[0],
        bf16_data[bf16_data.len() - 1]
    );
}
