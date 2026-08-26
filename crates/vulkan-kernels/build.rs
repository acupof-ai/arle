//! GLSL → SPIR-V build step for the AIPC Vulkan lane.
//!
//! The main corpus is adapted at runtime from ggml-org/llama.cpp
//! `vulkan-shaders` @ d2462f8f. This build script never edits the vendored
//! directory; it compiles selected `.comp` files into `$OUT_DIR/vulkan-spv`.

use std::path::{Path, PathBuf};
use std::process::Command;

struct ShaderSpec {
    name: &'static str,
    source: &'static str,
    defines: &'static [(&'static str, &'static str)],
}

/// Shared `mul_mmq.comp` define set; only `DATA_A_*` differs per quant type.
/// Kept as macros rather than a helper fn because `ShaderSpec::defines` is a
/// `&'static [_]` in a `const` array.
macro_rules! mmq_defines {
    ($data_a:literal) => {
        &[
            ("FLOAT16", "1"),
            ("FLOAT_TYPE", "float16_t"),
            ("FLOAT_TYPEV2", "f16vec2"),
            ("FLOAT_TYPEV4", "f16vec4"),
            ("ACC_TYPE", "float"),
            ("ACC_TYPEV2", "vec2"),
            ($data_a, "1"),
            ("D_TYPE", "float"),
        ]
    };
}

const MMQ_DEFINES_Q4_K: &[(&str, &str)] = mmq_defines!("DATA_A_Q4_K");
const MMQ_DEFINES_Q5_K: &[(&str, &str)] = mmq_defines!("DATA_A_Q5_K");
const MMQ_DEFINES_Q6_K: &[(&str, &str)] = mmq_defines!("DATA_A_Q6_K");
const MMQ_DEFINES_Q8_0: &[(&str, &str)] = mmq_defines!("DATA_A_Q8_0");

/// Shared `mul_mm.comp` COOPMAT define set. Mirrors llama.cpp's
/// `matmul_shaders(fp16=true, coopmat=true, f16acc=false)` for the *unaligned*
/// `<quant>_f16` variant (`vulkan-shaders-gen.cpp:584`), which is the one whose
/// B operand is a plain `float16_t` row-major `[N][K]` — no `ALIGNED`/vec4
/// packing, so N (the token count) is unconstrained.
///
/// `LOAD_VEC_A` is `load_vec_quant`, and it is NOT cosmetic: `load_a_to_shmem`
/// computes `buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2`, so a wrong
/// value silently aliases shared-memory rows. Q4_K/Q5_K/Q8_0 dequantize 4
/// values per invocation (`LOAD_VEC_A = 4`), Q6_K only 2.
macro_rules! mm_coopmat_defines {
    ($data_a:literal, $load_vec_a:literal) => {
        &[
            ("FLOAT16", "1"),
            ("FLOAT_TYPE", "float16_t"),
            ("FLOAT_TYPEV2", "f16vec2"),
            ("FLOAT_TYPEV4", "f16vec4"),
            ("ACC_TYPE", "float"),
            ("ACC_TYPEV2", "vec2"),
            ("COOPMAT", "1"),
            ($data_a, "1"),
            ("LOAD_VEC_A", $load_vec_a),
            ("B_TYPE", "float16_t"),
            ("D_TYPE", "float"),
        ]
    };
}

const MM_CM_DEFINES_Q4_K: &[(&str, &str)] = mm_coopmat_defines!("DATA_A_Q4_K", "4");
const MM_CM_DEFINES_Q5_K: &[(&str, &str)] = mm_coopmat_defines!("DATA_A_Q5_K", "4");
const MM_CM_DEFINES_Q6_K: &[(&str, &str)] = mm_coopmat_defines!("DATA_A_Q6_K", "2");
const MM_CM_DEFINES_Q8_0: &[(&str, &str)] = mm_coopmat_defines!("DATA_A_Q8_0", "4");

/// Shared `mul_mat_vec.comp` define set for the two NVFP4 GEMVs. `MUL_MAT_ID`
/// is the ONLY difference between the plain and the fused-expert variant, so it
/// is passed as an extra rather than duplicated into a second literal list.
///
/// Note `B_TYPE = float`, not `block_q8_1_x4`: unlike every other GEMV this
/// crate registers, the NVFP4 pair runs `mul_mat_vec.comp` (dequantize-to-float
/// then `dot`), not `mul_mat_vecq.comp` (integer dot). See the `ShaderSpec`
/// comment below for why.
macro_rules! nvfp4_gemv_defines {
    ($($extra:expr,)*) => {
        &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_NVFP4", "1"),
            ("B_TYPE", "float"),
            ("B_TYPEV2", "vec2"),
            ("B_TYPEV4", "vec4"),
            ("D_TYPE", "float"),
            ("USE_SUBGROUP_ADD", "1"),
            $($extra,)*
        ]
    };
}

const NVFP4_GEMV_DEFINES: &[(&str, &str)] = nvfp4_gemv_defines!();
const NVFP4_GEMV_ID_DEFINES: &[(&str, &str)] = nvfp4_gemv_defines!(("MUL_MAT_ID", "1"),);

const VENDORED: &[ShaderSpec] = &[
    ShaderSpec {
        name: "mul_mat_vec_iq2_xxs",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vec_iq2_xxs.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_IQ2_XXS", "1"),
            ("B_TYPE", "float"),
            ("B_TYPEV2", "vec2"),
            ("B_TYPEV4", "vec4"),
            ("D_TYPE", "float"),
        ],
    },
    ShaderSpec {
        name: "mul_mat_vec_q2_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vec_q2_k.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_Q2_K", "1"),
            ("B_TYPE", "float"),
            ("B_TYPEV2", "vec2"),
            ("B_TYPEV4", "vec4"),
            ("D_TYPE", "float"),
        ],
    },
    // The decode GEMVs run a BLOCK_SIZE=64 workgroup pinned to a single 64-wide
    // subgroup (see `SPEC_GEMV_K_Q8_1` + `Kernel::required_subgroup_size`), so
    // `USE_SUBGROUP_ADD` collapses the cross-lane row reduction to ONE hardware
    // `subgroupAdd` instead of a 6-step barrier'd shmem tree. This mirrors
    // llama.cpp's `SHADER_REDUCTION_MODE_SUBGROUP` choice for the q8_1 decode
    // pipelines on AMD non-GCN (use_subgroups = subgroup_arithmetic && !GCN).
    ShaderSpec {
        name: "mul_mat_vecq_q4_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_Q4_K", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    ShaderSpec {
        name: "mul_mat_vecq_q5_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_Q5_K", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    ShaderSpec {
        name: "mul_mat_vecq_q6_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_Q6_K", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    ShaderSpec {
        name: "mul_mat_vecq_q8_0",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_Q8_0", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    // MXFP4 (E8M0 shared exponent + 16 packed E2M1 nibbles per 32 values,
    // 17 B/block). Unsloth's "UD-Q*_XL" dynamic quants store the routed
    // experts and most attention projections in MXFP4 — 90% of a
    // Qwen3.5-122B-A10B's elements — so without this variant that checkpoint
    // has no GEMV at all. The vendored shader already carries the
    // `DATA_A_MXFP4` arms (`mul_mat_vecq_funcs.glsl:19,115,135`,
    // `types.glsl:1717`), so this is a define, not a new kernel.
    ShaderSpec {
        name: "mul_mat_vecq_mxfp4",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_MXFP4", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    // Fused MoE expert GEMV (`mul_mat_vec_id`) — the same `mul_mat_vecq.comp`
    // body compiled with `MUL_MAT_ID=1`, which swaps the batch-offset push tail
    // for the expert-id contract (`nei0/ne11/expert_i1/nbi1` + a 6th `IDS`
    // binding) so ONE dispatch runs a token through ALL its top-k routed experts
    // (gl_WorkGroupID.y = expert slot, expert_id = data_ids[...]). Collapses the
    // per-layer 8×3 per-expert GEMVs into 3 dispatches. The expert tensors in the
    // 35B-A3B are Q4_K/Q5_K/Q6_K/Q8_0; the 122B-A10B adds MXFP4.
    ShaderSpec {
        name: "mul_mat_vec_id_q4_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_Q4_K", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("MUL_MAT_ID", "1"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    ShaderSpec {
        name: "mul_mat_vec_id_q5_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_Q5_K", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("MUL_MAT_ID", "1"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    ShaderSpec {
        name: "mul_mat_vec_id_q6_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_Q6_K", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("MUL_MAT_ID", "1"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    ShaderSpec {
        name: "mul_mat_vec_id_q8_0",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_Q8_0", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("MUL_MAT_ID", "1"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    // The routed experts themselves. In the 122B-A10B every `ffn_gate_exps` /
    // `ffn_up_exps` (48 layers) and most `ffn_down_exps` are MXFP4, so this is
    // the variant the MoE hot path actually dispatches.
    ShaderSpec {
        name: "mul_mat_vec_id_mxfp4",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("DATA_A_MXFP4", "1"),
            ("D_TYPE", "float"),
            ("ACC_TYPE", "float"),
            ("MUL_MAT_ID", "1"),
            ("USE_SUBGROUP_ADD", "1"),
        ],
    },
    // NVFP4 (four UE4M3 sub-block scales, one per 16 values, then 32 packed
    // E2M1 nibble bytes — 36 B per 64-value block; ggml-common.h:211). The
    // routed experts of Qwen3.8-Flash-Next are NVFP4 and nothing else is, so
    // without these two variants that checkpoint's MoE has no GEMV.
    //
    // These are the ONLY GEMVs here built from `mul_mat_vec.comp` rather than
    // `mul_mat_vecq.comp`, and that is forced, not a preference:
    // `mul_mat_vecq_funcs.glsl` has no `DATA_A_NVFP4` arm and cannot get one by
    // a define. Its `mmvq_dot_product` contract is ONE `get_dm(ib)` scale per
    // dot-product group, and it walks A in `QUANT_K_Q8_1`-sized (32-value)
    // steps; NVFP4 carries FOUR scales per 64-value block, so an integer-dot
    // arm would need a different accumulator decomposition, i.e. new vendored
    // shader code on the hot path. Upstream llama.cpp made the same call —
    // NVFP4 appears in `dequant_funcs.glsl` and `mul_mm_funcs.glsl`, never in
    // `mul_mat_vecq_funcs.glsl`.
    //
    // Consequence for callers: B is a plain f32 activation vector, NOT
    // `block_q8_1_x4`. The MoE path skips the `q8_1_quantize` dispatch for
    // NVFP4 experts, and feeding these pipelines q8_1 bytes is silent garbage.
    ShaderSpec {
        name: "mul_mat_vec_nvfp4",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vec.comp",
        defines: NVFP4_GEMV_DEFINES,
    },
    ShaderSpec {
        name: "mul_mat_vec_id_nvfp4",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vec.comp",
        defines: NVFP4_GEMV_ID_DEFINES,
    },
    // Batched prefill GEMM (`mul_mmq`) — the integer-dot-product tiled matmul
    // that consumes the SAME `block_q8_1_x4` activations the decode GEMVs
    // already produce (`block_q8_1_x4` and `block_q8_1_x4_packed128` are
    // byte-identical 144-byte blocks), so one `q8_1_quantize` dispatch feeds
    // both lanes. `FLOAT16` picks the f16 shmem cache (halves the `block_a_cache`
    // / `block_b_cache` scale footprint) while `ACC_TYPE=float` keeps the f32
    // accumulator — the non-`f16acc` variant llama.cpp registers for AMD.
    ShaderSpec {
        name: "mul_mmq_q4_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mmq.comp",
        defines: MMQ_DEFINES_Q4_K,
    },
    ShaderSpec {
        name: "mul_mmq_q5_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mmq.comp",
        defines: MMQ_DEFINES_Q5_K,
    },
    ShaderSpec {
        name: "mul_mmq_q6_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mmq.comp",
        defines: MMQ_DEFINES_Q6_K,
    },
    ShaderSpec {
        name: "mul_mmq_q8_0",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mmq.comp",
        defines: MMQ_DEFINES_Q8_0,
    },
    // Batched prefill GEMM on the MATRIX CORES (`mul_mm.comp` + `COOPMAT`). The
    // 8060S advertises `VK_KHR_cooperative_matrix` with an f16xf16->f32 subgroup
    // tile, and on a dense 27B that path is worth 3.32x over the integer-dot
    // `mul_mmq` fallback (llama.cpp on this box: 61.96 t/s vs 18.64 t/s with
    // `GGML_VK_DISABLE_COOPMAT=1`). Unlike `mul_mmq` the B operand is f16, not
    // q8_1_x4 — the shader dequantizes A into shared memory and issues
    // `coopMatMulAdd`, so the activation side is an `f16_kv_pack` away, not a
    // `q8_1_quantize`. Registered unconditionally; `VulkanContext::coopmat()`
    // decides at runtime whether these pipelines are ever built.
    ShaderSpec {
        name: "mul_mm_cm_q4_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mm.comp",
        defines: MM_CM_DEFINES_Q4_K,
    },
    ShaderSpec {
        name: "mul_mm_cm_q5_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mm.comp",
        defines: MM_CM_DEFINES_Q5_K,
    },
    ShaderSpec {
        name: "mul_mm_cm_q6_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mm.comp",
        defines: MM_CM_DEFINES_Q6_K,
    },
    ShaderSpec {
        name: "mul_mm_cm_q8_0",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mm.comp",
        defines: MM_CM_DEFINES_Q8_0,
    },
    ShaderSpec {
        name: "rms_norm",
        source: "vendor/llama.cpp/vulkan-shaders/rms_norm.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("A_TYPE", "float"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
        ],
    },
    ShaderSpec {
        name: "rope_neox",
        source: "vendor/llama.cpp/vulkan-shaders/rope_neox.comp",
        defines: &[("A_TYPE", "float"), ("ROPE_D_TYPE", "float")],
    },
    ShaderSpec {
        name: "rope_norm",
        source: "vendor/llama.cpp/vulkan-shaders/rope_norm.comp",
        defines: &[("A_TYPE", "float"), ("ROPE_D_TYPE", "float")],
    },
    ShaderSpec {
        name: "silu",
        source: "vendor/llama.cpp/vulkan-shaders/silu.comp",
        defines: &[("A_TYPE", "float"), ("D_TYPE", "float")],
    },
    ShaderSpec {
        name: "swiglu",
        source: "vendor/llama.cpp/vulkan-shaders/swiglu.comp",
        defines: &[("A_TYPE", "float"), ("D_TYPE", "float")],
    },
    ShaderSpec {
        name: "add",
        source: "vendor/llama.cpp/vulkan-shaders/add.comp",
        defines: &[
            ("A_TYPE", "float"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"),
            ("ADD_RMS", "0"),
        ],
    },
    ShaderSpec {
        name: "get_rows",
        source: "vendor/llama.cpp/vulkan-shaders/get_rows.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("TEMP_TYPE", "FLOAT_TYPE"),
            ("DATA_A_F32", "1"),
            ("B_TYPE", "int"),
            ("D_TYPE", "float"),
        ],
    },
    ShaderSpec {
        name: "soft_max",
        source: "vendor/llama.cpp/vulkan-shaders/soft_max.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("A_TYPE", "float"),
            ("B_TYPE", "float"),
            ("D_TYPE", "float"),
        ],
    },
    ShaderSpec {
        name: "argmax",
        source: "vendor/llama.cpp/vulkan-shaders/argmax.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("A_TYPE", "float"),
            ("D_TYPE", "int"),
        ],
    },
    ShaderSpec {
        name: "flash_attn",
        source: "vendor/llama.cpp/vulkan-shaders/flash_attn.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("FLOAT_TYPEV2", "vec2"),
            ("FLOAT_TYPEV4", "vec4"),
            ("ACC_TYPE", "float"),
            ("ACC_TYPEV2", "vec2"),
            ("ACC_TYPEV4", "vec4"),
            ("Q_TYPE", "float"),
            ("D_TYPE", "float"),
            ("D_TYPEV4", "vec4"),
        ],
    },
];

const LOCAL: &[ShaderSpec] = &[
    ShaderSpec {
        name: "q8_1_quantize",
        source: "crates/vulkan-kernels/shaders/q8_1_quantize.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "geglu",
        source: "crates/vulkan-kernels/shaders/geglu.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "dsv4_prepare_qk",
        source: "crates/vulkan-kernels/shaders/dsv4_prepare_qk.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "dsv4_compressor_update",
        source: "crates/vulkan-kernels/shaders/dsv4_compressor_update.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "dsv4_csa_select",
        source: "crates/vulkan-kernels/shaders/dsv4_csa_select.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "dsv4_hybrid_attention",
        source: "crates/vulkan-kernels/shaders/dsv4_hybrid_attention.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "dsv4_swa_attention",
        source: "crates/vulkan-kernels/shaders/dsv4_swa_attention.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "dsv4_mhc",
        source: "crates/vulkan-kernels/shaders/dsv4_mhc.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "dsv4_output_inverse_rope",
        source: "crates/vulkan-kernels/shaders/dsv4_output_inverse_rope.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "swiglu_clamped",
        source: "crates/vulkan-kernels/shaders/swiglu_clamped.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "scaled_add",
        source: "crates/vulkan-kernels/shaders/scaled_add.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "sigmoid_mul",
        source: "crates/vulkan-kernels/shaders/sigmoid_mul.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "f16_kv_pack",
        source: "crates/vulkan-kernels/shaders/f16_kv_pack.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "qwen35_ssm_conv",
        source: "crates/vulkan-kernels/shaders/qwen35_ssm_conv.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "qwen35_gated_delta_net",
        source: "crates/vulkan-kernels/shaders/qwen35_gated_delta_net.comp",
        defines: &[
            ("FLOAT_TYPE", "float"),
            ("USE_SUBGROUP_CLUSTERED", "0"),
            ("USE_SUBGROUP_ADD", "0"),
        ],
    },
    ShaderSpec {
        name: "qwen36_router_topk",
        source: "crates/vulkan-kernels/shaders/qwen36_router_topk.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "qwen36_router_gemv",
        source: "crates/vulkan-kernels/shaders/qwen36_router_gemv.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "qwen36_moe_weighted_accum",
        source: "crates/vulkan-kernels/shaders/qwen36_moe_weighted_accum.comp",
        defines: &[],
    },
];

fn main() {
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=ARLE_VULKAN_GLSLC");
    println!("cargo:rerun-if-env-changed=PATH");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let shader_dir = workspace_root.join("vendor/llama.cpp/vulkan-shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("vulkan-spv");
    std::fs::create_dir_all(&out_dir).expect("create vulkan-spv out dir");
    println!("cargo:rustc-env=ARLE_VULKAN_SPV_DIR={}", out_dir.display());

    let shader_names: Vec<_> = VENDORED.iter().chain(LOCAL).map(|spec| spec.name).collect();
    let manifest = format!("{}\n", shader_names.join("\n"));
    std::fs::write(out_dir.join("registered-shaders.txt"), manifest)
        .expect("write registered shader manifest");

    for spec in VENDORED.iter().chain(LOCAL) {
        println!(
            "cargo:rerun-if-changed={}",
            workspace_root.join(spec.source).display()
        );
    }
    for include in [
        "types.glsl",
        "generic_head.glsl",
        "generic_binary_head.glsl",
        "generic_unary_head.glsl",
        "utils.glsl",
        "rope_params.glsl",
        "mul_mat_vec_base.glsl",
        "mul_mat_vec_iface.glsl",
        "mul_mat_vecq_funcs.glsl",
        "dequant_funcs.glsl",
        "dot_product_funcs.glsl",
        "flash_attn_base.glsl",
        "flash_attn_dequant.glsl",
        "flash_attn_mmq_funcs.glsl",
        "mul_mmq_shmem_types.glsl",
        "mul_mmq_funcs.glsl",
        "mul_mm_funcs.glsl",
        "mul_mm_id_funcs.glsl",
        "rope_head.glsl",
        "rope_funcs.glsl",
        "glu_head.glsl",
        "glu_main.glsl",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            shader_dir.join(include).display()
        );
    }

    let Some(glslc) = find_glslc() else {
        println!(
            "cargo:warning=vulkan-kernels: glslc not found (ARLE_VULKAN_GLSLC/VULKAN_SDK/PATH); \
             skipping SPIR-V compilation — typecheck-only lane, compile on the Vulkan box"
        );
        println!("cargo:rustc-env=ARLE_VULKAN_GLSLC_PRESENT=0");
        return;
    };
    println!("cargo:rustc-env=ARLE_VULKAN_GLSLC_PRESENT=1");

    for spec in VENDORED.iter().chain(LOCAL) {
        let src = workspace_root.join(spec.source);
        let dst = out_dir.join(format!("{}.spv", spec.name));
        let mut cmd = Command::new(&glslc);
        cmd.arg("-O")
            .arg("--target-env=vulkan1.2")
            .arg("-fshader-stage=compute")
            .arg("-I")
            .arg(&shader_dir);
        for (key, value) in spec.defines {
            cmd.arg(format!("-D{key}={value}"));
        }
        cmd.arg("-o").arg(&dst).arg(&src);
        match cmd.output() {
            Ok(output) if output.status.success() => {}
            Ok(output) => {
                let _ = std::fs::remove_file(&dst);
                panic!(
                    "vulkan-kernels: glslc failed for {} with status {}\nstdout:\n{}\nstderr:\n{}",
                    spec.name,
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Err(e) => panic!(
                "vulkan-kernels: failed to run {} for {}: {e}",
                glslc.display(),
                spec.name
            ),
        }
    }
}

fn find_glslc() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ARLE_VULKAN_GLSLC") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    // On Windows the binary is `glslc.exe`; check both names so a bare `glslc`
    // under VULKAN_SDK\bin or on PATH (e.g. MSYS2's glslc.exe) is still found.
    const NAMES: &[&str] = &["glslc", "glslc.exe"];
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        let bin = Path::new(&sdk).join("bin");
        for name in NAMES {
            let path = bin.join(name);
            if path.exists() {
                return Some(path);
            }
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            NAMES
                .iter()
                .map(|name| dir.join(name))
                .find(|candidate| candidate.exists())
        })
    })
}
