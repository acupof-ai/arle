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
    defines: &'static [&'static str],
}

const VENDORED: &[ShaderSpec] = &[
    ShaderSpec {
        name: "mul_mat_vec_iq2_xxs",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vec_iq2_xxs.comp",
        defines: &[
            "DATA_A_IQ2_XXS",
            "DATA_B_F32",
            "DATA_D_F32",
            "FLOAT_TYPE=float",
            "D_TYPE=float",
        ],
    },
    ShaderSpec {
        name: "mul_mat_vec_q2_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vec_q2_k.comp",
        defines: &[
            "DATA_A_Q2_K",
            "DATA_B_F32",
            "DATA_D_F32",
            "FLOAT_TYPE=float",
            "D_TYPE=float",
        ],
    },
    ShaderSpec {
        name: "mul_mat_vecq_q4_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            "DATA_A_Q4_K",
            "DATA_B_Q8_1",
            "DATA_D_F32",
            "FLOAT_TYPE=float",
            "D_TYPE=float",
        ],
    },
    ShaderSpec {
        name: "mul_mat_vecq_q5_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            "DATA_A_Q5_K",
            "DATA_B_Q8_1",
            "DATA_D_F32",
            "FLOAT_TYPE=float",
            "D_TYPE=float",
        ],
    },
    ShaderSpec {
        name: "mul_mat_vecq_q6_k",
        source: "vendor/llama.cpp/vulkan-shaders/mul_mat_vecq.comp",
        defines: &[
            "DATA_A_Q6_K",
            "DATA_B_Q8_1",
            "DATA_D_F32",
            "FLOAT_TYPE=float",
            "D_TYPE=float",
        ],
    },
    ShaderSpec {
        name: "rms_norm",
        source: "vendor/llama.cpp/vulkan-shaders/rms_norm.comp",
        defines: &["A_TYPE=float", "D_TYPE=float", "FLOAT_TYPE=float"],
    },
    ShaderSpec {
        name: "rope_neox",
        source: "vendor/llama.cpp/vulkan-shaders/rope_neox.comp",
        defines: &["A_TYPE=float", "D_TYPE=float", "FLOAT_TYPE=float"],
    },
    ShaderSpec {
        name: "rope_norm",
        source: "vendor/llama.cpp/vulkan-shaders/rope_norm.comp",
        defines: &["A_TYPE=float", "D_TYPE=float", "FLOAT_TYPE=float"],
    },
    ShaderSpec {
        name: "silu",
        source: "vendor/llama.cpp/vulkan-shaders/silu.comp",
        defines: &["A_TYPE=float", "D_TYPE=float", "FLOAT_TYPE=float"],
    },
    ShaderSpec {
        name: "swiglu",
        source: "vendor/llama.cpp/vulkan-shaders/swiglu.comp",
        defines: &[
            "A_TYPE=float",
            "B_TYPE=float",
            "D_TYPE=float",
            "FLOAT_TYPE=float",
        ],
    },
    ShaderSpec {
        name: "add",
        source: "vendor/llama.cpp/vulkan-shaders/add.comp",
        defines: &[
            "A_TYPE=float",
            "B_TYPE=float",
            "D_TYPE=float",
            "FLOAT_TYPE=float",
        ],
    },
    ShaderSpec {
        name: "get_rows",
        source: "vendor/llama.cpp/vulkan-shaders/get_rows.comp",
        defines: &["A_TYPE=float", "D_TYPE=float", "FLOAT_TYPE=float"],
    },
    ShaderSpec {
        name: "soft_max",
        source: "vendor/llama.cpp/vulkan-shaders/soft_max.comp",
        defines: &[
            "A_TYPE=float",
            "B_TYPE=float",
            "C_TYPE=float",
            "D_TYPE=float",
            "FLOAT_TYPE=float",
        ],
    },
    ShaderSpec {
        name: "argmax",
        source: "vendor/llama.cpp/vulkan-shaders/argmax.comp",
        defines: &["A_TYPE=float", "D_TYPE=uint", "FLOAT_TYPE=float"],
    },
    ShaderSpec {
        name: "flash_attn",
        source: "vendor/llama.cpp/vulkan-shaders/flash_attn.comp",
        defines: &[
            "A_TYPE=float",
            "B_TYPE=float",
            "D_TYPE=float",
            "FLOAT_TYPE=float",
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
        name: "gelu",
        source: "crates/vulkan-kernels/shaders/gelu.comp",
        defines: &[],
    },
    ShaderSpec {
        name: "geglu",
        source: "crates/vulkan-kernels/shaders/geglu.comp",
        defines: &[],
    },
];

fn main() {
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=ARLE_VULKAN_GLSLC");

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
        "mul_mat_vec_base.glsl",
        "mul_mat_vec_iface.glsl",
        "mul_mat_vecq_funcs.glsl",
        "dequant_funcs.glsl",
        "flash_attn_base.glsl",
        "flash_attn_dequant.glsl",
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
        return;
    };

    for spec in VENDORED.iter().chain(LOCAL) {
        let src = workspace_root.join(spec.source);
        let dst = out_dir.join(format!("{}.spv", spec.name));
        let mut cmd = Command::new(&glslc);
        cmd.arg("-O")
            .arg("-I")
            .arg(&shader_dir)
            .arg("-o")
            .arg(&dst)
            .arg(&src);
        for define in spec.defines {
            cmd.arg(format!("-D{define}"));
        }
        match cmd.status() {
            Ok(status) if status.success() => {}
            Ok(status) => {
                println!(
                    "cargo:warning=vulkan-kernels: glslc failed for {} with status {}; \
                     shader will report ShaderMissing until the specialization matrix is fixed",
                    spec.name, status
                );
                let _ = std::fs::remove_file(&dst);
            }
            Err(e) => {
                println!(
                    "cargo:warning=vulkan-kernels: failed to run {} for {}: {}; skipping",
                    glslc.display(),
                    spec.name,
                    e
                );
            }
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
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        let path = Path::new(&sdk).join("bin").join("glslc");
        if path.exists() {
            return Some(path);
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join("glslc"))
            .find(|candidate| candidate.exists())
    })
}
