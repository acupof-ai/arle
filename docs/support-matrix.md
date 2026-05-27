# ARLE Support Matrix

This document is the canonical support-status truth for `ARLE`.

It states what the repository currently supports, what is still limited, and
what validation exists for each area. If something is not listed as supported
here, do not assume it is supported just because it compiled locally.

State reflected here is based on repository evidence as of 2026-05-10.
Project framing lives in [index.md §Current Positioning](index.md#current-positioning).

---

## 1. Runtime Backends

| Backend | Status | Meaning |
| --- | --- | --- |
| CUDA | Supported | Primary serving path. Main runtime, scheduler, and benchmark focus. |
| Metal | Beta | Usable for local validation and live scheduler-backed serving. Qwen3.5 ships live prefix reuse via replayed compiled-path snapshots; `scripts/start_metal_serve.sh` is the canonical first-time Apple bring-up path. Qwen3.5-0.8B MLX 4bit single-request step-driver is measured at 305.5 tok/s on M4 Pro 20c for `1024/256`. The matched GGUF Q4_K_M exact default is 202.1 tok/s direct; the opt-in native-q4 load path reaches 236.7 tok/s direct / 239.8 tok/s step-driver and remains a separate exact packed-K-quant kernel/format gap. Metal is still missing full batched-decode parity with CUDA, especially on variable-length Qwen3.5 decode. |
| Metal DFlash | Beta | Apple Silicon speculative decode path. Default-on for Qwen3.5; benchmark before production use. |
| no-cuda / CPU-only | Development-oriented CPU backend | Build, test, and smoke-validation path for non-GPU logic. Not a production inference target. |

---

## 2. Platform Matrix

| Platform | Backend | Status | Validation |
| --- | --- | --- | --- |
| Linux x86_64 + NVIDIA GPU | CUDA | Supported | Release workflow builds CUDA artifacts; primary target. |
| macOS Apple Silicon | Metal | Beta | CI checks and tests Metal/no-cuda surfaces. |
| Linux/macOS without GPU | no-cuda | Development-oriented CPU backend | Unit tests, compile checks, and CPU backend smoke validation. |

### CUDA GPU / SM Matrix

Tier policy and rationale: see [`plans/sm-coverage.md`](plans/sm-coverage.md).
Env var contract: see [`environment.md`](environment.md) §`TORCH_CUDA_ARCH_LIST`.

| Tier | SM | Representative GPUs | Status | Default-built |
| --- | --- | --- | --- | --- |
| T1 | sm_80 | A100 40/80GB | Supported | yes |
| T1 | sm_86 | A10, RTX 3090, A40, A6000 | Supported | yes |
| T1 | sm_89 | L4, RTX 4090, L40 | Supported | yes |
| T1 | sm_90 | H100, H200 | Supported | yes |
| T2 | sm_100 | B100, B200 | Beta — opt-in via `TORCH_CUDA_ARCH_LIST` | no |
| T2 | sm_120 | RTX 5090, RTX PRO 6000 | Beta — opt-in via `TORCH_CUDA_ARCH_LIST` | no |
| T0-legacy | sm_70 | V100 | Legacy — SM-pinned Qwen3.5 BF16 attention + GDR lane | no |
| T3 | other sm < 80 | T4, Pascal, older | Unsupported — build rejects | n/a |

Notes:

- Hosted CI does not provide full CUDA runtime correctness coverage.
- CUDA correctness and performance still require dedicated GPU validation.
- T1 ship gate requires four-card bench validation (sm_80 + sm_86 + sm_89 + sm_90); see [`plans/sm-coverage.md`](plans/sm-coverage.md) §5.
- sm_70 builds must be SM-pinned (`TORCH_CUDA_ARCH_LIST=7.0`) and are limited
  to the V100 Qwen3.5 BF16 attention + GDR path while Volta fallbacks are
  validated.

---

## 3. Model Family Matrix

| Model family | Status | Notes |
| --- | --- | --- |
| Qwen3.5 | Supported | Primary supported family. Supported on normal runtime paths; Metal live runtime has a narrow same-length decode batch path with packed-batch concurrent decode (2026-04-16 fix). Qwen3.5-0.8B has two measured Metal single-request paths: MLX SafeTensors 4bit step-driver reaches 305.5 tok/s for `1024/256`, while GGUF Q4_K_M exact default is 202.1 tok/s direct and its opt-in native-q4 load path reaches 236.7 tok/s direct / 239.8 tok/s step-driver on the same `1024/256` profile. RoPE scaling (YARN / Linear / NtkAware) wired through `Qwen35Config::rope_scaling` for long-ctx extend (Phase 1+2 closed; Phase 3 bench pending). Metal DFlash is Beta; see §4a for the current validation note. |
| Qwen3.6 / Qwen3.5-MoE | Beta (Metal), CUDA stub | Metal loads and runs `mlx-community/Qwen3.6-35B-A3B-4bit` locally. A 2026-04-27 M4 Pro short diagnostic confirmed load/execute behavior, but DFlash performance decisions for this family should use long-context / ultra-long-sequence workloads only. CUDA intentionally returns a GPU-required stub for Qwen3.6 MoE. Full Qwen 3.6 serving coverage is the **#2 next-model priority** — see roadmap note below. |
| DeepSeek V4 | In progress — V4-only substrate + CPU reference smoke | `crates/deepseek-spec` is V4-only for the local `infer/models/dsv4-mini-1B-init` checkpoint. `cpu_serve` has a slow Rust reference path that mmaps the 2.0 GB safetensors and runs a 1-token HTTP completion smoke. CUDA optimized V4 attention/MoE/MTP kernels remain pending, so this is not a serving-performance target yet. The `arle train pretrain-dsv4` command was retired in the 2026-05-18 OPD-only pivot; DSv4 scratch pretrain is not a supported ARLE workflow. **#1 next-model priority for inference.** |
| Llama 3/4 | Planned | Not yet supported. |
| DeepSeek-V3/R1 | Not carried | Deleted from the current registry/spec/train surface; reintroduction would require a new explicit project, not a compatibility branch inside DSv4. |
| Mistral / Mixtral / Gemma / Phi | Planned | Not yet supported. |

**Next-model roadmap priority** (canonical in [`ROADMAP.md` §Next-Model Priority Order](../ROADMAP.md#next-model-priority-order)):

1. **DeepSeek V4 (DS4)** — V4-only substrate and CPU reference smoke landed; CUDA V4 hybrid attention + MoE + MTP kernels are the active runtime blockers.
2. **Qwen 3.6** — planned / scoping; CUDA serving and kernel coverage land after the DS4 runtime substrate is producing benches. Metal load path already exists for diagnostic use.

Other "Planned" families above sit behind these two and are not actively scheduled.

---

## 4. Quantization Matrix

**Canonical map**: [`docs/quantization.md`](quantization.md). That doc is
the source of truth for KV-cache and weight quantization status, code
locations, test-harness semantics, and the active TileLang HD128
batched paged-prefill investigation (2026-05-27). The summary table
below is the one-glance view — for any change, edit
`quantization.md` first and re-sync here.

| Capability | Status | One-line |
| --- | --- | --- |
| BF16 KV cache | production | Default via `--kv-cache-dtype auto`; correctness-safe reference. |
| INT8 KV cache (CUDA) | production | `--kv-cache-dtype int8`; per-(token, head) /127; +57–113% throughput vs BF16 on A100 (`wins/2026-05-26-bench-int8-vs-bf16-kv-a100`). |
| FP8 E4M3 KV cache (CUDA, +KIVI) | opt-in | `--kv-cache-dtype fp8`; KIVI per-channel K + per-token V scaffolding (`8c6d92db`/`73a72615`/`25c7d409`); quality verdict deferred pending §5 paged-prefill investigation. |
| TurboQuant KV 2/3/4-bit (CUDA) | experimental | `--kv-cache-dtype tq{2,3,4}`; FWHT + packed indices; page_size=1 bypasses the HD128 paged prefill — the only KV format that matches the HF first token on the 2026-05-27 chat audit. |
| Weights — W4A16 / W8A16 / W2A16 | production / experimental (W2) | Native GEMV + Marlin W4 prefill; safetensors auto-detect. |
| Weights — MarlinW4A8 prefill-graph | production, **Tier-1 wins** | `INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1` → engine TTFT p50 –92.5%, +632% throughput (`a56b7a9`/`c44788f`). |
| Weights — GGUF Q3/Q4/Q5/Q6_K | production (CUDA & Metal) | Packed superblock kernels; `.gguf` auto-detect. Metal-native-q4 opt-in via `AGENT_INFER_METAL_GGUF_NATIVE_Q4=all`. |
| Weights — TurboQuant | experimental | Tensor-local gate only (`errors/2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill`). |
| Weights — DSv4 FP8/FP4 block-scaled | in progress | `Dsv4Fp8BlockScaled` / `Dsv4Fp4BlockScaled`; pending CUDA V4 attention/MoE/MTP kernels. |

Backend reach:
- Quantized KV cache is **CUDA-only** today. Metal stores KV in the
  model's native dtype (`bf16` / `f16`) and does not expose
  `--kv-cache-dtype`. Metal weight-quantized MLX models are
  unaffected.

---

## 4b. Multi-turn KV Reuse / Tiered KV Matrix

The KV-reuse architecture that the README calls out (slot-sticky multi-turn
reuse + radix-backed `T0 GPU → T1 host pinned → T2 NVMe → T3 cluster-shared`).
Code lives in `infer/src/prefix_cache.rs` (radix tree) and
`infer/src/kv_tier/` (tiered-KV plumbing); see
[`docs/codebase-map.md`](codebase-map.md) for the per-file map.

| Capability | Status | Notes |
| --- | --- | --- |
| Slot-sticky multi-turn KV reuse | Supported (CUDA), Beta (Metal) | Prior-turn KV stays in slot for the next turn so only new user tokens prefill. CUDA is the primary path; Metal Qwen3.5 ships live prefix reuse via replayed compiled-path snapshots (see §1). |
| Radix-backed prefix cache (T0 GPU) | Supported (CUDA) | Direct GPU-page attach + tail-page CoW on shared prefixes; `RadixNode` carries `hit_count`, `tier_location`, `session_id`, `fingerprint`, `soft_pin_until`, `byte_len`. |
| T1 host-pinned spillover | Beta (CUDA) | Cold blocks demote from GPU to host pinned memory via `HostPinnedPool` (`kv-native-sys` arena); promote-on-use through `ReadmissionPlan`. |
| T2 NVMe local-disk transport | Beta (CUDA) | Node-local persistence via `kv_tier/transport/disk.rs` on top of `crates/kv-native-sys` (file/block ABI, mmap, WAL). |
| T3 cluster-shared backend | Experimental | Minimal `transport/shared_fs.rs` reference backend ships; **NIXL transport remains stub-only** (`nixl-sys` activates the stub feature, no real link). Treat T3 as scaffolding, not a production tier today. |

---

## 4a. Speculative Decoding Matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Metal DFlash (Qwen3.5) | Beta | End-to-end correctness landed 2026-04-17 (commits `4db4fe9`, `439293d`); benchmark before production use. |
| Metal DFlash (Qwen3.6 / Qwen3.5-MoE) | Beta / diagnostic | Target/draft pairing is wired for `mlx-community/Qwen3.6-35B-A3B-4bit` + `z-lab/Qwen3.6-35B-A3B-DFlash`. Short checks are smoke diagnostics only; future DFlash optimization claims must come from long-context / ultra-long-sequence runs. |
| CUDA speculative decoding | Not shipped | CUDA plumbing exists (`infer/src/speculative.rs`, `infer/src/speculative/cuda.rs`, `infer/src/scheduler/cuda/spec_path.rs`) for external/self verifier experiments, but no CUDA spec-decode mode is shipped as throughput-positive. Classical/self/external paths are killed or regressed; Qwen3.5 Medusa is blocked on recurrent-state accepted-length rollback. See [`plans/2026-05-01-longctx-spec-decode-phase2.md`](plans/2026-05-01-longctx-spec-decode-phase2.md) and [`plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md`](plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md). |

---

## 5. Public API Matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `/v1/completions` | Stable | Documented public API. |
| `/v1/chat/completions` | Stable | Documented public API. |
| `/v1/models` | Stable | Loaded-model discovery endpoint. |
| `/v1/responses` | Beta | Non-streaming and SSE forms shipped. Streaming emits `response.created`, `response.output_text.delta`, and terminal `response.completed`; structured outputs are still missing. |
| SSE streaming | Stable at high level | Intended to remain OpenAI-style; edge behavior may improve. |
| `/metrics` | Stable | Prometheus endpoint; Metal now reports live queue / latency / MLX memory gauges. |
| `/v1/stats` | Stable | Human-readable stats endpoint; Metal now reports live queue / latency / MLX memory gauges. |
| Train-side `/v1/train/status|events|stop|save` | Substrate landed; OPD-CLI wiring pending | Control-plane truth lives in `crates/train/src/server.rs` and survives the 2026-05-18 OPD-only pivot. The per-binary `pretrain --serve` / `train_sft --serve` / `train_grpo --serve` / `train_multi_turn --serve` wiring was retired alongside those binaries. OPD CLI (`arle train opd <dir>`) shipped 2026-05-24 (`14c3be9`) as a one-shot runner without `--serve`; reusing the control plane via `arle train opd --serve` is a separate task not yet licensed. `infer` can still expose the surface as an optional proxy via `--train-control-url`. |
| Metal runtime memory knobs | Beta | `metal_request`, `metal_bench`, and `metal_serve` expose `--memory-limit-bytes`, `--cache-limit-bytes`, and `--wired-limit-bytes` for MLX allocator control. |
| CLI agent slash commands | Beta | Usable and documented, but not yet treated like the HTTP API for compatibility. |
| `arle serve` front door | Beta | Launches the matching serving binary (`infer`, `metal_serve`, or `cpu_serve`) from the release artifact or PATH. This is a packaging/DX front door over existing server binaries, not a second HTTP implementation. |
| CLI built-in shell/python tools | Beta | Enabled by default for local trusted agent use. `--no-tools` disables them, and `arle --doctor` reports the detected sandbox backend (`nsjail`, `sandbox-exec`, or `bare`). Do not expose tool-enabled local agent prompts to untrusted users. |
| Structured-output grammar (xgrammar FFI) | Scaffold (Phase 1) | `crates/xgrammar-sys` Rust safe wrapper over upstream `mlc-ai/xgrammar` v0.1.34 (codex's #26 WIP, FFI substrate landed; default build = stub, `--features real` builds C++ shim via `cc` + pinned upstream checkout). No HTTP, scheduler, sampler, or GPU sampling integration yet. Tracked under [`docs/plans/M_xgrammar-ffi-scaffold.md`](plans/M_xgrammar-ffi-scaffold.md). |

## 5a. Training Surface Matrix

> **2026-05-18 pivot — OPD only.** Scratch pretrain, SFT, GRPO, and
> multi-turn RL surfaces were retired in commit `bd94c09`
> ([`docs/projects/2026-05-18-opd-only-pivot.md`](projects/2026-05-18-opd-only-pivot.md)).
> Rationale: the nanochat-d12 industry baseline measured 56 291 tok/s
> single-GPU on this hardware vs ARLE 174.7 tok/s = 322× gap, making
> from-scratch pretrain not a winnable axis; SFT/GRPO/multi-turn
> duplicate mature OSS (vLLM+verl, TRL, axolotl). OPD is the one
> training surface where ARLE's pure-Rust runtime authority is
> structurally differentiating — teacher hosted in `infer`, student
> LoRA on the same backend, no Python on the hot path. Historical
> validation evidence for the retired surfaces lives in
> `docs/experience/wins/` (immutable per bench-spec §9) and is not
> removed.

| Surface | Status | Notes |
| --- | --- | --- |
| `arle train opd` | **Supported (Beta)** | End-to-end CLI shipped 2026-05-24 (`14c3be9`): `arle train opd --student-model <dir> --teacher-model <dir>` runs HF/ModelScope-cached models through `qwen35_loader` + autograd `Tape` + `opd_step` + AdamW directly, no example script needed. CUDA backend. Wins: [`2026-05-24-arle-train-opd-from-dirs`](experience/wins/2026-05-24-arle-train-opd-from-dirs.md). Live task queue tracked in [`2026-05-24-opd-mainline-task-backlog`](projects/2026-05-24-opd-mainline-task-backlog.md). |
| `arle train env` / `arle train estimate-memory` | Supported | Diagnostic surfaces preserved across the OPD-only pivot. `arle train test` was retired permanently in the 2026-05-24 T3 prune (`81842cc`); the test stubs were removed in `cli_smoke` cleanup (`e049787`). |
| Infer-side unified `/v1/train/*` bridge | Supported (optional proxy) | `infer` exposes `/v1/train/status|events|stop|save` when `--train-control-url http://...` is configured, forwarding to the train-side server in `crates/train/src/server.rs`. OPD progress event wiring is separate scope from the OPD CLI ship — `arle train opd` currently has no `--serve` mode; the proxy will host OPD events when that wiring lands. |

---

## 6. CI Coverage Matrix

| Area | Coverage |
| --- | --- |
| Rust CPU-only compile/test | Yes |
| Python tests | Yes |
| Metal compile/test | Yes |
| CUDA compile | Partial |
| CUDA runtime correctness | No full hosted CI |
| Performance regression gating | Not yet standardized |

---

## 7. Update Rule

If support changes for a backend, model family, platform, or quantization path,
update all of the following together:

1. `README.md`
2. `ROADMAP.md` if roadmap status changed
3. `docs/index.md` if the active-doc listing changed
4. this file
5. `CHANGELOG.md` when user-visible

Related docs:

- [stability-policy.md](stability-policy.md)
