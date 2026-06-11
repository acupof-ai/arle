# ARLE Support Matrix

This document is the canonical support-status truth for `ARLE`.

It states what the repository currently supports, what is still limited, and
what validation exists for each area. If something is not listed as supported
here, do not assume it is supported just because it compiled locally.

State reflected here is based on repository evidence as of 2026-06-10.
**The device-neutral rewrite (`crates/infer-*`) IS the product** — PR #53 merged
to `main` 2026-06-04, and the legacy monolithic `infer/` crate is **deleted**.
Sections 1–7 below were written against the legacy runtime; those capabilities
are now served by the rewrite, and a per-capability re-verification pass is
ongoing — read §0 for the **current** new-stack status, and treat any §1–§7
"Supported" as the *capability intent*, verified on the rewrite only where §0 or
a dated `wins/` entry says so. Project framing:
[index.md §Current Positioning](index.md#current-positioning).

---

## 0. Rewrite-stack support (the shipped `crates/infer-*` graph)

> **Merged to `main` (PR #53, 2026-06-04); legacy `infer/` deleted. This is the
> product.** The stack is `infer-plan` (IR) → `infer-seam` (host-only traits) →
> `infer-core` (`Engine<E,K>`) → `infer-cuda` / `infer-metal` (executors) →
> `infer-server` / `infer-api` (serving front door). Per-capability
> re-verification on the new stack is ongoing; the rows below are current status.
> Source of truth:
> [`projects/2026-06-04-qwen35-dsv4-final-report.md`](projects/2026-06-04-qwen35-dsv4-final-report.md)
> + [`projects/2026-06-04-rewrite-completion-verification-report.md`](projects/2026-06-04-rewrite-completion-verification-report.md).

| Backend | New-stack status | What is verified | Open |
| --- | --- | --- | --- |
| Metal (`infer-metal`) | **Verified** | Real MLX Qwen3.5/3.6 forward, **bit-identical greedy parity** vs legacy across 4 configs (Qwen3.5-0.8B single-token / full 16-tok / chunked prefill, and canonical Qwen3.6-35B-A3B-4bit MoE). Cross-step decode pipeline recovered the c=1 perf regression. Metal serve is intentionally single-flight on macOS: the executor reports one live request and one plan row, so a second live HTTP request is rejected instead of queued. | Full packed batched-decode parity with CUDA; FP8/4-bit Metal MoE quant swap points. |
| CUDA (`infer-cuda`) | **Verified (serve: prefill + decode)** | Qwen3 dense **16/16** vs HF gold; DSv4-Flash **TP=8/EP=8** (MLA+CSA/HCA+FP8 DeepGEMM MoE, FlashMLA, DeepEP) serves in-process multi-rank (`63d814a4`); per-layer RoPE theta fix (`fa355315`) → needle exact at 32K; decode ~39 tok/s c=1. | Long-ctx ≥241 trailing-digit residual (#56); 256K admission band-aid (#57); KV-precision-parity gate re-port (#58); batched lane license (#60/#61); spec-decode default (#62). |

**New-stack model coverage:** Qwen3.5 / Qwen3.6 on Metal (verified); Qwen3 dense +
DeepSeek-V4-Flash (TP=8/EP=8) on CUDA (prefill verified, decode in progress);
Qwen3.5/3.6 hybrid on CUDA (parity follow-up).

**Now in the new stack (was legacy-only):** TP / EP / DeepEP multi-GPU, DeepGEMM
(FP8 grouped GEMM), DSv4 (MLA + FP8 KV), and the HTTP/serving surface
(`infer-server` + `infer-api`, both executors wired). **Still pending re-port /
verification:** PP (pipeline parallel), the full weight/KV **quantization** Rust
dispatch, tiered KV (T1–T3), speculative decode (IR hooks only), and DSv4
incremental decode — tracked in §1–§7 + the active tasks. The capability detail
in §1–§7 below predates the rewrite; verify against §0 + dated `wins/` entries.

---

## 1. Runtime Backends

> Legacy `infer/` (shipped product). For the new rewrite stack see [§0](#0-rewrite-stack-support-new-crate-graph-not-yet-shipped).

| Backend | Status | Meaning |
| --- | --- | --- |
| CUDA | Supported | Primary serving path. Main runtime, scheduler, and benchmark focus. |
| Metal | Beta | Usable for local validation and live scheduler-backed serving. Qwen3.5 ships live prefix reuse via replayed compiled-path snapshots; `arle serve --backend metal` is the canonical Apple bring-up path (in-process serve). Qwen3.5-0.8B MLX 4bit single-request step-driver is measured at 305.5 tok/s on M4 Pro 20c for `1024/256`. The matched GGUF Q4_K_M exact default is 202.1 tok/s direct; the opt-in native-q4 load path reaches 236.7 tok/s direct / 239.8 tok/s step-driver and remains a separate exact packed-K-quant kernel/format gap. Metal is still missing full batched-decode parity with CUDA, especially on variable-length Qwen3.5 decode. |
| Metal DFlash | **Substrate only (serving not re-ported)** | The DFlash draft-model FFI lives in `crates/mlx-sys` and hub discovery rejects draft-only checkpoints, but the rewrite Metal serve has **no DFlash/MTP route**. `arle serve --backend metal --spec-type mtp` and `--mtp-draft-model` fail closed at CLI/API validation. The monolith-era default-on DFlash serving died with the rewrite. |
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
| Qwen3.5 | Supported | Primary supported family. Supported on normal runtime paths; Metal live runtime has a narrow same-length decode batch path with packed-batch concurrent decode (2026-04-16 fix). Qwen3.5-0.8B has two measured Metal single-request paths: MLX SafeTensors 4bit step-driver reaches 305.5 tok/s for `1024/256`, while GGUF Q4_K_M exact default is 202.1 tok/s direct and its opt-in native-q4 load path reaches 236.7 tok/s direct / 239.8 tok/s step-driver on the same `1024/256` profile. RoPE scaling (YARN / Linear / NtkAware) wired through `Qwen35Config::rope_scaling` for long-ctx extend (Phase 1+2 closed; Phase 3 bench pending). Metal DFlash is substrate-only in the rewrite serve path; see §4a for the current validation note. |
| Qwen3.6 / Qwen3.5-MoE | Supported (Metal canonical), CUDA pending (#65) | `mlx-community/Qwen3.6-35B-A3B-4bit` is the **canonical Metal production model** (globally unified 2026-05-07) — every Metal serve/bench/test defaults to it. CUDA classifies Qwen3 MoE checkpoints (`infer-api` `classify_cuda_model`), but Qwen3.6 CUDA serving needs the second `ModelKvAdapter` (#65, Phase 3). |
| DeepSeek V4 | Serving (CUDA 8×H20 TP=8/EP=8) | DSv4-Flash serves via `arle serve --backend cuda` in-process multi-rank: FlashMLA + DSA/CSA/HCA hybrid attention, FP8 KV, DeepGEMM FP8 MoE, DeepEP/allreduce transports. Needle-exact to 32K after the per-layer RoPE theta fix (`fa355315`); decode ~39 tok/s c=1. Open debt tracked in #55 (Phase 0: #56–#58; batched lane #60/#61; MTP default #62). `crates/deepseek-spec` remains V4-only; DSv4 scratch pretrain stays retired (2026-05-18 OPD-only pivot). |
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
Code lives in `crates/infer-core/src/{prefix,radix}.rs` (radix tree) and the
`crates/kv-native-sys` persistence substrate (tiered-KV plumbing); see
[`docs/codebase-map.md`](codebase-map.md) for the per-file map.

| Capability | Status | Notes |
| --- | --- | --- |
| Slot-sticky multi-turn KV reuse | Supported (CUDA), Beta (Metal) | Prior-turn KV stays in slot for the next turn so only new user tokens prefill. CUDA is the primary path; Metal Qwen3.5 ships live prefix reuse via replayed compiled-path snapshots (see §1). |
| Radix-backed prefix cache (T0 GPU) | Supported (CUDA) | Direct GPU-page attach + tail-page CoW on shared prefixes; `RadixNode` carries `hit_count`, `tier_location`, `session_id`, `fingerprint`, `soft_pin_until`, `byte_len`. |
| T1 host-pinned spillover | Beta (CUDA) | Cold blocks demote from GPU to host pinned memory via `HostPinnedPool` (`kv-native-sys` arena); promote-on-use through `ReadmissionPlan`. |
| T2 NVMe local-disk transport | Beta (CUDA), rewrite serve not active | Node-local persistence on top of `crates/kv-native-sys` (file/block ABI, mmap, WAL). The disk transport was legacy `infer/`-only; re-porting below the executor seam is sequenced in the multi-GPU port roadmap (see §0). `arle serve --kv-ssd-path ...` validates the request and fails closed until a backend exposes real SSD recall. |
| T3 cluster-shared backend | Experimental | A minimal shared-FS reference backend shipped in the legacy tree; **NIXL transport remains stub-only** (`nixl-sys` activates the stub feature, no real link). Treat T3 as scaffolding, not a production tier today; not yet re-ported into the rewrite stack (see §0). |

---

## 4a. Speculative Decoding Matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Metal DFlash (Qwen3.5) | Substrate only | End-to-end correctness existed in the legacy tree, but the rewrite serve path has no Metal DFlash route. |
| Metal DFlash (Qwen3.6 / Qwen3.5-MoE) | Substrate only / diagnostic | Target/draft assets exist, but rewrite serving is fail-closed until the external draft route is re-ported and benchmarked. |
| CUDA speculative decoding | Opt-in DSv4 MTP only | `arle serve --backend cuda --spec-type mtp --mtp-draft-tokens N` lowers into the DSv4 checkpoint-native MTP head. Classical/self/external draft routes remain not shipped; Qwen3.5 Medusa is blocked on recurrent-state accepted-length rollback. See [`plans/2026-05-01-longctx-spec-decode-phase2.md`](plans/2026-05-01-longctx-spec-decode-phase2.md) and [`plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md`](plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md). |

---

## 5. Public API Matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `/v1/completions` | Stable | Documented public API. |
| `/v1/chat/completions` | Stable | Documented public API. |
| `/v1/models` | Stable | Loaded-model discovery endpoint. |
| `/v1/responses` | Beta | Non-streaming and SSE forms shipped. Streaming emits `response.created`, `response.output_text.delta`, and terminal `response.completed`; structured outputs are still missing. |
| SSE streaming | Stable at high level | Intended to remain OpenAI-style; edge behavior may improve. |
| `/metrics` | **Not re-ported (rewrite gap)** | Route does not exist in `infer-server`. The monolith's Prometheus endpoint died with the rewrite; re-port pending. |
| `/v1/stats` | Minimal rewrite surface | JSON scheduler counters plus prefix-cache hit counters. SSD KV recall reports `available=false` because the rewrite serve path has no active SSD recall tier. |
| Train-side `/v1/train/status|events|stop|save` | Substrate landed; OPD-CLI wiring pending | Control-plane truth lives in `crates/train/src/server.rs` and survives the 2026-05-18 OPD-only pivot. The per-binary `pretrain --serve` / `train_sft --serve` / `train_grpo --serve` / `train_multi_turn --serve` wiring was retired alongside those binaries. OPD CLI (`arle train opd <dir>`) shipped 2026-05-24 (`14c3be9`) as a one-shot runner without `--serve`; reusing the control plane via `arle train opd --serve` is a separate task not yet licensed. The CUDA serving path can still expose the surface as an optional proxy via `--train-control-url`. |
| Metal runtime memory knobs | Auto (no flags) | The rewrite Metal executor auto-pins model weights via `mlx::set_wired_limit` (model size + 1 GiB headroom, `infer-metal/src/wired_limit.rs`) at construction. The old `--memory-limit-bytes` / `--cache-limit-bytes` / `--wired-limit-bytes` flags died with the monolith; no CLI override currently exists. |
| CLI agent slash commands | Beta | Usable and documented, but not yet treated like the HTTP API for compatibility. |
| `arle serve` front door | Stable | **In-process** serving (`crates/cli/src/serve.rs`): the single `arle` binary loads the model and serves OpenAI v1 directly — no standalone backend binaries are spawned or searched. `--bind` is honored by every backend. |
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
