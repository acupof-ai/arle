# HIP on-box verification runbook — AI Max 395 (gfx1151), DSv4-Flash 2-bit

Tracks #77/#78. Code state: off-box complete at `98674323` (kernels + shim +
loader + executor + CLI wiring; 53 unit tests green on Mac). This is the ordered
checklist for the first sessions on the box. Each step has a pass criterion;
record evidence inline and pin every version (pin-from-proven-env rule).

## 0. Box prep

| # | Step | Pass criterion |
|---|---|---|
| 0.1 | Ubuntu 24.04+ (Linux only — ROCm APU support is Linux Preview). Record `uname -a`. | — |
| 0.2 | BIOS: UMA/dedicated VRAM to **minimum** (512M–4G); GPU memory comes from GTT. | — |
| 0.3 | Kernel params for ≥96 GB GTT: `ttm.pages_limit` + `amdttm.pages_limit` (page count = bytes/4096; 110 GB ≈ 28835840), reboot. Reference: tinycomputers ran 96 GB GTT. | `cat /sys/module/*ttm*/parameters/pages_limit` shows the value |
| 0.4 | ROCm 7.x known-good stack per [llama.cpp #20856](https://github.com/ggml-org/llama.cpp/discussions/20856) (or TheRock nightly with native gfx1151). **Pin exact versions in the wins entry.** | `rocminfo \| grep gfx1151`; `hipcc --version`; `rocm-smi` shows the 8060S |
| 0.5 | `export ROCM_PATH=/opt/rocm` (build.rs detection order: ROCM_PATH → HIP_PATH → /opt/rocm). | — |
| 0.6 | Artifacts: a DSv4-Flash 2-bit GGUF ([ds4 q2-imatrix](https://github.com/antirez/ds4) preferred; else [nsparks](https://huggingface.co/nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF)/batiai community quants) **+ `tokenizer.json`** from the [unsloth safetensors mirror](https://huggingface.co/unsloth/DeepSeek-V4-Flash) placed in the SAME directory as the .gguf (loader uses `OpenAiTokenizer::from_model_dir`). | ~58–75 GB on disk; sha256 recorded |

## 1. ARLE compile lane

```bash
git clone https://github.com/cklxx/arle && cd arle   # or pull; record HEAD
cargo build --release --no-default-features --features hip,no-cuda,cli --bin arle 2>&1 | tee /tmp/hip_build.log
```

| # | Check | Pass criterion |
|---|---|---|
| 1.1 | hip-kernels build script output in the log | **NO** "hipcc not found … skipping" warning; hipcc invoked on **10 csrc files** (`iq2_mmvq.cu`, `quantized_gemv_mma_stub.cu`, `dsv4_attention.cu`, `dsv4_mhc.cu`, `elementwise_basic.cu`, `decode_prep_paged.cu`, `dsv4_grouped_gemm.cu`, `moe_grouped_gemm.cu`, `norm.cu`, `sampling.cu`, `quantized_gemv.cu`) |
| 1.2 | Expected first-failure class: shim overload gaps (`__hadd/__hadd2` bf16, fp8 casts, `__ldg` bf16) flagged PENDING-REMOTE in `crates/hip-kernels/csrc/arle_hip_shim.h` | fix loop in the shim header only — if a fix requires touching a csrc file, STOP and re-read plan §2.2 (the audit said zero source rewrites; a csrc edit means the audit missed something — record it) |
| 1.3 | Link | binary at `target/release/arle`; `strings target/release/arle \| grep dsv4_hybrid_attention` non-empty (symbol-check rule) |

## 2. Hardware-gated unit smoke

```bash
cargo test -p hip-sys   --release --features hip   # probe + H2D/D2H roundtrip
cargo test -p hip-kernels --release --features hip
cargo test -p infer-hip --release --features hip 2>&1 | tee /tmp/hip_tests.log
```

| # | Check | Pass criterion |
|---|---|---|
| 2.1 | hip-sys probe prints device + total mem | name = Radeon 8060S class; mem ≈ GTT size (NOT the small BIOS carve-out — if it shows 512M–4G, step 0.3 didn't take) |
| 2.2 | All `--features hip` tests | green (off-box-green tests must not regress on-box) |
| 2.3 | Dequant golden: write the ~20-line compare script — run ARLE's `infer_hip::dequant` on 3+ tensors of any real GGUF vs llama.cpp's output for the same tensors (`llama-gguf` dump or a tiny ggml program) | byte-identical f32 (same algorithm — any mismatch is a port bug, fix before proceeding) |

## 3. Reference engines + lane-license A/B (llama.cpp)

```bash
# ROCm build (mainline; known-good flags)
cmake -B build-hip -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=Release
# Vulkan build (radv; needs vulkan SDK + glslc)
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
# DSv4 reference: nisparks fork, branch pr/01-deepseek-v4-arch (same flags)
```

| # | Check | Pass criterion / record |
|---|---|---|
| 3.1 | `llama-bench` on a mid-size dense model, ROCm arm ×{`ROCBLAS_USE_HIPBLASLT=0,1`} × VMM both ways (`GGML_HIP_NO_VMM=ON/OFF` — survey conflict: dense known-good says ON, the DSv4 GTT run needed OFF) | pp512 + tg128 table, 2 runs each |
| 3.2 | Same model on the Vulkan build | **lane license**: if Vulkan ≥ ROCm on BOTH pp and tg AND ROCm showed instability, the executor lane flips to Vulkan (plan §3 sanctions it) — decision recorded with the table |
| 3.3 | nisparks fork boots the DSv4 GGUF (`--no-warmup`, `-c 256`, greedy) | reproduces the ~1–2 tg baseline; this is the floor ARLE must beat AND the output-correctness cross-reference |
| 3.4 | If ds4 builds on ROCm (`antirez/ds4`, MIT): build + run same GGUF | the 30+ tg bar measured on OUR box, not borrowed from M5 Max |

## 4. ARLE DSv4 bring-up

```bash
./target/release/arle --doctor                      # compiled backend: hip
./target/release/arle serve --backend hip --model-path /models/dsv4-q2/model.gguf --port 8000
```

| # | Check | Pass criterion |
|---|---|---|
| 4.1 | Loader: residency plan log + upload completes | total device bytes ≈ plan §1 estimate (~58–75 GB); any fail-loud "matmul role on non-gemv residency" error = recipe mismatch between the GGUF's tensor types and the policy table — fix policy, not kernel |
| 4.2 | `/v1/models` responds; one greedy completion (`temp 0`, short prompt) | **decode and READ the actual tokens** (distilled lesson — judge text, not metrics). Cross-check the same prompt on the nisparks fork (3.3): outputs need not match token-exact (different kernels), both must be coherent |
| 4.3 | Garbage output? config-suspect first: dump GGUF metadata vs our config map (`infer_hip::config` values vs llama.cpp's printed hparams), THEN layer-by-layer (per-layer RoPE theta switch and swiglu clamp are the two known foot-guns) | — |
| 4.4 | Needle gate ×3 same-config repeats + same-config-twice floor (`scripts/dsv4_needle_gate.py` against the HTTP endpoint; ladder as deep as the box's ctx budget allows) | needle-exact at every rung; repeat variance within the nondeterminism floor |

## 5. Perf + license (#78)

| # | Check | Pass criterion / record |
|---|---|---|
| 5.1 | B=1 decode tok/s: ≥3 runs × ≥256 tok, greedy, 4K ctx | p50 vs the three anchors: floor 35–47 (plan §1), ds4-on-this-box (3.4), nisparks 1–2 (3.3). Gap-to-floor = engineering list, not physics |
| 5.2 | `scripts/bench_guidellm.sh hip-gfx1151-dsv4-q2 --model <gguf>` (canonical params) | raw table into the wins entry |
| 5.3 | Wins entry `docs/experience/wins/2026-MM-DD-hip-gfx1151-dsv4-q2-bringup.md`: env pins (ROCm version, kernel params, GGUF sha, ARLE commit), every table above, license-or-kill verdict | ships before any default/README claim; #77 closes on 4.4, #78 closes on 5.x |

## Known failure playbook (from the OSS survey)

- **VMM/GTT allocator**: tinycomputers needed `GGML_HIP_NO_VMM=OFF` + `--no-warmup` for the 58 GB model — if our upload stalls or OOMs below the GTT limit, suspect VMM pool behavior first (hip-sys uses plain `hipMalloc`; GTT-backed allocations may need `hipDeviceMallocUncached`/`hipMallocManaged` experiments — record which works).
- **hipBLASLt "no solution found"**: llama.cpp-reference-only (ARLE's path uses no BLAS); don't chase it in our lane.
- **First-token hang**: check `rocm-smi` for a wedged queue; RDNA GPU hangs under huge single launches were survey-reported — our per-token launches are small, but the 61-token-loop prefill on a long prompt is the stress case.
- **Sequential prefill is slow by design** (MVP): ~decode-rate per prompt token. Do NOT bench prefill-heavy shapes against ds4 until the batched-mmq prefill lands; tg is the licensed comparison.
