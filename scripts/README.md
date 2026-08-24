# Scripts Index

ARLE utility scripts, categorized by function. All paths relative to this
directory unless noted.

## Benchmarking

| Script | Purpose |
|---|---|
| `bench_throughput.py` | Canonical OpenAI-compatible streaming throughput runner. |
| `bench_ab.sh` | Matched A/B benchmark driver wrapping `bench_throughput.py`. |
| `bench_compare.py` | Compare two `bench_throughput.py` v1 snapshots with Δ% and threshold. |
| `bench_dsv4_trace_http.py` | DSv4 trace-driven HTTP benchmark. |
| `bench_local_metal.py` | Local Metal backend benchmark. |
| `bench_local_metal_all.sh` | Run full Metal benchmark grid. |
| `bench_local_metal_supplement.sh` | Supplemental Metal benchmark shapes. |
| `bench_mlx_http_decode.py` | MLX HTTP decode benchmark. |
| `bench_multitenant_burst.py` | Multi-tenant burst throughput benchmark. |
| `bench_agent.py` | Local agent end-to-end benchmark. |
| `bench_agent_trace.py` | Agent trace-driven benchmark. |
| `bench_long_agent.py` | Long-horizon agent benchmark. |
| `bench_sglang_longctx.sh` | SGLang long-context baseline runner. |
| `run_dsv4_bench.sh` | DSv4 benchmark orchestration. |
| `run_fp8_probe.sh` | Run the FP8 component probe; when a producer manifest is supplied, require its build ID to match the binary-embedded `KERNEL_BUILD_ID`. |
| `operator_e2e_artifact.py` | Wrap a `needle_gate.py` log into the `arle.operator-e2e/v1` artifact `run_fp8_probe.sh` consumes (identity = kernel bundle id). |
| `reduce_operator_evidence.py` | Reduce qualified probe runs into the generated dispatch policy (`--check` validates without writing). |

## Model Conversion

| Script | Purpose |
|---|---|
| `convert.py` | Shared checkpoint conversion IO (load_all_tensors, save_checkpoint, copy_config_files). Used by the conversion scripts below. |
| `convert_gptq_to_w4a16.py` | Convert a local GPTQ v1 directory to ARLE W4A16; download first with `hf download REPO --local-dir DIR`. |
| `convert_gptq_w4a16_to_w4a8_marlin.py` | Convert W4A16 to hybrid W4A8 Marlin format. |
| `convert_dspark_speculators.py` | Convert DSpark speculator checkpoints. |
| `gguf_to_safetensors.py` | Convert GGUF to safetensors format. |
| `merge_w4_hybrid_checkpoint.py` | Merge hybrid W4A16/W4A8 checkpoint shards. |
| `setup_qwen3_yarn_config.py` | Set up YaRN RoPE config for Qwen models. |

## Quantization

| Script | Purpose |
|---|---|
| `quantize.py` | Unified quantization entry: `--format fp8\|w8a16\|w4a8-marlin\|turboquant`. Wraps the per-format scripts below. |
| `fp8_block_cast.py` | BF16 → DeepSeek-style FP8 block-scaled (128×128). |
| `w8a16_quant.py` | BF16 → W8A16 per-group signed INT8. |
| `quantize_qwen3_w4a8.py` | Quantize Qwen to W4A8 (pack_w4a8 + Marlin). |
| `turboquant_weights.py` | TurboQuant 4-bit quantization. |
| `qwen35_tq4_dense_parity.py` | Verify Qwen3.5 TQ4 dense parity. |
| `qwen36_dense_to_nvfp4.py` | Convert Qwen3.6 dense to NVFP4 format. |
| `requant_dspark_mxfp4_to_fp8.py` | Re-quantize DSpark MXFP4 to FP8. |
| `marlin_repack.py` | GPTQ int32 → Marlin tile layout repack. |
| `verify_gptq_w4a8_repack_quality.py` | Verify W4A8 repack quality vs baseline. |
| `diag_w4a8_pack_roundtrip.py` | W4A8 pack round-trip diagnostic. |

## Correctness Gates

| Script | Purpose |
|---|---|
| `needle_gate.py` | Needle-in-haystack retrieval correctness gate. |
| `needle_concurrent.py` | Concurrent needle gate: N in-flight requests, distinct needle per row (catches cross-row state mix-up). |
| `needle_summary.py` | Shared parser for `needle_gate.py` SUMMARY lines. |
| `lever_gate.sh` | Model/backend-neutral correctness gate: boots serve, runs needle ladder + temp + concurrent arms, validates against baseline envelope. |
| `sampling_gate.py` | End-to-end sampling-parameter gate (penalties, logit_bias, liveness). |
| `longctx_numerical_gate.py` | Long-context numerical quality gate. |
| `dsv4_batched_decode_validate.py` | DSv4 batched decode correctness. |
| `dsv4_multigpu_parity.sh` | DSv4 multi-GPU parity test. |
| `assert_kernel_fired.sh` | Assert a specific CUDA kernel was launched. |

## DSv4

| Script | Purpose |
|---|---|
| `dsv4_c_sweep.py` | DSv4 concurrency sweep benchmark. |
| `dsv4_toolchain.sh` | DSv4 native DeepEP/DeepGEMM toolchain validator. |

## Evaluation

| Script | Purpose |
|---|---|
| `arle_capability_eval.py` | ARLE capability evaluation (MMLU, GSM8K, etc.). |
| `arle_capability_compare.py` | Compare capability eval results across backends. |
| `arle_swe_pro_eval.py` | SWE-Pro evaluation harness. |
| `score_rubric_eval.py` | Rubric-based evaluation scoring. |
| `analyze_multi_seed.py` | Multi-seed eval analysis with mean±σ and Wilson CI. |
| `probe_report.py` | Probe logit-lens report generator. |

## OPD Training

| Script | Purpose |
|---|---|
| `agent_opd_curve.sh` | Agentic OPD capability curve runner. |
| `opd_capability_curve.py` | OPD capability curve generator. |
| `opd_security_filter.py` | OPD rollout security content filter. |
| `h20_teacher_student_opd_curve.sh` | H20 teacher-student OPD curve. |
| `clean_opd_corpora.py` | Clean and deduplicate OPD training corpora. |
| `fetch_opd_corpora.py` | Fetch OPD training corpora. |
| `filter_inband.py` | Filter in-band calibrated task pool. |
| `gen_36_warm_prefix_mix.py` | Generate warm prefix mix for eval. |
| `gen_agent_opd_tasks.py` | Generate agent OPD training tasks. |
| `gen_terminal_tasks.py` | Generate Terminal-Bench tasks. |
| `stage_opd_run_corpus.py` | Stage OPD run corpus. |
| `stage_swe_pro.py` | Stage SWE-Pro eval. |
| `tbench_calibrate.sh` | Terminal-Bench calibration. |
| `tbench_full.sh` | Terminal-Bench full run. |
| `tbench_opd_loop.sh` | Terminal-Bench OPD loop. |
| `tb_exclude_security.sh` | Terminal-Bench security exclusion. |
| `terminal_bench_eval.sh` | Terminal-Bench evaluation. |
| `terminal_bench_serve.sh` | Terminal-Bench serve setup. |
| `terminus_to_records.py` | Convert Terminus output to records. |
| `train_and_chat.sh` | Quick train+chat smoke test. |

## Build & Deploy

Canonical pod flow is receipt-bound: `sync` atomically installs the complete
working tree and source receipt; `build` records that source identity and exact
artifact; `run` atomically records argv, GPU, build identity, and process
ownership before launch; `ready` verifies that owned serve against `/v1/stats`.
`status` and `kill` fail closed unless the exact helper, operation, PID,
start-time, PGID, and binary still match.

```bash
scripts/pod.sh push-scripts
scripts/pod.sh sync
scripts/pod.sh build <build-label> [cargo args...]
scripts/pod.sh status <build-label>
scripts/pod.sh run <build-label> [run-label] [auto|GPU] -- serve [args...]
scripts/pod.sh ready <run-label> [timeout]
scripts/pod.sh status <run-label>
scripts/pod.sh kill <run-label>
```

Labels use `[A-Za-z0-9_.-]+`; omitted build/run labels are unique timestamps.
Labels are immutable. Receipts and logs live under
`$POD_STATE/{builds,runs}/<label>/`.

**Kernels Publish** candidate mode (CI, auto on push) generates and packs the
bundle once. GPU qualification is **manual** — there is no GPU CI runner: run the
strict gate on each target GPU (pod H20 / `ssh v100`), emit one qualification
fragment per GPU, aggregate, and publish the unchanged candidate plus its
qualification sidecar:

```bash
scripts/kernel_artifacts.sh qualify-fragment CANDIDATE STATS_JSON FRAGMENT_JSON
scripts/kernel_artifacts.sh aggregate-qualification CANDIDATE AGGREGATE_JSON FRAGMENT_JSON...
scripts/kernel_artifacts.sh qualify-publish CANDIDATE AGGREGATE_JSON
```

Dispatch `kernels-publish.yml` without inputs to create the candidate; run the
GPU gate manually per target and hand `qualify-publish` the aggregated fragments.

| Script | Purpose |
|---|---|
| `install.sh` | One-line installer (Linux x86_64 / Apple Silicon). |
| `docker_build_dev.sh` | Build dev Docker image. |
| `docker_push.sh` | Push Docker image to registry. |
| `pod.sh` | Remote pod build/run orchestration. |
| `pod-build-env.sh` | Pod build environment setup. |
| `pod-remote-build.sh` | Remote pod build wrapper. |
| `pod-remote-run.sh` | Remote pod run wrapper. |
| `pod-tilelang-env.sh` | Pod TileLang Python environment setup. |
| `pod_serve.sh` | Pod serving setup. |
| `cuda_prebuilt_manifest.sh` | Shared hashing and strict producer-manifest validation helpers. |
| `export_prebuilt_cuda_kernels.sh` | Validate and export a producer manifest plus its exact artifacts. |
| `package_macos_metal_artifact.sh` | Package macOS Metal artifact. |
| `kernel_artifacts.sh` | Pack immutable CUDA candidates, create qualification fragments, aggregate exact-candidate evidence, and publish the unchanged payload with a qualification sidecar. |
| `validate_release.sh` | Fail-closed tag/product/blocker/kernel-evidence release validator. |
| `ci-fmt-check-changed.sh` | CI rustfmt check on changed files. |
| `ci-patch-tvm-ffi.sh` | CI TVM FFI patch. |
| `pre_push_checks.sh` | Pre-push validation checks. |
| `check_repo_hygiene.py` | Repo hygiene checker (wins cap, frozen archive seal, etc.). |
| `clean_repo.py` | Sweep untracked build/run residue; dry-run by default, `--apply` to delete. |
| `archive_experience.py` | Seal wins/errors entries into the frozen archive with link repair and manifest update. |
| `pick-gpu.sh` | GPU selection helper. |
| `start_agent.sh` | Start local agent. |
| `v100_qwen35_9b_load_smoke.sh` | V100 Qwen3.5-9B load smoke test. |
| `reap_run.py` | Pod subreaper process wrapper. |
| `vllm_serve_control.sh` | vLLM serve control for nsys benchmarks. |

## Profiling

| Script | Purpose |
|---|---|
| `profile_bench_common.sh` | Shared profiling benchmark functions. |
| `profile_ncu_bench.sh` | Nsight Compute (ncu) profiling benchmark. |
| `profile_nsys_bench.sh` | Nsight Systems (nsys) profiling benchmark. |

## Kernel Dev

| Script | Purpose |
|---|---|
| `tilelang_jit_smoke.py` | TileLang JIT smoke test. |
| `tilelang_metal_dev_backend.py` | TileLang Metal dev backend. |
