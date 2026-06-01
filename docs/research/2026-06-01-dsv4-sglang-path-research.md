# DSv4 SGLang Path Research Before Profiling

## Scope

This is a source/config research note only. No SGLang or ARLE performance test was
run for this note, and the existing remote service process was not used as a
baseline.

Goal: understand what a valid SGLang DSv4 comparison must mean before running a
PyTorch profiler trace or making more ARLE changes.

## Current Evidence

- Remote SGLang source inspected: `/sgl-workspace/sglang`.
- Remote DSv4 model inspected: `/data01/models/DeepSeek-V4-Flash`.
- The model config says `architectures=["DeepseekV4ForCausalLM"]` and
  `model_type="deepseek_v4"`.
- The inspected SGLang source does not register `DeepseekV4ForCausalLM`; grep for
  `DeepseekV4`, `deepseek_v4`, and `V4ForCausalLM` under `python/sglang` returned
  no implementation.
- The DSv4 model directory includes a standalone reference implementation under
  `inference/`, with V4-specific attention compression/indexing and a simple
  MoE path.

This means the current readable `/sgl-workspace/sglang` tree is not yet proven to
be the same SGLang DSv4 path that produced any claimed 18 ms TPOT number. Before
profiling, we must identify the exact DSv4 SGLang fork/patch or model registry
that actually supports `DeepseekV4ForCausalLM`.

## DSv4 Model Shape

Remote `config.json` headline fields:

| Field | Value |
|---|---:|
| layers | 43 |
| hidden size | 4096 |
| heads | 64 |
| head dim | 512 |
| window size | 128 |
| routed experts | 256 |
| active experts/token | 6 |
| shared experts | 1 |
| MoE intermediate | 2048 |
| scoring | `sqrtsoftplus` |
| route scale | 1.5 |
| topk method | `noaux_tc` |
| hash layers | 3 |
| quant | FP8 e4m3, UE8M0 block scales `[128,128]` |
| MTP layers | 1 |

The standalone `inference/model.py` path implements:

- sliding-window + compressed sparse attention;
- indexer top-k over compressed KV (`index_topk=512`);
- hyper-connection pre/post mixing;
- first `n_hash_layers=3` MoE layers with token-id routing;
- normal routed layers using `sqrtsoftplus` scores plus bias;
- per-rank local experts plus all-reduce in the simple reference MoE.

Important correction: `o_groups=8` is attention output grouping in the reference
implementation, not proof of MoE expert-group routing. The generic SGLang
DeepSeek V3 path uses `n_group/topk_group` grouped top-k, but the V4 config read
here does not contain those fields.

## Generic SGLang MoE Path

The inspected SGLang MoE stack has three relevant layers:

1. `--moe-a2a-backend`
   - default: `none`;
   - choices include `deepep`;
   - when set to `deepep`, SGLang forces `ep_size = tp_size`;
   - DeepEP dispatch group is `get_tp_group().device_group`.

2. `--moe-runner-backend`
   - default: `auto`;
   - for FP8 MoE, `auto` chooses `deep_gemm` only when JIT DeepGEMM is enabled
     and A2A is `deepep` or `mooncake`;
   - otherwise FP8 `auto` falls back to Triton in the generic path.

3. `--deepep-mode`
   - default: `auto`;
   - resolves to `normal` for extend/prefill batches;
   - resolves to `low_latency` for decode batches.

So a valid SGLang DeepGEMM+DeepEP comparison cannot be inferred from a launch
command that omits `--moe-a2a-backend deepep` unless a DSv4-specific hook
overrides it elsewhere.

## DeepGEMM Warmup

SGLang enables JIT DeepGEMM by default on SM90 when the `deep_gemm` Python module
is importable.

Key mechanics:

- `SGLANG_ENABLE_JIT_DEEPGEMM=True` by default.
- `SGLANG_JIT_DEEPGEMM_PRECOMPILE=True` by default.
- cache directory comes from `SGLANG_DG_CACHE_DIR`, defaulting to
  `~/.cache/deep_gemm`.
- `python3 -m sglang.compile_deep_gemm ...` starts a server and sends a warmup
  request to trigger compilation.
- without precompile/cache, the first matching DeepGEMM execution may enter a
  "DeepGEMM warmup" loop over many M sizes. Source warning says this can take
  10-20 minutes.

Profiling must exclude this warmup. A profile that includes DeepGEMM compile or
first-use warmup is not a steady-state SGLang baseline.

## DeepEP `num_sms`

SGLang `DeepEPConfig` loads `--deepep-config` if provided. The config has
`normal_dispatch` and `normal_combine`, and both must agree on `num_sms`.

If no config is supplied, SGLang uses DeepEP's default `Buffer.num_sms`.

`num_sms` affects normal-mode communication resources and also participates in
DeepEP buffer initialization:

- normal mode: `num_qps_per_rank = DeepEPConfig.num_sms`;
- low-latency mode: `num_qps_per_rank = num_experts / group.size()`;
- auto mode: `num_qps_per_rank = max(num_sms, num_experts / group.size())`.

SGLang warns when normal-mode DeepEP uses fewer than half the GPU SMs and TBO is
off. Therefore `num_sms=80` is not a random tuning knob: it is part of the
communication resource and queue-pair sizing contract.

## SGLang Profiler Path

SGLang already has a PyTorch profiler API:

```bash
python3 -m sglang.profiler \
  --url http://127.0.0.1:30000 \
  --num-steps N \
  --profile-by-stage \
  --output-dir <dir> \
  --merge-profiles
```

The `/start_profile` request with `profile_by_stage=true` records prefill
(`EXTEND`) and decode (`DECODE`) separately:

- first prefill batch starts the EXTEND profile;
- first decode batch stops EXTEND and starts DECODE;
- after the target decode count, SGLang stops and flushes DECODE.

Caveat: CUDA graph visibility is separate. `--enable-profile-cuda-graph` profiles
graph capture paths and adds top operator tables there. It does not automatically
make steady-state CUDA graph replay equivalent to a fully expanded per-kernel
decode trace. If we disable CUDA graph for visibility, that is a different
execution path and must be labeled as such.

## Current ARLE Comparison Risks

The current ARLE DSv4 path and the intended SGLang fast path are not yet proven
to be isomorphic.

Known ARLE state from local source/docs:

- `ARLE_DSV4_MOE_BACKEND` default is `allreduce` for the current replicated-token
  path.
- `native-deepep` is guarded because DeepEP dispatch/combine assumes token-sharded
  EP ownership; passing replicated full-token rows creates a fanout/transport
  mismatch.
- `infer/src/tensor_parallel.rs` contains SGLang-style axis math, but comments
  state it is a diagnostics/contract input, not proof that DP/CP/MoE-DP execution
  is fully wired.
- `ARLE_DSV4_SGLANG_PATH=1` is fail-closed unless the DSv4 SGLang path contract
  is complete.

Script risk:

- `scripts/dsv4_beat_sglang_bench.sh` launches SGLang with only
  `--tp 8 --kv-cache-dtype fp8_e4m3`; it does not force DeepEP, does not
  precompile DeepGEMM, and assumes `/sgl-workspace/sglang` supports DSv4.
- That script is therefore not a valid SGLang-best-practice profiler harness yet.

## What Must Be Locked Before Running Perf

1. Identify the exact SGLang DSv4 implementation:
   - source tree or patch that registers `DeepseekV4ForCausalLM`;
   - attention backend used for DSv4 Flash;
   - required kernel package versions.

2. Lock the SGLang launch contract:
   - TP/DP/EP sizes;
   - whether `--enable-dp-attention` is required;
   - `--moe-a2a-backend deepep` or DSv4-specific override;
   - `--moe-runner-backend auto` versus explicit `deep_gemm`;
   - `--deepep-mode auto`;
   - `--deepep-config` and its `num_sms`;
   - CUDA graph settings;
   - speculative decoding on/off.

3. Exclude warmup from profiler windows:
   - run `sglang.compile_deep_gemm` or an equivalent warmup until DeepGEMM cache
     is populated;
   - verify there is no "DeepGEMM warmup" in the measured window.

4. Separate metrics:
   - no-spec raw target-step TPOT;
   - speculative output-token TPOT with accept rate;
   - TTFT and TPOT should not be mixed across these definitions.

5. Match correctness before performance:
   - same prompt, tokenizer, `ignore_eos`, max tokens, and output sanity;
   - decode actual generated text when output is suspicious.

## Research Conclusion

The next correct step is not to run another trace immediately. The correct next
step is to locate or reconstruct the actual SGLang DSv4 path first. The inspected
SGLang tree explains DeepGEMM warmup, DeepEP grouping, `num_sms`, and profiler
mechanics, but it does not by itself establish a runnable DSv4 SGLang baseline.

Once the SGLang DSv4 implementation is identified, the first profiler run should
be a warm, stage-split PyTorch profile with DeepGEMM warmup excluded and with the
launch contract recorded verbatim.
