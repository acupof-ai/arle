# M_quant D2 — W4A8 GPTQ greedy_consistency gate ready-to-execute

**Trigger:** `e753af7` Phase 1b LICENSED + loader detection chain complete
**Owner:** TBD(codex executor or user direct)
**Effort:** ~10 min total(5-10 min CPU convert + 1-2 min GPU greedy gate)
**Master strategy:** §1.2.1.A weight axis 全套 → W4A8 default-on flip

## Why this is ready

End-to-end loader chain audited complete:
1. `scripts/convert_gptq_w4a16_to_w4a8_marlin.py`(via `pack_w4a8(gptq_scales=...)`)
   - Writes safetensors with `marlin_w4a8_*` naming
   - **Patches `config.json` with `quantization_config: {quant_type: marlin_w4a8, group_size: 128}`**(per latest script edit)
   - Does not write `quantize_config.json`; that file would force the GPTQ loader branch and disable `marlin_w4a8`
2. Loader detection at `infer/src/weight_loader.rs:514`:
   ```rust
   Ok("marlin_w4a8" | "w4a8_marlin")
   ```
   → `marlin_w4a8 = true` flag set
3. Loader at `:663-715` reads tensors via the W4A8-specific path
4. Linear FFI at `linear.rs:777-859` calls `gemm_w4a8_marlin_cuda` with:
   - INT8 activation(per-token max/127 quant)
   - W4 packed bytes(W4A8Layer 4-consecutive perms,validated `0be5967`)
   - F32 s_channel,F16 s_group(GPTQ-aware,~0.02% drift `e753af7`)
5. Kernel(`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`):
   - 0-diff verbatim from PR #31 W4A8Layer kernel(audit `01ace86`)

All script-level + wiring-level checks pass。Greedy gate is the
end-to-end correctness validation。

## Step 1 — Convert checkpoint(5-10 min CPU)

```bash
.venv/bin/python scripts/convert_gptq.py \
    infer/models/Qwen3-4B-GPTQ-Int4 \
    --output infer/models/Qwen3-4B-GPTQ-Int4-converted-zpfix

.venv/bin/python scripts/convert_gptq_w4a16_to_w4a8_marlin.py \
    --src infer/models/Qwen3-4B-GPTQ-Int4-converted-zpfix \
    --dst infer/models/Qwen3-4B-GPTQ-W4A8-marlin \
    --groupsize 128
```

Expected output:
- `convert_gptq.py` decodes GPTQ `qzeros` with the required `+1` zero-point offset
- `~250 layers re-packed`
- First re-pack reports shapes:e.g. `qweight=[160, 5120] s_channel=[1, 2560] s_group=[20, 2560]`
- Saves `model.safetensors`(~2.66 GB)
- Patches `config.json` with `quantization_config: marlin_w4a8`
- Does not write `quantize_config.json`; W4A8 loader detection uses the
  inline `config.json` `quantization_config`.

If existing dst exists,delete first:`rm -rf infer/models/Qwen3-4B-GPTQ-W4A8-marlin/`

## Step 2 — Greedy consistency gate(~1-2 min GPU)

```bash
INFER_TEST_W4A8_MODEL_PATH=infer/models/Qwen3-4B-GPTQ-W4A8-marlin \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
    cargo test --release -p infer --features cuda \
    --test greedy_consistency test_w4a8_vs_bf16_token_diff -- --nocapture
```

### Decision tree

**PASS** → axis 3 calibration LICENSED end-to-end:
1. Mark task #34 W4A8 accuracy fix complete
2. Run `scripts/bench_guidellm.sh m_quant-w4a8-gptq-canonical-bench`
3. Compare TTFT/ITL/throughput vs:
   - BF16 baseline
   - W4A16 Marlin(`f6f3af3`)
   - W4A8 naive max-scale(known fast-garbage,for noise calibration)
4. If perf good → flip default-on(after `62e75ee` graph capture wired)
5. Update master strategy §1.2.1.A `❌` → `✅ via GPTQ-Marlin re-pack`

**FAIL but partial improvement**(token diff < 100% but not exact match)
→ residual activation INT8 quant compounding through 36 layers:
1. Investigate per-layer first-divergence(instrument forward pass)
2. If divergence at layer 0:check w4a8_activation_quant.cu interaction
3. If cascading from layer N:cumulative noise issue,document and
   consider activation calibration step
4. Update `12a54da` patch decision with new observation

**FAIL same as naive max-scale**(period spam / multilingual gibberish)
→ kernel-side issue not caught by audit:
1. Run `scripts/diag_w4a8_pack_roundtrip.py` on the produced checkpoint
   directly(read tensors back, manual unpack, compare to GPTQ source)
2. If round-trip passes → bug is in kernel storage interpretation(rare,
   would require `marlin_w4a8_kernel.cu` deeper audit)
3. If round-trip fails → check `convert_gptq_w4a16_to_w4a8_marlin.py`
   field naming alignment with loader expectations

## Step 3 — Bench(if Step 2 PASS)

```bash
./scripts/bench_guidellm.sh m_quant-w4a8-gptq-canonical \
    --model-path infer/models/Qwen3-4B-GPTQ-W4A8-marlin \
    --backend cuda
```

Expected output(`docs/experience/wins/2026-05-08-w4a8-gptq-bench.md`):
- TTFT p50/p99
- ITL p50/p99
- Throughput tok/s
- σ across 3 runs

CLAUDE.md mandatory bench rule:every runtime change → wins/ entry。
W4A8 GPTQ converted-checkpoint enabling counts as runtime change。

## Step 4 — Default-on flip decision

**If Step 3 numbers competitive vs W4A16 + accuracy preserved**:
- Flip default-on:`infer/src/ops/linear.rs` decode threshold update
- Or via `infer/src/scheduler/types.rs` config default
- Update master strategy §1.2.1.A:
  - `⚠ W4A16 Marlin(production但未 calibrated)`
  - `✅ W4A8 GPTQ(default-on)` ← new
- Bench artifact crosslinked from master strategy

**If TTFT/ITL regression**:
- Hold default-on,document W4A8 as "throughput-tier substrate"
- Use W4A16 for short-context decode where Marlin 1.64× ITL was licensed(`f6f3af3`)
- Use W4A8 for prefill-heavy workloads where TTFT win surfaces

## Open follow-ups not blocking D2

1. **W3 c=16 8/384 tail failure investigation**(D1 from `fdb951f`)— separate axis 1 sub-task
2. **TTFT p99 tail-latency plan**(D4)— write `M_pf-tail-latency` low-priority
3. **Medusa axis 2 implementation**(D3)— start after D2 lands

## Cross-references

- Phase 1b LICENSE: [`e753af7`](../experience/wins/2026-05-08-w4a8-gptq-rep-canonical.md)
- GPTQ-aware patch: [`12a54da`](.../tools-quant-pack-gptq-aware.md)
- Convert script: `scripts/convert_gptq_w4a16_to_w4a8_marlin.py`(latest:line 133-153 add config.json patching)
- M_quant plan: [`662cbbb`](M_quant-autogptq-marlin-integration.md)
- W4A8 root cause: [`39237b9`](../research/2026-05-08-w4a8-naive-max-scale-too-lossy-need-calibration.md)
- 2-axis milestones consolidation: [`fdb951f`](../research/2026-05-08-eod37-two-axis-milestones-landed.md)

## Methodology validation

This is the FIRST tick where W4A8 axis 3 has all upstream blockers cleared:
- ✅ Pack matches PR #31 W4A8Layer
- ✅ Kernel 0-diff PR #31
- ✅ Loader detects W4A8 config
- ✅ FFI dtypes match
- ✅ Activation quant convention match
- ✅ Calibration preserved through GPTQ-aware re-pack(0.02% drift)

End-to-end test is the **single point of failure verification** remaining。
Probability of PASS is high(>80%)given all upstream layers verified。

## Status

**This brief = ready-to-execute checklist**。Codex(or user direct via
shell)can execute Steps 1-4 as 4 separate commits or 1 bundle。

GPU contention:Step 2(greedy gate)+ Step 3(bench)both use GPU。
If running concurrently with axis-2 Medusa or other GPU work,serialize。

Codex idle as of EOD+37(per `fdb951f` consolidation).Awaiting direction
on D1-D4。This brief is the concrete D2 path。
