# M_quant Phase 1b — `convert_gptq_w4a16_to_w4a8_marlin.py` smoke verification + end-to-end path

> Verifies `09869bc` Phase 1b shortcut script implementing `8bb57ea`
> correction(re-pack from `*.qweight` U8,not `marlin_qweight`)。
> Smoke test on synthetic input passes — script structurally correct。
> This brief documents end-to-end verification path before full Qwen3-4B
> conversion。

## Smoke verification(Claude this tick)

Synthetic test:`(N=256, K=128, groupsize=128)` random U8 qweight + BF16
scales → repack:

```bash
$ .venv/bin/python -c "import importlib.util; spec=...; m.repack_w4a16_to_w4a8(...)"
OK — repack output shapes:
  qweight = [8, 512] dtype=torch.int32
  s_channel = [1, 256] dtype=torch.float32
  s_group = [1, 256] dtype=torch.float16
```

Output shapes match `pack_w4a8` convention exactly:
- qweight (k/16, n*16/8) = (8, 512) ✓
- s_channel (1, n) = (1, 256) ✓
- s_group (k/gs, n) = (1, 256) ✓

→ Script passes structural verification。Imports + chain works end to end。

## End-to-end test path(codex action after current bench finishes)

### Step 1 — Convert Qwen3-4B checkpoint(~5-10 min CPU)
```bash
.venv/bin/python scripts/convert_gptq_w4a16_to_w4a8_marlin.py \
    --src infer/models/Qwen3-4B-GPTQ-Int4-marlin \
    --dst infer/models/Qwen3-4B-GPTQ-W4A8-marlin
```

Expected:
- Reads ~1154 tensors from src safetensors
- Repacks each `*.qweight` Linear via repack_w4a16_to_w4a8
- Writes new safetensors with `marlin_w4a8_*` naming convention
- Copies non-quantized tensors (embed/lm_head/biases) unchanged
- Updates config.json with `quant_method: marlin_w4a8`

### Step 2 — Verify pack quality(0.5 min CPU)
```bash
.venv/bin/python scripts/diag_w4a8_pack_roundtrip.py \
    --shape 2560 2560 --groupsize 128
```

Expected:`✅ PASS` per existing diag(production-shape round-trip
already confirmed in `0be5967` verbose verification)。

### Step 3 — Greedy_consistency gate(~1-2 min GPU)
```bash
INFER_TEST_MODEL_PATH=infer/models/Qwen3-4B-GPTQ-W4A8-marlin \
    cargo test --release -p infer --features cuda \
    --test greedy_consistency::test_w4a8_vs_bf16_token_diff
```

**Expected outcomes**:
- **PASS** → calibration preserved through re-pack。Phase 1b succeeds
  → proceed to Phase 3 bench
- **FAIL but improved character**(English-frag like H3+H3b state)→
  re-quantization noise additive,need adjustment in pack_w4a8 to use
  GPTQ scales directly instead of re-deriving max-scale。Sub-step 3a:
  modify `pack_w4a8` to accept pre-quantized integer + scale args
- **FAIL same as naive max-scale**(period spam / multilingual gibberish)
  → calibration LOST through re-pack。KILL Phase 1b shortcut,fall back
  to AutoGPTQ-direct path

### Step 4 — Bench guidellm(if Step 3 PASS)
```bash
./scripts/bench_guidellm.sh m_quant-w4a8-gptq-rep \
    --model-path infer/models/Qwen3-4B-GPTQ-W4A8-marlin \
    --backend cuda
```

Expected output:TTFT/ITL/throughput numbers vs:
- BF16 baseline
- W4A16 Marlin (`f6f3af3` license)
- W4A8 naive max-scale (known fast-garbage)

Ship `wins/2026-05-08-w4a8-gptq-rep-canonical-bench.md` per CLAUDE.md
mandatory bench rule。

### Step 5 — Default-on flip decision(if Step 4 numbers good)
- If TTFT/ITL improved vs W4A16 + accuracy preserved → flip default-on
  for W4A8 path
- Update master `2026-05-07-arle-master-strategy.md` §1.2.1.A:
  - W4A8 status `❌` → `✅ via GPTQ-Marlin re-pack`
- Mark task #34 W4A8 accuracy fix complete

## Risk for Step 3

**Re-quantization noise hypothesis**:Phase 1b decodes GPTQ-quantized
weights to FP,then re-quantizes via `pack_w4a8 round(w/s)` with
naive max-scale。

If GPTQ-calibrated weight `w_calib` lives at `q × s_gptq` for q ∈ [-7, 7]:
- `s_pack = max(|w_calib|) / 7` per group
- `s_pack` ≈ `s_gptq` if max element of group is also at integer level 7
- `round(w_calib / s_pack)` recovers `q` exactly when scale ratio = 1
- → Near-zero added noise

But:
- If max-element-of-group is at integer 5 instead of 7 → s_pack ≠ s_gptq
- → quantization re-rounds to different integer levels → calibration drift

Empirical signal in Step 3 will tell。If drift > 1% on greedy gate,
need pack_w4a8 modification to accept GPTQ scales directly。

## Probability estimate

P(Step 3 PASS first try)= 50%(re-quant noise is small but non-zero)
P(Step 3 FAIL but recovers with pack_w4a8 GPTQ-aware path)= 35%
P(Step 3 FAIL needs full AutoGPTQ-direct path)= 15%

Best case Phase 1b timeline:0.5 day to write + 5-10 min convert + test → bench in 1 day total
Worst case:Phase 1b KILL → revert to original M_quant Phase 1 (1d AutoGPTQ generate + 2-3d adapter)

## Cross-references

- M_quant plan: [`662cbbb`](../plans/M_quant-autogptq-marlin-integration.md)
- W4A8 root cause: [`39237b9`](2026-05-08-w4a8-naive-max-scale-too-lossy-need-calibration.md)
- da19d71 Phase 0 reconnaissance
- `8bb57ea` re-pack correction (perm mismatch)
- `09869bc` Phase 1b shortcut script
- Existing GPTQ checkpoint: `infer/models/Qwen3-4B-GPTQ-Int4-marlin/`
- Round-trip diag: `scripts/diag_w4a8_pack_roundtrip.py`
- pack_w4a8: `scripts/quantize_qwen3_w4a8.py:93-137`

## Status

Smoke verification ✅ done(this tick)
Step 1-5 deferred until codex's current W3/W4 admission-fix bench finishes
(currently Working,1h 02m,W4 c=8 advancing 384→516 requests,
prefill_queue=0 stable per pane)。

W3 c=16 + W4 c=8 admission-fix unblock is **the higher-priority strategic
deliverable in flight**(master §7.1 P0.0 axis 1 真 agent workload first
production data)。Phase 1b queues behind it。
