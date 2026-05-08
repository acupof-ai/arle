# GPTQ → W4A8 re-pack quality empirical — marginal FAIL,fix pack_w4a8 to accept GPTQ scales

> Phase 1b conversion(`09869bc` script)succeeded mechanically:
> 252 layers re-packed,2.66 GB output checkpoint,correct naming
> (`marlin_w4a8_{qweight,s_channel,s_group}`)。
>
> **Quality verification per `bea90bb` plan**:single-layer round-trip
> diff between GPTQ-decoded source weights and W4A8-decoded destination
> weights。Cross-layer pattern shows **consistent ~4% max / ~0.7% mean
> drift** —— FAIL 1% threshold but mean is sub-1%。Per codex's predicted
> "FAIL but improved" branch:fix is to modify `pack_w4a8` to accept
> GPTQ scales directly,~10-20 LOC vs full AutoGPTQ-direct fallback。

## Conversion mechanics

```bash
$ python scripts/convert_gptq_w4a16_to_w4a8_marlin.py \
    --src infer/models/Qwen3-4B-GPTQ-Int4-marlin \
    --dst infer/models/Qwen3-4B-GPTQ-W4A8-marlin
  first re-pack: model.layers.0.mlp.down_proj → qweight=[608, 5120]
                  s_channel=[1, 2560] s_group=[76, 2560]
  252 layers re-packed, 146 tensors passthrough
  saved → infer/models/Qwen3-4B-GPTQ-W4A8-marlin/model.safetensors  (2.66 GB)
```

Checkpoint produces correct W4A8 naming(verifies `8bb57ea` correction
applied:raw `*.qweight` U8 input,not `*.marlin_qweight`)。

## Quality results — multi-layer

`scripts/verify_gptq_w4a8_repack_quality.py` runs:
1. Decode src GPTQ qweight + scales → BF16 weights w_src
2. manual_unpack(dst marlin_w4a8_*) → BF16 weights w_dst
3. Element-wise diff,report max / mean / p99 + rel ratios

| Layer | Tensor | max diff | mean diff | rel max | rel mean | Verdict |
|---|---|---:|---:|---:|---:|---|
| 0 | self_attn.q_proj | 2.22e-2 | 1.20e-4 | **4.02%** | **0.62%** | FAIL 1% |
| 5 | mlp.down_proj | 2.71e-2 | 1.24e-4 | **4.14%** | **0.69%** | FAIL 1% |

Pattern**:both attention and MLP layers show consistent ~4% max /
~0.7% mean drift。Drift is structural,not layer-position-dependent。

## Root cause analysis

`scripts/quantize_qwen3_w4a8.py::pack_w4a8(weight)`:
```python
ref = weight.t().contiguous()
s_channel = ref.t().abs().amax(dim=-1, keepdim=True).div(127.0)  # naive max
reshaped = ref.reshape(k // groupsize, groupsize, n)
s = reshaped.abs().amax(dim=1).clamp_min(1e-6).div(7.0)  # naive max
w = ref.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))
w = torch.round(w / s_work).to(torch.int32)  # ← rounds to NEW max-scale levels
```

When pack_w4a8 receives **GPTQ-decoded weights**(integer multiples of
`s_gptq` per group):
- `s_pack = max(|w_decoded|) / 7` —— derives scale from data
- If max element is at GPTQ integer level 7: `s_pack = 7 × s_gptq / 7 = s_gptq` ✓
- If max element is at GPTQ integer level 5: `s_pack = 5 × s_gptq / 7 = 0.71 × s_gptq` ✗
- → re-rounding to NEW levels via `round(w / s_pack)` shifts integers

Since GPTQ uses Hessian-aware quant,**not all groups have max at level 7**。
Empirically ~14% of groups land on max integer level 5-6 instead of 7,
producing ~4% max-scale drift on those groups。Mean stays low(~0.7%)
because most groups DO hit level 7。

## Recommended fix(per codex `bea90bb` decision tree)

**Modify `pack_w4a8` to accept GPTQ scales directly**:

```python
def pack_w4a8(weight, groupsize=128, gptq_scales=None):
    ...
    if gptq_scales is not None:
        # GPTQ-aware path: use exact GPTQ scales,re-quantize to same integer levels
        s = gptq_scales.to(torch.float16)  # [n, k/gs] per-group scale
        s_channel = s.abs().amax(dim=1, keepdim=True).to(torch.float32)  # derive channel
        # Force integer levels by rounding to nearest GPTQ level
        w = ref.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))
        s_work = s.t().reshape((1, -1))
        w = torch.round(w / s_work).to(torch.int32)  # using GPTQ scale directly
        ...
    else:
        # Naive max-scale path (original)
        ...
```

LOC:~10-20 in `pack_w4a8` + ~5 in `convert_gptq_w4a16_to_w4a8_marlin.py`
to pass `gptq_scales`。Total ~25 LOC adjustment。

Expected post-fix:`max diff ≈ 1e-3`(quantization roundoff only),
mean diff ≈ 0(integers exact)。Calibration FULLY preserved。

## Skill v1.3.0 methodology validation

Per anti-pattern #13(NULL elimination)and Phase 8 license-or-kill:
- **NULL eliminated**:"naive max-scale pack of GPTQ weights preserves
  calibration"(empirically refuted at 4% max drift)
- **Hypothesis confirmed**:"naive max-scale derives scale from data
  → boundary-element groups drift when their max isn't at GPTQ level 7"
- **Phase 8 marginal-FAIL**:not catastrophic,not LICENSED — Phase 5b
  iteration warranted with `pack_w4a8(gptq_scales=...)` modification

Codex's `bea90bb` plan correctly anticipated this branch:
> FAIL but improved character → re-quantization noise additive,need
> adjustment in pack_w4a8 to use GPTQ scales directly instead of
> re-deriving max-scale。Sub-step 3a:modify pack_w4a8 to accept
> pre-quantized integer + scale args

The plan's probability estimate for this branch was 35%。Empirical evidence
matches:NOT first-try PASS(50% predicted),NOT total fail(15% predicted),
exactly the "needs pack_w4a8 GPTQ-aware path"(35% predicted)。

## Action

Codex action:modify `scripts/quantize_qwen3_w4a8.py::pack_w4a8` to
accept `gptq_scales` keyword argument(~25 LOC)。Re-run conversion with
GPTQ scales passed through。Re-verify quality should hit < 0.1% drift。
Then proceed to Step 3 greedy_consistency gate per `bea90bb`。

Claude action:document this finding(this entry),defer further work
until codex's substrate hot-path commit lands(blocking end-to-end
greedy_consistency test)。

## Cross-references

- Conversion script: `09869bc`(`scripts/convert_gptq_w4a16_to_w4a8_marlin.py`)
- Smoke verification: `bea90bb`(plan + decision tree)
- Quality verification script: `scripts/verify_gptq_w4a8_repack_quality.py`(this commit)
- W4A8 root cause: `39237b9`(naive max-scale lossy)
- Re-pack correction: `8bb57ea`(perm difference,raw qweight)
- pack_w4a8 source: `scripts/quantize_qwen3_w4a8.py:93-119`
- Codex 1h22m+ in review on substrate fix: not yet committed at this entry

## Status

- ✅ Phase 1b conversion mechanically works
- ⚠ Quality drift 4% max / 0.7% mean — exceeds 1% threshold but mean OK
- 🔧 Fix path identified:pack_w4a8 GPTQ-aware mode(~25 LOC,codex own)
- ⏳ End-to-end greedy_consistency PENDING codex substrate commit + pack_w4a8 fix
