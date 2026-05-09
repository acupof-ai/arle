---
title: M_rope-yarn-scaling Phase 3 — CUDA-side Qwen3-4B 64k/128k YARN bench plan
date: 2026-05-10
type: plan
status: ready-post-phase2-step3
---

# Phase 3 — CUDA-side Qwen3-4B long-ctx YARN bench(无需 Mac)

> 修正:Qwen3-4B native ctx = **40960(40k)**(per `infer/models/Qwen3-4B/config.json`),
> 不是 32k。意思:64k YARN factor=2 / 128k YARN factor=4 是 **CUDA-side 可行**
> 的 Phase 3 validation,**无需 Mac**(memory + compute 全 fits RTX 4070 Ti SUPER 16GB)。

## 1. 修正 model + memory 数学(Qwen3-4B specific)

### 实际 Qwen3-4B config

| 维度 | 值 |
|------|----:|
| layers | 36 |
| attention heads | 32 |
| KV heads(GQA)| 8 |
| head_dim | 128 |
| native ctx(`max_position_embeddings`)| **40960** |
| rope_theta | 1_000_000 |
| current rope_scaling | None |

### KV cache size per token

```
K + V = 2 × layers × kv_heads × head_dim × bytes
     = 2 × 36 × 8 × 128 × 2 (BF16)
     = 147 KB / token
```

| ctx | KV (BF16) | KV (FP8) | weight (BF16) | scratch | total |
|-----|----------:|---------:|--------------:|--------:|------:|
| 40k native | 5.9 GB | 2.95 GB | 8 GB(BF16)/ 2.5 GB(W4)| 1-2 GB | **~10 GB**(W4)/ 14 GB(BF16) |
| 64k YARN ×2 | 9.4 GB | 4.7 GB | 8 GB / 2.5 GB | 1-2 GB | **~13 GB**(W4)/ 18 GB(BF16 OOM)|
| 128k YARN ×4 | 18.8 GB | 9.4 GB | 8 GB / 2.5 GB | 1-2 GB | **~14 GB**(W4 + FP8 KV)/ OOM(BF16)|
| 256k YARN ×8 | 37.7 GB | 18.8 GB | 8 GB / 2.5 GB | 1-2 GB | OOM |

→ **64k YARN ×2 fits 16GB on W4-hybrid + BF16 KV**,**128k YARN ×4 needs FP8
KV**(已支持 per ARLE substrate)。256k+ 需 W4 KV OR multi-GPU。

## 2. Phase 3 validation phases — Qwen3-4B CUDA-side

### Phase 3a — 40k native baseline(no scaling)

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh p3a-qwen3-4b-40k-native \
  --concurrencies 1 --max-seconds 60 --warmup 10 \
  --data 'prompt_tokens=40000,prompt_tokens_stdev=1,prompt_tokens_min=40000,prompt_tokens_max=40000,output_tokens=128,output_tokens_stdev=1,output_tokens_min=128,output_tokens_max=128'
```

**License**:server 不 OOM + completion valid + TTFT < 30s。
**记录**:TTFT p50, ITL p50, tok/s, σ across n=3。

### Phase 3b — 64k YARN factor=2 + Qwen3-4B(同 model,custom config patch)

需修改 `infer/models/Qwen3-4B/config.json` OR 加 CLI 注入 rope_scaling:
```json
"rope_scaling": {
  "type": "yarn",
  "factor": 2.0,
  "original_max_position_embeddings": 40960
}
```

```bash
# 需要 cp Qwen3-4B → Qwen3-4B-yarn-2x first OR add CLI override flag
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh p3b-qwen3-4b-64k-yarn2 \
  --concurrencies 1 --max-seconds 90 --warmup 15 \
  --data 'prompt_tokens=64000,prompt_tokens_stdev=1,prompt_tokens_min=64000,prompt_tokens_max=64000,output_tokens=128,output_tokens_stdev=1,output_tokens_min=128,output_tokens_max=128'
```

**License**:
- Server boot + 200 OK + valid completion(server log shows YARN inv_freq applied)
- TTFT < 60s(reasonable for 64k prefill)
- PPL ≤ 1.20× 40k baseline(if PPL eval available)
- σ < 5% across n=3

**KILL**:
- gibberish output → YARN math bug
- OOM → memory budget too tight
- TTFT > 5×(40k baseline)→ unscoped regression

### Phase 3c — 128k YARN factor=4 + FP8 KV

```bash
# Need --kv-cache-dtype fp8 for memory fit at 128k
INFER_PREFILL_GRAPH=1 \
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh p3c-qwen3-4b-128k-yarn4-fp8 \
  --concurrencies 1 --max-seconds 180 --warmup 30 \
  --data 'prompt_tokens=128000,prompt_tokens_stdev=1,prompt_tokens_min=128000,prompt_tokens_max=128000,output_tokens=128,output_tokens_stdev=1,output_tokens_min=128,output_tokens_max=128'
```

**License**:
- 200 OK + valid completion(quality may degrade — informational tier)
- TTFT < 180s
- KV cache pool 不爆(per /v1/stats peak_kv_util check)

**KILL**:
- OOM at 128k → needs W4 KV cache impl OR pool config tune
- output completely incoherent(repetition / single token loop)→ YARN factor=4 too aggressive for Qwen3-4B,need re-tune attention_factor

## 3. CLI rope_scaling 注入(避免改 model config.json)

ARLE 当前 CLI 不支持 rope_scaling 注入(per `infer/src/main.rs`)。3 选项:

| 选项 | LOC | 风险 |
|------|----:|------|
| A. Add `--rope-scaling-yarn FACTOR ORIG_MAX_POS` CLI flag | 30-50 | 低 |
| B. Add env var `INFER_ROPE_SCALING='{"type":"yarn",...}'` JSON parse | 20-40 | 低 |
| C. Patch model `config.json` in place | 0 | 中(覆盖 production config)|

**推荐 A**(CLI flag,opt-in,don't pollute model config)。LOC 30-50。可作为
Phase 3 prep PR(Claude < 100 LOC bound)。

## 4. Phase 3 dependencies + ordering

```
1. M_rope-yarn-scaling Phase 2 step 3 (qwen3/weights.rs caller opt-in)
   ↓ blocks
2. CLI rope_scaling injection (option A above, ~30-50 LOC)
   ↓ blocks
3. Phase 3a/b/c bench validation
   ↓ produces
4. Phase 3 wins entry (PPL + tok/s + memory comparison table)
```

## 5. Cross-references

- M_rope-yarn-scaling plan:`docs/plans/M_rope-yarn-scaling.md`
- Phase 1+2 wins consolidation:`docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md`
- Phase 2 step 3 patch:`docs/plans/2026-05-10-phase2-step3-qwen3-caller-optin-patch.md`
- Qwen3.6 260k 用户课题:`docs/research/2026-05-09-qwen36-35b-a3b-260k-context-feasibility.md`(Mac path,this Phase 3 is CUDA path complement)
- 32k-128k leadership project:`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`

## 6. ROI

- **Phase 3 CUDA path 不需 Mac**,可在 RTX 4070 Ti SUPER 立即 validate(post Phase 2 step 3 + CLI flag)
- 64k YARN bench validates math correctness + memory headroom
- 128k YARN bench validates FP8 KV co-existence
- **直接 unblocks 32k-128k leadership project Phase 2-4**(no longer waiting on Qwen3.6 Mac infra)

## 7. 状态

Phase 3 CUDA-side bench plan ready post Phase 2 step 3 + CLI rope_scaling injection
flag(~30-50 LOC,Claude pickup)。Qwen3-4B 64k YARN factor=2 是 first 真实
long-ctx > native validation,可在 RTX 4070 Ti SUPER 跑(KV memory 13 GB 内 fits)。
