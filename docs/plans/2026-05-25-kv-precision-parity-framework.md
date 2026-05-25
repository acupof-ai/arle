# Plan — KV precision parity framework + audit fix (2026-05-25)

**Status**: Phase 1-3 landed (2026-05-26) · **Driver**: ckl · **Executor**: Claude

## Outcome (2026-05-26)

- Phase 1 (framework) ✅ — `infer/tests/kv_precision_parity.rs` + JSON report
  + 4-precision matrix (BF16/INT8/FP8/TQ4) with knobs.
- Phase 2 (audit on L4) ✅ — see numbers below.
- Phase 3 (fixes for what audit surfaced):
  - ✅ TQ4 prefill routing: paged kernel hard-asserts `page_size==16`; gated
    at three dispatch sites + skipped Pass 3 warmup for non-paged formats.
  - ✅ `auto` KV default flipped to BF16 (away from broken FP8).
  - 🟡 FP8 catastrophic step-1 divergence: reproduced, gated to report-only,
    errors entry filed with Phase 3 diagnostic next steps.
  - 🟡 INT8 long-decode drift (step ~242 at 256-token horizon): observed,
    co-tracked in the same errors entry.
- Phase 4 (CI gate) — not started; harness is local-test ready. Awaiting
  explicit go on `scripts/ci/cuda_full.sh` integration.

### Audit numbers (Qwen3-4B, L4, 2026-05-26)

| Precision | 4×64 mean_match | 8×256 mean_match | Gate | Status |
|---|---:|---:|---:|---|
| BF16 | 1.0000 | 1.0000 | 1.0 | ✅ self-parity |
| INT8 | 1.0000 | 0.8901 (drift step 242) | 0.99 | ✅ short / ⚠️ long |
| FP8  | 0.0156 | 0.0039 (step-1 div) | None (was 0.95) | 🟡 known broken, deferred |
| TQ4  | 0.0000 | 0.0000 (lossy by design) | None (was 0.8) | 🟡 4-bit inherent loss |

### Files touched

- `infer/tests/kv_precision_parity.rs` (new)
- `infer/src/scheduler/cuda/prefill.rs:535` — page_size gate
- `infer/src/scheduler/cuda/core/construction.rs:107-122` — contiguous buffer sizing
- `infer/src/scheduler/cuda/core/warmup.rs:188-205` — skip Pass 3 for non-paged
- `infer/src/model/qwen3/forward.rs:444` — launch_prefill_batch gate
- `infer/src/model.rs:506` — trait default forward_prefill_batch gate
- `infer/src/main.rs:213` + `:1489` — auto-default + CLI doc to BF16
- `docs/experience/wins/2026-05-26-kv-precision-parity-framework-tq4-routing-fix.md` (new)
- `docs/experience/errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md` (new)


## Why now

2026-05 一连串 FP8 KV / TurboQuant / GPTQ INT4 kill (`docs/experience/errors/`
共 17 条):

- FP8 KV 32×256 trajectory gate 两次跪 (05-02 / 05-05),token-1 起发散
- FP8 KV 06 个优化尝试连 kill (05-12) — 改一个 op 不知道连带是否坏了
- TurboQuant 9B fixed-logits kill (05-21),tensor-local 通过 ≠ full-model logits 通过
- TQ4 lm_head kill,GPTQ INT4 loader kill,FP8 compressed-tensors layout kill

**结构性根因 — 没有跨 precision parity test**:

- `infer/tests/` 集成测试全是 BF16,unit test 只测 INT8/FP8 roundtrip
- 改 FP8 时不知道 INT8 / TQ 是不是连带坏了
- 改完没"verified all precisions still green"的 gate

## Surface 已有 (Explore agent 2026-05-25 map)

6 个 precision:

| Precision | Wire tag | Storage | Decode kernel |
|---|---|---|---|
| BF16 | 1 | 2B reference | `decode_attention_paged` |
| INT8 | 3 | 1B + f32 scales | `decode_attention_int8` |
| FP8 E4M3 | 5 | 1B + f32 scales | `decode_attention_fp8` / `_varlen_fp8` |
| TQ2 | 10 | 0.25B + f16 norms | `turboquant_fused_decode_attention` |
| TQ3 | 11 | 0.375B + f16 norms | 同上 |
| TQ4 | 12 | 0.5B + f16 norms | 同上 |

CLI: `--kv-cache-dtype {auto,bf16,fp8,int8,tq2,tq3,tq4}`,parser
`infer/src/main.rs:1441` `parse_kv_cache_mode`。

## Phase 1 — Framework (本次重点)

`infer/tests/kv_precision_parity.rs` + `infer/src/test_utils/kv_parity.rs`

输入:
- model path (default `infer/models/Qwen3-4B`)
- prompt set (32 个 fixed prompt,跟 2026-05-02 gate 同源)
- precision list: `[BF16, INT8, FP8, TQ4]` default,TQ2/3 opt-in 环境变量

每个 precision:
1. Boot scheduler with `--kv-cache-dtype <p>` + `--num-slots 16 --max-seq-len 5120`
2. 跑 32 prompt × 256 decode tokens
3. 抓: (a) 完整 token 序列 (b) 每 step 的 final logits top-K + cosine 准备

Diff (BF16 reference):
- **Trajectory**: `common_prefix_len / 256`(每 prompt 取 mean),报 ≥X% 通过率
- **Logit cosine**: 每 step 32 prompt 平均,报 ≥X 通过率
- **Top-K Jaccard** (K=16): top16 集合 IoU,报 ≥X 通过率

Gate (基于 2026-05-02 教训 + 行业经验,可按实测重校):

| Precision | trajectory match | logit cosine | top16 Jaccard | 备注 |
|---|---|---|---|---|
| BF16 | 100% | 1.000 | 1.00 | self-parity sanity |
| INT8 | ≥99% | ≥0.999 | ≥0.99 | 8-bit 应近无损 |
| FP8 | ≥95% | ≥0.99 | ≥0.95 | E4M3,允许 long tail 微漂 |
| TQ4 | ≥80% | ≥0.95 | ≥0.85 | 4-bit Hadamard,允许漂 |
| TQ3 / TQ2 | report-only | — | — | 不 gate,只输出 |

输出: `target/kv-parity-<model>-<commit>-<date>.json` (per-precision row + 是否通过 gate + 首发散 step / 首发散 prompt id)

**不做** (Phase 1 之外):
- 每层 KV tap (per-layer attribution) — 需 ModelForward debug hook,改动太大,Phase 3 才考虑
- W4A8 / W4FP4 等 weight quant 路径 (本次只 KV)
- nsys / kernel-level micro-benchmark

## Phase 2 — Audit + errors entries

跑 Phase 1 framework on current main 全 precision,产 audit 表:

| Precision | gate 通过? | 首发散 prompt | 首发散 step | 备注 |
|---|---|---|---|---|
| BF16 | ✓ self | — | — | sanity |
| INT8 | ? | ? | ? | 预期 pass |
| FP8 | 大概率 fail | — | — | 沿 2026-05-02 病灶 |
| TQ4 | 不确定 | — | — | 9B 跪了,4B 可能不同 |
| TQ3 | report | — | — | — |
| TQ2 | report | — | — | — |

每条 fail 写 errors entry:
- 复现命令 + bench-output 路径
- 首发散点 prompt/step
- 推测病灶 (引 2026-05 errors 串)
- 不立刻 fix,留待 Phase 3

## Phase 3 — Fix surfaced (按 fail 列表逐条)

预期病灶 (从 errors 串推测,实测 confirm 后再 fix):

1. **FP8 contiguous→paged migration 结构性 quant 不对称**:
   - INT8 path:contiguous 已存 i8 (single-quant at decode write),migration
     纯 byte-copy via `kv_cache_to_paged_int8_range_cuda`
     (`crates/cuda-kernels/src/paged_kv.rs:1656-1675`),scale 同迁移
   - FP8 path:contiguous 存 BF16,migration 时调
     `quantize_scatter_kv_fp8_range` (`paged_kv.rs:1725` + 1738),**在迁移
     boundary 才把 BF16 quant 成 FP8 + 算新 scale**,与 decode 后续写入的
     FP8 scale 计算上下文可能不一致 (prefill chunk vs single-token block)
   - 后果:首个 decode token 读取的 KV 涵盖 [prefill migration 出的 FP8 + 第一
     个 decode 写入的 FP8],两块 quant boundary scales 不同 → token 1 即漂
   - **2026-05-05 next-step #3 已指向**:`decode_attention_varlen_fp8`
     readback vs BF16 dequant 对账。需要 Phase 2 audit confirm 这条 path 是
     不是真正不收敛点
2. **TQ4 lm_head bypass**:9B kill 教训说明 lm_head 不能用 TQ4 envelope,4B
   可能继承同 bug,需要检查 `infer/src/model/qwen3/lm_head.rs` 是否 gate

每个 fix 单独 commit + wins entry,引 Phase 2 errors entry。

## Phase 4 — CI gate

`cargo test --release --test kv_precision_parity --features cuda` 加入
`scripts/ci/cuda_full.sh`,gate BF16+INT8+FP8 default,TQ opt-in。

## 工作约束

- 本地 Mac 写 + cargo check + cargo test (CPU-only 部分 sanity)
- L4 远端 `outdoors-arrow-guide-participate.trycloudflare.com` 跑实际 GPU 测试
- 与 codex (tmux:3 Axes 2+3) 并行不冲突 (不同 module:scheduler vs model/op kv)
- 远端 build ~3min,bench 30-60s per precision per c → 5 precision × 1 config ≈ 5min
- 每次远端 run 前 `rsync --dry-run` 确认无漏

## Acceptance

Phase 1 完成 = harness 在 L4 上能跑通 BF16+INT8+FP8 三档,输出 JSON 报表,且
BF16 self-parity = 100%(harness 自检)。

Phase 2 完成 = 5 precision audit 表填完,每条 fail 有 errors entry。

Phase 3 完成 = 至少 FP8 trajectory match 从 baseline (1.22%) 提升到 ≥80%
(80% 不达 gate 但接近,fix 进度可见),或写 errors entry 说明 deferred + 原因。

## Cross-refs

- 起点 errors:
  - [2026-05-02 FP8 KV trajectory fail](../experience/errors/2026-05-02-qwen3-fp8-kv-numerical-tier1-fail.md)
  - [2026-05-05 FP8 KV trajectory 仍 fail](../experience/errors/2026-05-05-fp8-kv-tier1-still-fail.md)
  - [2026-05-12 FP8 decode shared prefetch kill](../experience/errors/2026-05-12-fp8-kv-decode-shared-prefetch-kill.md)
  - [2026-05-21 TQ 9B FWHT fixed-logits kill](../experience/errors/2026-05-21-arle-turboquant-9b-fwht-fixed-logits-kill.md)
- KV quant 代码主线:
  - `crates/cuda-kernels/src/kv_quant.rs`
  - `crates/cuda-kernels/src/kv_turboquant.rs`
  - `crates/cuda-kernels/src/kv_types.rs`
  - `infer/src/main.rs:1441` `parse_kv_cache_mode`
