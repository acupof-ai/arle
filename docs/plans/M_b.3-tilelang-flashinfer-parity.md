# M_b.3 — TileLang HD128 mixed-batch FlashInfer parity (grid + launch collapse)

> Sequel to [`M_b-tilelang-hd128-decode.md`](M_b-tilelang-hd128-decode.md)
> + [`M_b.2-tilelang-hd128-fp8-decode.md`](M_b.2-tilelang-hd128-fp8-decode.md).
> 用户校正方向(2026-05-07):mixed-batch kernel **对标 FlashInfer
> `BatchPrefillWithPagedKVCacheWrapper` / `BatchDecodeWithPagedKVCacheWrapper`**。
> M_b.3 只覆盖既存 plan 没覆盖的 **mixed-batch 增量 scope**(grid +
> launch collapse);split-KV 由 M_b.2.2 负责,FP8 套用由 M_b.2 A1/B 负责,**不重复**。

## Priority & ROI

**Current rank:P1**(P0 = M_b.2.2 split-KV;P0' P0'' 已闭合,见 `5cacdcb` / `786a20a`)

| Track | Cost | ROI evidence | Negative case |
|---|---|---|---|
| **P0 M_b.2.2 split-KV in TileLang HD128 BF16** (老 plan,未起步) | est 4-6h | BF16 长 ctx decode 当前**单 CTA 串行扫整条 KV**(`grep "split"` 在 `batch_decode_paged_hd128.py` 0 命中);hand-CUDA `decode_attention_varlen_fp8.cu` 已实现 `kMaxSplits=16` 可镜像。M_b.1 wins entry baseline 给出可对标数字。 | 短 KV(<2048 token)split overhead 抵消收益 — 需要 dispatcher 在 KV 长度阈值下分流。 |
| **P1 M_b.3 G1 segment-aware grid** (本 plan) | est 6-10h | `batch_prefill_paged_hd128.py:73-78` 当前 `grid_x = ceildiv(max_qlen, BLOCK_M)` 全 batch 共用 — c=4 mixed(1 prefill qlen=4096 + 3 decode qlen=1)→ `grid_x=64`,**decode rows 在 bx=1..63 全空转**(每 row 64 个 CTA 只 bx=0 干活,其他 launch 后 if-out)。FlashInfer BatchPrefill 用 `qo_indptr` segment-aware persistent CTA scheduler 解决这条。 | host-side plan 阶段付出 CPU cost,短 batch + 短 prefill 时 plan overhead 抵消 SM 利用率收益 → 需要 fast-path bypass。 |
| **P2 M_b.3 G2 mixed prep collapse** (本 plan) | est 3-5h | `batch_decode.rs:1147-1292` 当前 per-prefill 一次 `prep_cuda` launch + 一次 page-table H2D。c=4 → 每 step 多 4 launch + 4 H2D;nsys 数据待 M_nsys P0(`9b1fb8c` 已 land)出。 | nsys profile 若显示 mixed step launch overhead < 5% wall time → kill criteria 触发 → 弃。 |
| **P2 M_b.2 A1/B FP8 套用** (老 plan,A0 已 done `c865f4b`) | est 8-13h | M_b.2.2 split-KV land 后,FP8 路径套用同样 grid → 收益叠加。 | TileLang FP8 cast 比 hand-CUDA `__nv_fp8_e4m3` PTX 慢 → kernel 回归。M_b.2 plan 已含 kill criteria。 |
| **P3 M_b.3 G3 decode kernel segment-aware grid** (待 G1 验证后再评估) | TBD | 若 G1 在 prefill 上验证有效 + nsys 显示 decode kernel grid 也是瓶颈,镜像方案到 `batch_decode_paged_hd128.py`。 | 当前没足够 evidence,先不规划。 |

**Evaluation basis**(bench/code 实证,非 hand-waving):

- **P0' 闭合 baseline**(`786a20a`):longctx 4k/c=4 TTFT p50 = **1976.4 ms**(F4-Small + default Split)。这是 M_b.3 优化的对照基线。
- **132% 回归证据**(`d083b2e` Phase 1A v3 bad run):Mixed re-enable → TTFT 7905.4 ms = +132%。说明**当前 mixed kernel 实现劣于 split**,kernel-side 优化(本 plan)是关键。
- **Grid 一刀切实证**:`tools/tilelang/batch_prefill_paged_hd128.py:73-78` 源码:
  ```python
  with T.Kernel(
      T.ceildiv(max_qlen, BLOCK_M),  # ← 全 batch 共用 max_qlen
      num_q_heads,
      batch_size,
  )
  ```
- **per-prefill launch 实证**:`infer/src/model/qwen3/batch_decode.rs:1255-1292` for-loop。
- **vLLM 数字待补**:CLAUDE.md `support-matrix` 暂无 longctx 4k/c=4 vLLM 直接对照。**G1 land 前要先跑 vLLM baseline 进入 wins entry**(per `bench-and-trace-spec.md`)。

## Scope(只含老 plan 未覆盖的增量)

### G1 — Prefill kernel segment-aware grid

**目标**:`tools/tilelang/batch_prefill_paged_hd128.py` 的 grid 从
`(ceildiv(max_qlen, BLOCK_M), num_q_heads, batch_size)` 改为
host plan 阶段计算的扁平 `(work_item, num_q_heads)`,其中 work_item =
(request_id, q_tile_id) 对,只包含每个 request 实际需要的 q_tile 数。

**对标**:FlashInfer `BatchPrefillWithPagedKVCacheWrapper.plan()` 的 work-stealing 调度。

**实现路径**:
1. Host 侧 plan(`infer/src/ops/attention.rs::tilelang_tc_run_layer` 或一个新 `plan_mixed_batch` helper)从 `qo_indptr` 算 work item 列表,H2D 上传成 `work_indptr` / `work_request_ids` / `work_tile_ids`
2. TileLang kernel grid 改为 `(num_work_items, num_q_heads)`,kernel 内通过 `work_request_ids[bx]` 查 request,用 `work_tile_ids[bx]` 算 q-tile 起点
3. 兼容 pure-prefill / pure-decode / mixed 三种 shape

**Acceptance**:
- mixed batch (c=4 longctx 4k) bench:Mixed mode TTFT ≤ Split mode TTFT × 0.95(即 Mixed 至少 5% 优于 Split,否则 G1 没价值)
- Pure-prefill / pure-decode 路径性能不回归(±2% 以内)

**Kill criteria for G1**:
- 实现完 mixed bench Mixed mode TTFT > Split mode TTFT → 假设错误,弃 G1
- 实现复杂度爆炸(>1500 行 diff)→ 暂停,重新评估方案是否要走 hand-CUDA

### G2 — Mixed `prep_cuda` 单 launch 化 + page-table H2D 收集

**目标**:消除 `infer/src/model/qwen3/batch_decode.rs:1147-1292` 的 per-prefill loop。

**实现路径**:
1. **kernel side**:`prefill_attention_paged_prep_cuda` → `mixed_prefill_paged_prep_cuda`,接受 `qo_indptr` + 多 prefill 的拼接 page table,kernel 内段内分支
2. **host side**:H2D 一次拼好所有 prefill 的 page table(用 `kv_indptr` / `kv_indices` 复用现有数据结构)

**前置**:G1 land 之后做(grid 重做后 prep 协同改更顺)。

**Acceptance**:
- nsys profile 显示 mixed step kernel launch count 减少 ≥ 60%(c=4 时 5 launch → 2 launch)
- mixed bench TTFT 进一步改善 ≥ 3%(launch overhead 锦上添花)

**Kill criteria for G2**:
- M_nsys profile 显示 mixed step launch overhead < 5% wall time → 放弃 G2(锦上添花没必要)

## Out of scope(由其他 plan 负责,本 plan 不重复)

- **TileLang HD128 BF16 split-KV** — `M_b.2.2`(老 plan,task #13)
- **TileLang HD128 FP8 path A1/B** — `M_b.2-tilelang-hd128-fp8-decode.md`
- **HD256 mixed kernel** — Qwen3.5 路径,先在 HD128 验证后再决定是否端口
- **causal mask in mixed prefill+decode** — 已在 M_b.1 处理(老 plan 备注)

## Dependency / 启动顺序

```
M_b.2.2 split-KV (P0, task #13)
    ↓
M_b.3 G1 segment-aware grid (P1, task #14)
    ↓
M_b.3 G2 prep collapse (P2, task #15)
    ↓
M_b.2 A1/B FP8 套用 (P2, M_b.2 plan)
```

为什么这个顺序:
- **M_b.2.2 先**:长 ctx decode 主瓶颈,有 hand-CUDA 模板可镜像,风险最小 ROI 最高
- **G1 第二**:mixed batch 主瓶颈,但 host plan 改动复杂,要先有 split-KV 把 KV-axis 收益拿走再看 grid-axis 增量
- **G2 第三**:nsys evidence-driven,看 G1 之后 launch overhead 还剩多少
- **FP8 套用最后**:BF16 端验证完模式后,FP8 直接套用同结构

## Out-of-band gates per CLAUDE.md

- 起步前必跑 vLLM longctx 4k/c=4 baseline(per `feedback_docs_priority_roi_evidence.md` evaluation basis 要求)
- G1 land 必跑 `scripts/bench_guidellm.sh m_b3-g1-mixed-grid` + `--scheduler-mixed-policy mixed` 对照 default split
- G2 land 必跑 nsys 出 launch count 数字进 wins entry
- 不允许 silent skip:CLAUDE.md "MANDATORY — every runtime change produces a bench entry"

## Suggested first move

**等用户对启动顺序拍板**(默认建议:M_b.2.2 先开)。然后:
1. paste M_b.2.2 directive 给 codex(模板:镜像 `decode_attention_varlen_fp8.cu` kMaxSplits=16,用 TileLang IR 重写)
2. 让 codex 起步前先跑 vLLM baseline 进 wins entry
3. M_b.2.2 land + bench → 评估 G1 是否仍是 P1(可能 split-KV 已经把长 ctx 拉平,G1 ROI 重估)

## Cross-references

- 既存 plan(本 plan 引用,不重复):
  - [`M_b-tilelang-hd128-decode.md`](M_b-tilelang-hd128-decode.md) — M_b.1 done + M_b.2.2 split-KV pending
  - [`M_b.2-tilelang-hd128-fp8-decode.md`](M_b.2-tilelang-hd128-fp8-decode.md) — A0 done + A1/B pending
- P0' 闭合证据:
  - [`5cacdcb`](#) `fix(scheduler): default mixed prefill to split`
  - [`786a20a`](#) wins:close M3.9 mixed policy bench(−41.9% TTFT)
  - [`adb2757`](#) wins:Phase 1A v3 fix validated +25.6%
- 数据收集工具:
  - [`9b1fb8c`](#) M_nsys P0 SIGUSR1/SIGUSR2 → cuProfilerStart/Stop(M_b.3 G1/G2 nsys 数据来源)
- Memory rules:
  - `feedback_docs_priority_roi_evidence.md` — Priority/ROI/negative case/kill criteria 强制
  - `feedback_no_half_states.md` — G1 实现要么完整要么 revert
