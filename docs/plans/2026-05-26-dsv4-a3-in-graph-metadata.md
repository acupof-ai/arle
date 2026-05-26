---
title: DSv4 A3 — In-graph metadata 灭 D2H readback
date: 2026-05-26
type: implementation plan
status: ready for execution（codex 自上而下推进，每 phase 一组 commit）
owner: ckl
related:
  - docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md
  - docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md
  - docs/experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md
  - https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
---

# DSv4 A3 — In-graph metadata 灭 D2H readback

## Goal

把 DSv4 单 token decode 路径 344-347 个 `cuMemcpyDtoHAsync_v2` 灭到 ≤ 50。
SGLang DSv4 day-0 blog 的对应解法："captured kernels rebuild metadata inside
the graph" —— 让 launch param 由 device-side helper kernel 直接产，next kernel
读 device 上的 metadata 自己 dispatch，整段不出 CUDA graph。

## ARLE 现状 — 5 个 D2H site 完整 inventory（`infer/src/model/deepseek/mlp.rs`，HEAD = `e39429e9`）

| # | 行号 | trace 标签 | Class | 模式 |
|---|---|---|---|---|
| 1 | 1781 | `ffn_route_count_d2h` | A 单卡 decode 入口 | local expert count → 算 offsets/total → 决策 early-return + per-expert launch grid |
| 2 | 3316 | `ffn_deepep_count_by_rank` (AllGather 分支) | B DeepEP | all-rank counts → 算 send/recv pair sizes |
| 3 | 3330 | `ffn_deepep_count_by_rank` (SendRecv 分支) | B DeepEP | send rank counts → 同上 |
| 4 | 3462 | `ffn_deepep_count_exchange` | B DeepEP | recv rank counts → 算 recv offsets/total |
| 5 | 3748 | recv local count D2H | B DeepEP（recv 后再次 local 分组） | DeepEP 收齐后 → local expert counts → 同 Site 1 |

每层 ≥ 1 个 D2H（Class A 必走），DeepEP 路径再 +3。61 layer × (1+3) ≈ 244 D2H/
token。加上 sampler / scheduler 杂项的 ~50 个 = 实测的 344–347（L5 binding-
constraints 实证）。

## Class A vs Class B 的根本差异

- **Class A — Expert dispatch grid dims（site 1, 5）**：host 拿到 counts 用来给
  每个 expert 的 grouped GEMM 计算 grid dim。每 expert 一次 launch、64 expert/
  rank → 每层最多 64 次 launch + 各自 host-side dims。SGLang 用 **persistent
  grouped-GEMM kernel**：device 自己读 counts，在 kernel 内做 per-expert work
  dispatch，一次 launch 干完所有 active expert。

- **Class B — DeepEP collective sizes（site 2, 3, 4）**：host 拿到 counts 用来
  给 NCCL all-gather / send-recv 计算 send/recv buffer sizes。NCCL API 接的是
  host-side counts，**无法直接 device 化**。SGLang 在 DeepEP 内部用
  **NVSHMEM symmetric memory + device-side dispatch**，collective 本身读
  device counts、按 symmetric pointer 自己路由。ARLE 要切到 DeepEP 的
  device-count 模式。

## Phased plan（按 PR-able tranche 切，每 phase 独立 license-or-kill）

### Phase 1 — Class A 的 host-side scan + H2D 灭口

**Scope**：仅 site 1（mlp.rs:1781）。Site 5（3748）下个 phase 复用同款。

**Change**：
1. 新增 CUDA kernel `dsv4_exclusive_scan_i32_cuda(in_counts, out_offsets, n, stream)`
   —— 简单的 single-block scan（Hillis-Steele 或 Brent-Kung，n=experts_per_rank ≤ 64
   一个 block 搞定），放 `crates/cuda-kernels/csrc/moe/dsv4_route.cu`，FFI 在
   `crates/cuda-kernels/src/ffi/moe.rs`。
2. mlp.rs:1779-1798 改造：
   - 删除 `clone_dtoh(&local_counts)` + host loop + `offsets_host` Vec 构造
     + 后续 `clone_htod(&offsets_host)` 这段
   - 替换为 `dsv4_exclusive_scan_i32_cuda(local_counts, offsets_gpu_alloc, ...)`
     直接产 device 上的 offsets
   - 但 `total_local_routes`（early-return + tensor sizing）仍需 host 知道。
     方案：scan 的最后一个 thread 把 total 写到 device 上独立的 i32 slot →
     **一次 D2H 拿 total**（payload 4 byte），整段从"D2H counts + host scan +
     H2D offsets"（payload 256 byte counts + 256 byte offsets，两次 sync）
     压成"一次 D2H total"（4 byte，一次 sync）。
3. `forward_compact_local_routes_gpu` 签名改：原来吃 `&counts_host` /
   `&offsets_host` / `total_local_routes`，改成吃 device 上的
   `counts_gpu` / `offsets_gpu` + host 上仅传 `total_local_routes`（一个
   scalar）。
4. 删除原本的 `offsets_gpu = ctx.stream.clone_htod(&offsets_host)`。

**PASS**：
- nsys 单 token decode：site 1 上的 2 次 sync（D2H + H2D）压到 1 次 D2H
  scalar。整段 wall-clock 减少（site 1 自身 NVTX scope）≥ 30%。
- 全 token wall-clock：不退步（早 PASS 阈值，因为 Class B 还在）。
- output greedy byte-identical。

**KILL**：
- scan kernel + scalar D2H 比原 host loop 还慢（极小 n 上 host loop 可能赢）。
- 数值不一致（host scan vs device scan 误差，i32 上不应该有但要测）。
- early-return 路径 `total_local_routes == 0` 退化（应该极少 hit）。

**Implementation boundary**：
- `crates/cuda-kernels/csrc/moe/dsv4_route.cu`（新增 scan kernel）
- `crates/cuda-kernels/src/ffi/moe.rs`（FFI 声明）
- `infer/src/model/deepseek/mlp.rs`（call site 重构 + signature 改）
- 不动 `forward_compact_local_routes_gpu` 的 device-side 逻辑，只改它读 device
  metadata。

### Phase 2 — Class A 的 persistent grouped-GEMM kernel

**Scope**：site 1 + site 5 都受益（同 Class A）。

**Change**：
1. 新增 `dsv4_grouped_gemm_persistent_cuda(weights, packed_hidden, counts,
   offsets, out, n_experts, ...)` —— 一个 persistent kernel，grid =
   `n_experts × ceil(max_seq / TILE)`（max_seq 取 maxes 上限或单独 D2H
   max_routes），每个 block 查 device 上的 `counts[expert_id]`，如果 0 直接
   return；否则用 `offsets[expert_id]` 拿到 packed_hidden 起点跑 grouped GEMM。
2. mlp.rs 删掉 per-expert launch loop（1820+ 的 64 次 launch 循环），换成单
   launch persistent kernel。
3. site 5 周边路径同款重构。

**PASS**：
- 单 token decode Class A 相关 D2H：site 1 上的 scalar D2H 也可以省（如果 grid
  dim 用 max_routes 上限而不是实际值）→ Class A 路径完全 0 D2H。
- wall-clock 单 token decode 减少 ≥ 5%（PASS 主门）。
- output greedy byte-identical。

**KILL**：
- persistent kernel register pressure 超阈、occupancy 退步、整体 wall-clock
  退步。
- max_routes 上限太大导致 grid 浪费 → 单 launch 比多 launch 还慢。
- 数值不一致（grouped GEMM 精度漂移）。

**Implementation boundary**：
- `crates/cuda-kernels/csrc/moe/`（新增 persistent grouped GEMM kernel）
- `crates/cuda-kernels/src/ffi/moe.rs`
- `infer/src/model/deepseek/mlp.rs`（call site 大改：64 launch → 1 launch）
- 保留旧 per-expert launch 路径作 fallback 开关（`DSV4_PERSISTENT_GROUPED=0/1`
  env），按 phase 2 PASS 后默认 ON。

### Phase 3 — Class B 的 DeepEP device-count 模式

**Scope**：site 2, 3, 4（DeepEP all-gather + send-recv + recv count 三处）。

**Change**：
1. 调研 DeepEP 当前 API：是否支持 device-count `dispatch_async` 入参（NVSHMEM
   symmetric memory backend）？
2. 如果支持：把 `comm.moe_all_gather_i32` / `moe_grouped_send_recv_i32` 切到
   device-count 模式，删除 site 2/3/4 三个 D2H + 配套 host-side
   offsets/counts/sizes 构造。
3. 如果 DeepEP 不直接支持：考虑跳过 Phase 3 或先 land Phase 1+2 拿单卡 win，
   多卡 win 等 DeepEP 上游或自写。

**PASS**：
- 多卡 nsys 单 token decode：D2H ≤ 50。
- 8-rank H20 wall-clock 单 token decode 减少 ≥ 5%。
- output byte-identical。

**KILL**：
- DeepEP device-count 路径在 H20 / NVLink 上 latency 比 host-count 路径还高
  （symmetric memory 在 H20 上 fast path 可能没 H100 那么 mature）。
- 把 DeepEP fork / 改造的工作量超出 A3 axis 边界 —— 升级为独立 axis A1.5。

**Implementation boundary**：
- `crates/cuda-kernels/src/ffi/`（DeepEP FFI 扩展）
- `infer/src/model/deepseek/mlp.rs`（call site 重构）
- **如果需要改 DeepEP 上游：先 ping ckl 走 architecture license**（这条踩了
  backlog Rules 的 A1 architecture-license 门，因为变成跨 lib 改动）。

## 跨 phase 共用规则

- 一 phase 一组 commit，message 用 `feat(cuda):` 或 `feat(dsv4):` scope。
- 每 phase PASS / KILL 落 `docs/experience/wins/` 或 `errors/`，按
  `CLAUDE.md §Benchmarks` 强制 entry。
- bench 命令固定：复用 `/sgl-workspace/bench-artifacts/dsv4-longseq-20260525/
  request.json` 同 prompt；`max_tokens=1` 只能标成 prefill/TTFT smoke，不能
  当 decode 证据；decode / wall-clock PASS/KILL 用 `max_tokens>=32`，最终按
  32K input / 1.5K output、c=8、qps=8 的 DSv4 SLO framing 对齐。
- nsys 对照取 `--cuda-flush-interval=1000`，命名 `/tmp/dsv4_a3_phase{N}_
  {before|after}.nsys-rep`。
- wall-clock framing 强制：每 phase 的"减少 X%"必须用 per-request total ms 算，
  不准用 nsys narrow window 占比（CLAUDE.md §0 SOLID anchor）。
- single-variable：每 phase 内部不许同时改其它 axis。
- 旧路径要保留 env 开关作 fallback（`DSV4_A3_PHASE1=0` etc），方便 KILL 时
  回滚比对。

## 推荐执行顺序

```
Phase 1 (site 1 scan + scalar D2H)
   ↓ PASS → commit + push + wins entry
Phase 2 (site 1 + 5 persistent grouped GEMM)
   ↓ PASS → 同上
Phase 3 (DeepEP device-count, conditional on API)
   ↓ PASS → 同上；KILL → 升 A1.5 axis 单写 plan
```

Phase 1 估算：~4-6 小时 codex 时间（写 scan kernel + 重构 site 1 + build
+ bench + nsys 对照）。
Phase 2 估算：~1-2 天（persistent kernel 是新写、要 sweep register/occupancy）。
Phase 3 估算：~2-3 天（DeepEP API 调研 + 重构 + 多卡 bench）。

## Execution log

- 2026-05-26 Phase 1 landed as device-side local-route offsets with
  `DSV4_A3_PHASE1=0` fallback. Result: H2D activity drops 546 → 352 calls
  and 26,240 B → 1,408 B in single-token decode nsys; D2H remains 344 calls;
  longseq output is byte-identical and wall-clock is flat/slightly positive
  (31.3438 s → 31.3414 s for `max_tokens=1`, 36.4854 s → 36.4439 s for
  `max_tokens=64`). See
  [`../experience/wins/2026-05-26-dsv4-a3-phase1-device-offsets.md`](../experience/wins/2026-05-26-dsv4-a3-phase1-device-offsets.md).
- 2026-05-26 Phase 2 route-grouped reuse was KILLed as a default path. The
  opt-in path can delete decode-window D2H (344 calls → 0 in the filtered
  nsys summary) and a rounding-order fix restored longseq byte identity, but
  wall-clock failed the gate: short decode improved only 0.7891 s → 0.7528 s
  (-4.60%, below the 5% PASS threshold) and longseq `max_tokens=32` regressed
  108.7749 s → 110.2519 s (+1.36%). Keep
  `ARLE_DSV4_ROUTE_GROUPED_EXPERTS` default-off; the next Phase 2 attempt must
  be true persistent grouped GEMM/DeepGEMM-style dispatch, not more tuning of
  the route-wise GEMV prototype. See
  [`../experience/errors/2026-05-26-dsv4-a3-phase2-route-grouped-kill.md`](../experience/errors/2026-05-26-dsv4-a3-phase2-route-grouped-kill.md).
- 2026-05-26 Phase 2 native DeepGEMM required mode reached real decode after
  toolchain/runtime hardening (NVCC JIT, per-CUDA-context runtime handles, and
  SFA stride support for reused scratch), but was KILLed as an optimization:
  `max_tokens=32` A/B mean was native 3.7632 s vs DeepGEMM 7.5347 s (+100.2%),
  and greedy output was not byte-identical. Keep
  `ARLE_DSV4_EXPERT_BACKEND=deepgemm` as the required/fail-fast toolchain
  validation mode, not a claimed optimization win. See
  [`../experience/errors/2026-05-26-dsv4-a3-phase2-deepgemm-kill.md`](../experience/errors/2026-05-26-dsv4-a3-phase2-deepgemm-kill.md).
- 2026-05-26 user licensed pivot to make DSv4 default to DeepEP-style
  dispatch/combine and the DeepGEMM auto expert backend. This changes the
  default integration path only; the previous wall-clock/correctness KILL still
  blocks claiming DeepGEMM as a performance win until native DeepEP LL +
  byte-identical grouped expert compute are validated. Default-path build,
  `max_tokens=32` smoke, and nsys evidence:
  [`../experience/wins/2026-05-26-dsv4-default-deepep-deepgemm.md`](../experience/wins/2026-05-26-dsv4-default-deepep-deepgemm.md).
- 2026-05-26 DeepGEMM native bridge now caches `cudaDeviceProp` per thread and
  current CUDA device. This removes `cudaGetDeviceProperties_v2_v12000` from
  the hot decode API frame and improves `max_tokens=32` smoke wall-clock
  12.1466 s → 8.2378 s, but does not complete A3 because D2H calls remain.
  Evidence:
  [`../experience/wins/2026-05-26-dsv4-deepgemm-device-prop-cache.md`](../experience/wins/2026-05-26-dsv4-deepgemm-device-prop-cache.md).
- 2026-05-26 padded B=1 DeepGEMM local experts now keep recv-side local counts
  and offsets on device by default (`ARLE_DSV4_DEEPGEMM_DEVICE_COUNTS=0` is the
  opt-out). This closes the A3 short-decode D2H gate for the default
  DeepEP+DeepGEMM path: D2H memcpy activity 10,711 calls / 1,365,180 B → 11
  calls / 44 B, warmed nsys profile request 3.3974 s → 3.1908 s (-6.08%),
  greedy output byte-identical for `max_tokens=32`. Final SLO evidence still
  needs the 32K / 1.5K, c=8, qps=8 frame. Evidence:
  [`../experience/wins/2026-05-26-dsv4-deepgemm-device-counts.md`](../experience/wins/2026-05-26-dsv4-deepgemm-device-counts.md).
- 2026-05-26 native DeepEP process-model gate says A3 Phase 3 cannot be a
  same-process drop-in. Official DeepEP multi-process LL/intranode tests pass
  on the target 8xH20 shape, but ARLE same-process 8-thread LL times out and
  same-process intranode fails at `cudaIpcOpenMemHandle` with
  `invalid device context`. Treat native DeepEP as the highest-priority
  communication axis, but enter through a process-per-rank transport design.
  Evidence:
  [`../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md`](../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md).

## Cross-refs

- A3 axis backlog 入口：
  [`../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`](../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md)
- L1-L6 binding constraints 实证表：
  [`../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`](../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
- SGLang DSv4 V0 工程动作集：
  https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
