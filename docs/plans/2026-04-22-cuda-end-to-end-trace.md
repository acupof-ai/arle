# 2026-04-22 CUDA 端到端 Trace 设计

## 目标

- 建立一条可复盘的单请求时间线：`HTTP ingress -> submit -> queue wait -> admission -> prefix/fetch -> prefill/decode launch -> readback -> stream emit -> finish`。
- 让一次 bench 之后能回答两个问题：`TTFT 花在哪`、`请求为什么在 c4+ 开始掉队/不完成`。
- 保持 canonical `guidellm` sweep 的数字可信：性能 bench 和重型 trace 采集分离，trace 用配对诊断跑，不污染主结果。

## 非目标

- 不把 trace 变成默认常开路径。
- 不在本轮直接接 OTLP / Jaeger；先用本地 artefact 跑通。
- 不用 trace 取代 `/v1/stats`；stats 继续承担低开销、持续观测。

## 现状

- `infer` 已有 `--trace-output-path`，通过 `fastrace` 输出 Chrome/Perfetto 可读 JSON：`infer/src/main.rs`。
- 当前请求级 span 主要在 `server_engine`：`generate`、`generate_streaming`、`decode_step`，见 `infer/src/server_engine.rs`。
- `scripts/bench_guidellm.sh` 已经会抓 `/v1/stats` 的 before/during/after 快照，但它只能看到聚合指标，无法把单请求和单个 scheduler tick 关联起来。
- CUDA scheduler 的真实运行顺序已经在 `docs/projects/tiered-kv-runtime-flow.md` 定义清楚，但这条时序还没有被 trace 工具化。

## 设计原则

- **一条 request，一条 root trace。** 根 span 在 HTTP ingress 建立，后续所有 scheduler / KV / stream 事件都挂在同一个 `trace_id` 下。
- **bench 与 trace 分跑。** 主 bench 只保留 `guidellm + /v1/stats`；trace 通过同 commit、同参数、缩小范围的诊断跑补充。
- **事件必须可 join。** 任何关键 span/event 至少带上：`trace_id`、`request_id`、`session_id?`、`scheduler_iter`、`batch_id?`、`slot_id?`。
- **先打通控制面，再细化算子面。** 第一阶段先回答 admission / fetch / launch / readback 卡在哪；kernel 级 root cause 再靠 `nsys` / `ncu` 锚定。
- **单一事实源。** `/v1/stats` 继续输出窗口统计，trace 输出单请求细节；二者通过相同字段命名对齐，不搞第二套指标语义。

## 要回答的诊断问题

- 某个请求的 TTFT 是卡在 `queue_wait`、`active_ttft`、`WaitingFetch` 还是 GPU prefill 本身？
- 某个请求第一次被 admission 的 scheduler iteration 是哪一轮？期间是否被 retract / requeue / cold fallback？
- 某个不完整请求是在什么阶段结束：client cutoff、guidellm time window、slot pressure、fetch backpressure，还是 decode/readback 没跟上？
- `c4+` 的吞吐塌陷到底是 admission 太保守，还是 active set 起不来，还是 fetch/promote 插队导致 decode 空转？

## 端到端 Trace Spine

建议把一条请求的关键节点固定成下面这组 span / event：

| 层 | Span / Event | 关键字段 |
|---|---|---|
| HTTP | `http.request` | `trace_id`, `request_id`, `session_id`, `route`, `stream` |
| Submit | `scheduler.submit` | `prompt_tokens`, `max_tokens`, `priority`, `arrive_ts` |
| Queue | `scheduler.queue_wait` | `waiting_pos`, `waiting_len`, `normalized_prompt_tokens` |
| Prefix | `scheduler.prefix_lookup` | `prefix_hit_tokens`, `lookup_result`, `recompute_advised` |
| Tiered KV | `scheduler.fetch_wait` / `scheduler.fetch_promote` | `ticket_id`, `tier`, `blocks`, `bytes`, `wait_ms`, `fallback_reason?` |
| Admission | `scheduler.admit` | `scheduler_iter`, `slot_id`, `phase`, `batch_id`, `prefill_tokens` |
| Execution | `scheduler.launch_prefill` / `scheduler.launch_decode` / `scheduler.launch_mixed` | `scheduler_iter`, `batch_id`, `decode_count`, `prefill_count`, `token_budget` |
| Readback | `scheduler.decode_readback` | `scheduler_iter`, `batch_id`, `tokens_ready`, `readback_ms` |
| Emit | `http.stream_emit` | `token_index`, `emitted_tokens`, `client_open` |
| Finish | `request.finish` | `finish_reason`, `completed_tokens`, `ttft_ms`, `service_ms`, `e2e_ms`, `incomplete_reason?` |

## 必带关联键

- `trace_id`：整条请求链路的主键。
- `request_id`：HTTP/OpenAI 层已存在的请求标识；用于日志、SSE、trace 对齐。
- `session_id`：多轮 agent / prefix 复用分析需要。
- `scheduler_iter`：当前请求第一次 admission、每次 readback、每次重排都要能落到具体迭代。
- `batch_id`：一次 prefill / decode / mixed launch 的稳定 ID；便于把多个 request 拼回同一 GPU batch。
- `slot_id`：定位活跃工作集是否真的撑到了 `c16`。
- `fetch_ticket_id`：串起 `WaitingFetch -> completion -> promote -> fallback`。

## 分层埋点方案

### Phase 1：HTTP 到 request 生命周期

- 在 `http_server` 建 root span：接收请求时创建 `trace_id`，把 `request_id` / `session_id` / 路由元信息挂上去。
- `IncomingRequest` 增加轻量 trace context 句柄，只做传播，不在热路径分配大对象。
- `request.finish` 收口统一写完：成功、client disconnect、guidellm timeout 造成的 incomplete 都要显式区分。

### Phase 2：Scheduler 控制面

- 在 `runtime.rs::run()` 上给每一轮 scheduler iteration 分配 `scheduler_iter`，只输出轻量 event，不给每个请求都套深层大 span。
- `assign_slots()` / `plan_step()` / `step()` 里只记录对单请求有诊断价值的节点：`queue_wait_end`、`prefix_lookup_result`、`WaitingFetch enter/exit`、`admit`、`retract`、`fallback_to_cold_prefill`。
- `execution.rs::step()` 需要把 `pending_decode` 的 readback 与下一轮 launch 放进同一条时间线，反映当前重构后的真实顺序。

### Phase 3：KV Tier / Readmission

- `ReadmissionPlan`、`submit_fetch()`、`promote_fetched_prefix()` 需要显式产出 `fetch_wait` / `fetch_done` / `promote_done` / `promote_failed` 事件。
- 对所有 fallback 分支记录 `fallback_reason`：`queue_backpressured`、`fetch_submit_none`、`fetch_failed`、`promotion_failed`、`recompute_advised`。

### Phase 4：GPU 执行锚点

- 先不在每个 kernel 上做高开销 trace；只记录 batch 级 launch/readback。
- 需要配对 `nsys` / `ncu` 的时候，用 `batch_id + scheduler_iter` 做 bench anchor，附到 profile 文档里。
- 诊断 decode ceiling 时，额外记录 `launches_per_token`，和 `docs/bench-and-trace-spec.md` §4/§6 对齐。

## 输出物与目录

- **主 bench：** `bench-output/<date>-<label>/` 保持现状，只存 `guidellm` 结果和 `/v1/stats` trace。
- **E2E trace 跑：** `bench-output/<date>-<label>/traces/`，保存 `--trace-output-path` 导出的 JSON。
- **Profile 跑：** `bench-output/<date>-<label>/profiles/`，保存 `.nsys-rep` / `.ncu-rep`，wins/profile 文档里只放 sha256 和截图。
- **文档：** bench 结果进 `docs/experience/wins/`；trace 诊断进 `docs/experience/wins/YYYY-MM-DD-profile-*.md`，必须锚到同 commit、同 workload 的 bench。

## Bench / Trace 配对工作流

1. **先跑 canonical bench。** 不开重型 request trace，只保留 `guidellm` 和 `/v1/stats`。
2. **选诊断腿。** 默认优先：
   - `c1`：确认单请求基线
   - `c4`：当前已知开始塌陷的位置
   - `c16`：看 active set 是否撑满
3. **重跑单腿 trace。** 同 commit、同 model、同 `num_slots/max_seq_len/mem_fraction_static`，但只跑单一 concurrency，开启 `--trace-output-path`。
4. **需要算子根因时再挂 profile。** `nsys` 只抓 `<=1s` steady state，bench 文档引用对应 profile。

这条工作流满足 bench spec：性能数字可信，trace 也有 bench anchor，不会变成孤立 artefact。

## 最小可交付版本（MVP）

- HTTP root trace 已建立，并能把 `request_id/session_id` 传到 finish。
- Scheduler 能输出 `queue_wait_end`、`admit`、`launch_*`、`decode_readback`、`finish` 五类关键事件。
- `WaitingFetch` 与 cold fallback 可见。
- 一次 `c4` 诊断跑能回答：请求在哪个 iteration 被 admission、有没有等 fetch、首 token 前经历了几轮 launch/readback。

## 验收标准

- 对任意一条 trace，能从 artefact 直接读出 `queue_wait_ms`、`active_ttft_ms`、`service_ms`、`e2e_ms`。
- 对任意一个 `c4+` incomplete request，能明确归因到 `client cutoff`、`guidellm window`、`WaitingFetch`、`slot pressure/retract` 或 `other internal failure`。
- 对任意一个 GPU batch，能回答它属于哪一轮 `scheduler_iter`、承载哪些 `request_id`、是 `prefill/decode/mixed` 哪种类型。

## 分阶段落地

- **W1**：把 HTTP ingress 变成 root span，打通 `trace_id/request_id/session_id` 传播。
- **W2**：补 scheduler iteration / admission / launch / readback 事件。
- **W3**：补 KV fetch/promote/fallback 事件。
- **W4**：补 finish/incomplete reason 收口，并把 trace artefact 放进 paired bench/profile 工作流。

## 结论

- 当前代码已经有 request 级 `fastrace` 输出和 bench 级 `/v1/stats` trace，但缺中间那段最关键的 scheduler / KV readmission 关联层。
- 端到端 trace 的核心不是“多打一堆日志”，而是固定一条 join spine：`request_id + scheduler_iter + batch_id + slot_id + fetch_ticket_id`。
- bench 与 trace 必须分跑：主 sweep 负责可信数字，trace 负责解释数字为什么会变坏。
- 最新 `c8/c16` 零 token 诊断见 `docs/experience/errors/2026-04-22-cuda-l4-zero-token-trace-root-cause.md`。
- 最新 `ede0daa` trace smoke + `c8/c16` rerun见 `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c8-c16-trace-ede0daa.md`。
- 最新 `39152ac + localfix` `c16` 端到端瓶颈 trace 见 `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-end-to-end-bottleneck-39152ac-localfix.md`。
- latest async-prefill-overlap `c16` paired bench + service-trace diagnosis:
  `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-18c116d-async-prefill-overlap.md`
