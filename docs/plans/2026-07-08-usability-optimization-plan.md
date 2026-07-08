# 易用性系统性优化计划

> Status: Shipped — 2026-07-08
>
> 基于代码级 usability review。每条改进附 file:line 证据。
> 经过 ponytail 过滤：只做解决真实摩擦的，不做看起来好但没人需要的。
> 二次核实：`logprobs` 和 `extra_args` 保留（有用途），`--max-tokens` help 已满足。

---

## 做 vs 不做

| 做 | 不做 | 为什么不做 |
|----|------|-----------|
| README 构建提示 | 配置文件 `config.toml` | `ARLE_MODEL` + shell alias 已够；四级优先级管理是过度设计 |
| CARGO_TARGET_DIR 文档 | OpenAPI `/docs` 端点 | `docs/http-api.md` 已覆盖；1 天工作量等有人抱怨再做 |
| `CompletionRequest::new()` | CUDA-only trait 隔离 | 只有内部跨 backend 代码遇到；半天工作量降为可选 |
| `MultimodalChatRequest::new()` | `max_tokens` OpenAI/Anthropic 对齐 | 行业惯例就是不一样，对齐反而违反"兼容对应官方 API" |
| `ARLE_MODEL` 解析 bug fix | env 变量全面收敛（Metal/DSv4） | 内部传输机制，用户不直接设；等有人抱怨再动 |
| `--max-tokens=0` help 文本 | `logprobs` 移除 | OpenAI API 标准字段，客户端依赖；保留，加注释说明当前永远 null |
| `--no-tools` help 澄清 | `extra_args` 删除 | 捕获 `--` 后参数给明确错误（"不转发到独立 binary"），比 clap 通用 "unexpected argument" 更友好 |
| `AGENT_INFER_MODEL` deprecation warn | | |
| `--bind` warn 清理 | | |
| `--no-cuda-graph` 标注 | | |

---

## Phase 0：构建 UX（15 分钟）

### 0.1 README 提示必须选 backend feature

**问题**：`cargo build --release` 默认 `features = ["cli"]`，产出的二进制不能做任何推理。新用户必踩。

**证据**：根 `Cargo.toml` `[features]` `default = ["cli"]`；`crates/cli/src/lib.rs:173` 报错 `"ARLE requires a local inference backend"`。

**修复**：`README.md` Quick Start 末尾加注。

**成本**：5 分钟。

### 0.2 onboarding.md 提示 `CARGO_TARGET_DIR`

**问题**：在 cuda / metal-no-cuda 两个 feature lane 间切换不设不同 `CARGO_TARGET_DIR` → 几乎全量重编。

**证据**：`docs/onboarding.md` §4 无此提示。

**修复**：§4 底部加 tip。

**成本**：5 分钟。

---

## Phase 1：API 摩擦（25 分钟）

### 1.1 `CompletionRequest::new()` + builder setters

**问题**：每次构造 `CompletionRequest` 必须手写全部 8 个字段，其中 5 个永远是 `None`/`false`。

**证据**：`crates/infer-api/src/types.rs:95-110`；example 和 test 都在重复手写。

**修复**：加 `new(prompt, max_tokens)` + `with_sampling`/`with_stop`/`with_session` builder setters。

**成本**：15 分钟。

### 1.2 `MultimodalChatRequest::new()`

**问题**：`types.rs:139-143` 裸 struct，无构造器。

**修复**：加 `new(messages, max_tokens)` + `with_sampling`。

**成本**：5 分钟。

---

## Phase 2：配置一致性（30 分钟）

### 2.1 `resolve_model_source()` 补 `ARLE_MODEL`

**问题**：`crates/infer-util/src/hf_hub.rs:137` 只查 `AGENT_INFER_MODEL`（旧），不查 `ARLE_MODEL`（新）。用户设 `ARLE_MODEL`，`arle run` 找不到。**这是 bug。**

**证据**：`hf_hub.rs:137` vs `serve.rs:473-479`（`model_from_env()` 两个都查）。

**修复**：env 查找改为先 `ARLE_MODEL`，fallback `AGENT_INFER_MODEL`。

**成本**：5 分钟。

### 2.2 `--max-tokens=0` help 文本

**问题**：默认 0 = auto from config.json，但 `--help` 不说。

**核实**：`args.rs:332-337` doc comment 已详细说明 `0 = auto` 行为。clap 自动将 doc comment 转为 help 文本。**已满足，无需改动。**

**成本**：0 分钟。

### 2.3 `--no-tools` help 澄清

**问题**：顶层（`args.rs:359`）和 `run` 子命令（`args.rs:548`）都有 `--no-tools`，用户困惑。

**核实**：顶层覆盖 REPL/agent 模式（`lib.rs:339`），run 级覆盖 `arle run --no-tools` 单次调用（`lib.rs:307` OR 两者）。两者都有用。

**修复**：顶层 `--no-tools` help 文本加注 "Also honored per-run via `arle run --no-tools`"。

**成本**：2 分钟。

### 2.4 `AGENT_INFER_MODEL` deprecation warn

**问题**：旧变量仍被使用但无迁移提示。

**修复**：命中 `AGENT_INFER_MODEL` 而非 `ARLE_MODEL` 时 `eprintln!` 警告。

**成本**：5 分钟。

---

## Phase 3：Server 清理（10 分钟）

### 3.1 `logprobs` 字段保留

**问题**：response schema 有 `logprobs` 但永远 null。传 `logprobs: true` 的用户被误导。

**证据**：`crates/infer-server/src/schema.rs` `CompletionChoice` 含 `logprobs: Option<...>`，handler 从不填充。

**修复**：保留字段（OpenAI API 标准），加注释说明 "always null — not yet surfaced"。

**成本**：2 分钟。

---

## Phase 4：死代码清理（10 分钟）

### 4.1 `--extra-args` 保留

**问题**：`args.rs` 定义了但 `serve.rs` 始终报错。

**决定**：保留。`#[arg(last=true)]` 捕获 `--` 后参数，给出明确错误信息（"in-process serve stack does not forward to a standalone binary"），比 clap 通用 "unexpected argument" 更有引导性。vLLM/SGLang 用户习惯 `-- --flag` 写法。

### 4.2 `--bind` non-Metal warn 清理

**问题**：`serve.rs:267` 非 Metal + 非默认 bind 打 warning。注释说是历史遗留。

**修复**：grep 确认 CUDA 无特殊处理后删除。

**成本**：5 分钟。

### 4.3 `--no-cuda-graph` 标 CUDA-only

**问题**：`args.rs:351` 是 global flag 但只对 CUDA 有意义。

**修复**：help 文本加 `(CUDA only; no-op on other backends)`。

**成本**：2 分钟。

---

## 可选（等有人需要再做）

- **CUDA-only trait 隔离**（半天）— `CudaTeacherOps` trait，编译期保证
- **OpenAPI `/docs`**（1 天）— `utoipa` + Swagger UI
- **env 全面收敛**（1 天）— Metal `INFER_METAL_*` → `ARLE_METAL_*`，DSv4 旋钮迁 CLI flag
- **配置文件**（1 天）— `~/.config/arle/config.toml`

---

## 执行顺序

```
立即做（~105 分钟）: Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
```
