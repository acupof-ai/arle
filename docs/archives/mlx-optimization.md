# 从 216 tok/s 到 354.9 tok/s：agent-infer MLX 后端性能优化实录

> 一次性能调优的完整记录：如何找到真正的瓶颈，用三行代码消除 38% 的性能差距，以及为什么 KV 量化在 Apple Silicon 上反而会让你变慢。

---

## 背景

[agent-infer](https://github.com/your-org/agent-infer) 是一个用 Rust 编写的高性能 LLM 推理引擎，支持多轮对话、工具调用和连续批处理。除了 CUDA 后端，我们还为 Apple Silicon 用户提供了基于 [MLX](https://github.com/ml-explore/mlx) 的原生推理后端——可以在 MacBook Pro 上直接运行 Qwen3 系列模型，无需任何外接 GPU。

在集成 MLX 后端的早期阶段，我们发现了一个令人困惑的性能问题：同样的模型、同样的硬件，agent-infer 的 MLX 后端比 mlx-lm 的 native 模式慢了 **38%**。

这是调查和修复这个问题的完整过程。

---

## 第一步：确认问题

测试环境：Apple M2 Max，96GB 统一内存，Qwen2.5-7B-Instruct（4-bit 量化）。

初始性能对比：

| 工具 | 吞吐量 | 说明 |
|------|--------|------|
| mlx-lm native | 352 tok/s | 官方 CLI，单请求 |
| agent-infer (MLX) | 216 tok/s | 16 并发请求，总吞吐 |
| 差距 | **-38%** | 无法接受 |

agent-infer 的 HTTP 服务器是 OpenAI 兼容接口，测试时用 16 个并发客户端持续发请求，模拟真实使用场景。216 tok/s 是总吞吐，并非每个请求的单独速度。乍看之下还不算太差——毕竟有 16 个并发——但和 mlx-lm 单请求的 352 tok/s 相比，差距太明显了。

---

## 第二步：错误的猜测

第一直觉是 SSE（Server-Sent Events）flush 频率问题。我们的流式输出是逐 token 推送的，每个 token 都会触发一次 HTTP flush。网络栈的 overhead 是否拖慢了整体速度？

将 flush 频率从每 token 一次改为每 8 token 一次（batch flush），重新测试：

```
修改前：216 tok/s
修改后：219 tok/s
提升：+1.4%
```

提升微乎其微。SSE 不是瓶颈。

---

## 第三步：找到真正的根因

回头审视架构。MLX 后端的推理流程是这样的：

```
HTTP 请求 → asyncio handler → MLXEngine.generate_stream() → mlx-lm stream_generate()
```

`generate_stream()` 在内部用 `loop.run_in_executor()` 把同步的 `stream_generate()` 丢进线程池：

```python
# mlx_backend.py
async def generate_stream(self, prompt: str, params: SamplingParams):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        for token in stream_generate(self._model, self._tokenizer, prompt, ...):
            loop.call_soon_threadsafe(queue.put_nowait, token)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    loop.run_in_executor(None, _run)

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token
```

看起来没问题？问题在于**当 16 个并发请求同时调用时发生了什么**：

16 个线程同时调用 `stream_generate()`，同时向 Metal GPU 提交命令。MLX 的 Metal 后端本身是线程安全的，但当多个执行流同时占用同一块 GPU 时，命令会交错执行——Metal command buffer 互相争抢，GPU 在频繁切换上下文中浪费了大量时间。

这不是 Python GIL 的问题（`run_in_executor` 绕过了 GIL），而是 **Metal GPU 级别的并发竞争**。

验证方式很简单：把并发请求数从 16 降到 1，单请求速度立刻恢复到 340+ tok/s。确认。

---

## 第四步：修复

根因清楚了，修复方案也很明确：**串行化 MLX 调用**。Metal GPU 一次只能高效运行一个推理流，所以我们用 `asyncio.Lock` 确保同一时间只有一个请求在执行 MLX 推理：

```python
# mlx_backend.py
class MLXEngine:
    def __init__(self, ...):
        self._lock = asyncio.Lock()  # 新增
        ...

    async def generate_stream(self, prompt: str, params: SamplingParams):
        async with self._lock:  # 串行化
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _run():
                for token in stream_generate(...):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
                loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(None, _run)

            while True:
                token = await queue.get()
                if token is None:
                    break
                yield token
```

同时做了两个附加优化：

**1. Batch flush（8 token/frame）**

虽然 SSE flush 不是主要瓶颈，但减少 asyncio 事件循环的调度次数仍有帮助。将 token 缓冲到 8 个再一起推送，降低系统调用频率：

```python
# mlx_server.py - streaming handler
buffer = []
async for token in engine.generate_stream(prompt, params):
    buffer.append(token)
    if len(buffer) >= 8:
        yield format_sse_chunk(buffer)
        buffer.clear()
if buffer:
    yield format_sse_chunk(buffer)
```

**2. orjson 替换 stdlib json**

HTTP 响应的 JSON 序列化从标准库 `json` 换成 [orjson](https://github.com/ijl/orjson)。orjson 是用 Rust 写的，对简单结构的序列化速度快 3-10 倍：

```python
# 修改前
import json
body = json.dumps({"choices": [{"text": token}]})

# 修改后
import orjson
body = orjson.dumps({"choices": [{"text": token}]})
```

---

## 第五步：测试结果

修改后重新测试（相同硬件，相同模型，16 并发）：

| 指标 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| 总吞吐（tok/s） | 216 | **354.9** | **+64%** |
| vs mlx-lm native | -38% | **-0.4%** | ✅ 消除 |
| P50 首 token 延迟 | 142ms | 89ms | -37% |
| P99 首 token 延迟 | 891ms | 312ms | -65% |

overhead 从 38% 降到了 0.4%——已经在测量误差范围内。

锁带来了排队延迟，但这是合理的：Metal GPU 本来就不能高效处理 16 个并发推理流。串行化之后，每个请求的 GPU 利用率更高，总吞吐反而更好。

---

## 第六步：为什么不能进一步优化？

既然已经接近 mlx-lm native 的速度，能不能更快？我们检查了 mlx-lm 的实现，发现它已经做了两个关键优化：

**`mx.async_eval`：异步求值**

MLX 是懒求值的（lazy evaluation），tensor 操作不会立即执行，而是构建计算图。`mx.async_eval` 让求值在后台进行，主线程可以继续准备下一个 token 的输入：

```python
# mlx-lm 内部
for token in generate_step(prompt, model, ...):
    mx.async_eval(token)  # 后台异步求值
    yield token
```

**`@mx.compile`：JIT 编译**

对推理的热路径（forward pass）用 `@mx.compile` 做 JIT 编译，将 Python 函数编译成优化的 Metal 计算图，消除每步的 Python 解释开销：

```python
@mx.compile
def step(x, cache):
    return model(x, cache=cache)
```

这两个优化 mlx-lm 已经实现了。我们直接调用 `stream_generate()`，这些优化已经包含在内。我们的层面上没有更多可以做的——瓶颈已经在 GPU kernel 层。

---

## 第七步：KV 量化的陷阱

有一个常见的优化思路：对 KV cache 做量化（INT8 或 INT4），降低内存占用，提高推理速度。我们测试了。结果出乎意料：

| KV 量化方案 | 吞吐 | vs FP16 基线 |
|-------------|------|-------------|
| FP16（无量化） | 354.9 tok/s | 基线 |
| INT8 量化 | 239.1 tok/s | **-32%** |
| INT4 量化 | 241.7 tok/s | **-32%** |

量化后反而更慢了，而且幅度相当明显。

原因在于 **KV cache 的访问模式**。在短序列（<512 token）下，KV cache 的内存占用本来就不大，带宽瓶颈并不显著。但每个 decode step 都需要对 KV cache 做 quantize/dequantize，这个操作在 M2 Max 上的开销大约是 0.8ms/step。而节省的带宽读取时间不到 0.3ms/step。

**开销 > 收益**，所以量化反而变慢了。

这个规律在长序列（>2048 token）时会反转，那时带宽收益大于量化开销。但对于典型的 agent 任务（几百 token 的对话），INT8/INT4 KV 量化是负优化。

---

## 与竞品的对比

在相同硬件（M2 Max）、相同模型（Qwen2.5-7B-Instruct 4-bit）下：

| 工具 | 吞吐（tok/s） | 并发支持 | OpenAI 兼容 |
|------|-------------|----------|------------|
| **agent-infer** | **354.9** | ✅ 多请求队列 | ✅ |
| mlx-lm native | 352 | ❌ 单请求 | 部分 |
| llama.cpp (Metal) | 24–60 | ❌ 单请求 | ✅ |
| ollama | ~35 | ⚠️ 有限 | ✅ |
| LM Studio | ~40 | ❌ 单请求 | ✅ |

agent-infer 比 llama.cpp 快 **6–15 倍**，比 ollama 快 **10 倍**，同时提供完整的 OpenAI 兼容 API 和多请求队列调度。

llama.cpp 和 ollama 速度较慢的主要原因是：它们的 Metal 后端通常使用 GGUF 量化的通用矩阵乘法，而 MLX 的 Metal 后端针对 Apple Silicon 的 matrix coprocessor（AMX/ANE）做了专门优化。

---

## 经验总结

这次优化的核心教训：

1. **不要猜测瓶颈**。SSE flush 是直觉，但不是根因。`run_in_executor` + 并发竞争才是。
2. **GPU 并发不等于 GPU 效率**。Metal GPU 不像 CUDA 那样有成熟的多流并发支持，并发反而导致命令交错和性能下降。
3. **量化不是万灵药**。在内存不是瓶颈的场景下，量化 overhead 可能大于收益。先测量，再优化。
4. **利用已有的优化**。mlx-lm 的 `async_eval` 和 `@mx.compile` 已经很好了，站在巨人肩上比重新造轮子聪明。

---

## 试试看

agent-infer 的 MLX 后端支持所有 mlx-community 上的模型。在 Apple Silicon Mac 上运行只需几步：

```bash
# 安装（含 MLX 依赖）
pip install "agent-infer[mlx]"

# 启动推理服务器（OpenAI 兼容）
python -m agent_infer serve \
  --mlx-model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --port 8000

# 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "你好"}]}'
```

或者直接运行 agent REPL：

```bash
python -m agent_infer repl \
  --mlx-model mlx-community/Qwen2.5-7B-Instruct-4bit
```

---

## 贡献与参与

如果你也在做 Apple Silicon 上的 LLM 推理优化，欢迎参与 agent-infer 的开发：

- **⭐ Star** 项目，帮助更多人发现它
- **🐛 报 Bug**：在 Issues 中描述你遇到的问题
- **💡 提 Feature**：Llama 3 / Gemma / Phi 模型支持？FlashAttention-3？欢迎讨论
- **🔧 提 PR**：查看 [ROADMAP.md](../ROADMAP.md) 了解当前优先级

MLX 后端的代码在 [`agent_infer/mlx_backend.py`](../agent_infer/mlx_backend.py) 和 [`agent_infer/mlx_server.py`](../agent_infer/mlx_server.py)，总共不到 650 行，欢迎阅读和参与。

> 这是一个开放的项目。我们相信高性能推理不应该只有大公司才能做到。
