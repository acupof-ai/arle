# 2026-04-15 · Metal runtime memory-limit controls

## Context

Metal `M0.4` 之前已经能观测到 live queue / latency / MLX allocator memory，
但 operator 还只能“看”，不能直接限制 MLX allocator 行为。

这批把 MLX memory-limit / cache-limit / wired-limit 绑到 Rust bridge，并把
它们暴露到所有用户可见的 Metal 入口：

- `metal_serve`
- `metal_bench`
- `metal_request`

环境：

- Machine: Apple M4 Pro
- OS: macOS 26.3.1(a)
- Build:
  `cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_serve --bin metal_request --bin metal_bench`

## What Worked

### 所有 Metal CLI 现在都有统一的 allocator controls

三条入口都已暴露：

```text
--memory-limit-bytes <BYTES>
--cache-limit-bytes <BYTES>
--wired-limit-bytes <BYTES>
```

这意味着 Metal 的内存控制不再停留在 MLX 默认值或内部 Python API。

### `metal_serve` 会在模型加载前应用这些限制

对 `Qwen3-0.6B-4bit` 的真实启动日志显示，限制会在 `hf_hub` / 权重加载前生效：

```text
Metal runtime memory limit set to 25769803776 bytes (previous 48962627174)
Metal runtime cache limit set to 4294967296 bytes (previous 48962627174)
Metal runtime wired limit set to 0 bytes (previous 0)
```

示例命令：

```bash
./target/release/metal_serve \
  --model-path mlx-community/Qwen3-0.6B-4bit \
  --port 8017 \
  --warmup 0 \
  --memory-limit-bytes 25769803776 \
  --cache-limit-bytes 4294967296 \
  --wired-limit-bytes 0
```

### 新的控制面和现有 observability 一起工作

同一进程里，启动后和真实请求后的 `/v1/stats` / `/metrics` 仍然正确：

```text
requests=1 active=0 waiting=0 tokens_out=8 kv_util=0.0% prefix_hit_rate=0.0% active_mem=679.6MB peak_mem=737.3MB cache_mem=104.9MB ttft_p50=0.1ms ttft_p99=0.1ms tpot_p50=5.0ms
```

对应 Prometheus 指标：

```text
infer_prefix_hit_rate{model="Qwen3-0.6B-4bit",} 0.0000
infer_memory_active_bytes{model="Qwen3-0.6B-4bit",} 231215716
infer_memory_peak_bytes{model="Qwen3-0.6B-4bit",} 330306124
infer_memory_cache_bytes{model="Qwen3-0.6B-4bit",} 99090520
infer_requests_total{model="Qwen3-0.6B-4bit",} 1
infer_tokens_generated_total{model="Qwen3-0.6B-4bit",} 8
```

边界保持明确：

- 这次补的是 allocator control，不是 prefix reuse
- `prefix_hit_rate` 仍然为 `0`，因为 `M0.3` 还没有把 shared-prefix reuse
  接入 live Metal serving

## Rule

Metal `M0.4` 不能只停留在“指标能看见”。

最小合格面应该同时包括：

- runtime-backed memory observability
- user-facing allocator controls
- 一次真实 server/request smoke，证明控制面没有和 HTTP/runtime 指标断开
