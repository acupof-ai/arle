# Metal 2.2 GiB RSS Analysis

## TL;DR

The README chart's **2.2 GiB RSS** is not saying the 35B-A3B model only needs
2.2 GiB of memory, and it is not saying weights are unloaded. It is the
process-attributed RSS sampled during streaming after ARLE stopped pinning MLX
Metal pages by default.

The model is still loaded and addressable:

- ARLE loads the Qwen3.6 safetensors through MLX mmap-backed arrays.
- `metal_serve` default no longer calls `mlx::set_wired_limit`.
- macOS unified-memory pages that are mmap-backed / Metal-managed / not wired
  do not have to remain resident or charged to the process RSS at all times.
- c=1 decode only retains a small KV working set, and Qwen3.6 MoE touches a
  sparse subset of expert weights per token.

So the low RSS number is mostly a residency/accounting result, not a model-size
miracle.

## What RSS Measured

The README sweep sampled:

```python
psutil.Process(pid).memory_info().rss
```

plus child process RSS, every 0.5 s during each streaming request. The same raw
record also keeps `system_used_gb`, because on Apple unified memory RSS is not a
complete memory-pressure model.

Important distinction:

| Term | Meaning in this bench |
|---|---|
| Model bytes | Weight files on disk / MLX tensors addressable by the process. |
| Resident / wired pages | Physical pages macOS currently keeps resident and cannot easily evict. |
| Process RSS | Pages currently attributed to the Unix process by the kernel. |
| System used | Host-wide memory pressure, including pages not cleanly attributable to this process RSS sample. |

The README chart uses process RSS because that was the regression users saw: the
ARLE process looked 8-10 GiB larger than mlx-lm. It is not a replacement for a
full `vmmap` / MLX allocator / memory-pressure trace.

## Evidence

### 1. The loader really loaded the model

Default no-wired run log:

```text
loading 4 shard(s) via MLX mmap ...
loaded 2090 tensors (memory-mapped)
dequantizing embed_tokens at load time
arch: Qwen3.6/Qwen3.5-MoE 40 layers ... moe=40
weights loaded into Metal unified memory
Warmup 1/1 finished ... prompt_tokens=8, completion_tokens=1
```

The corresponding code path is:

- `infer/src/backend/metal/loader.rs`: `load_tensor_map()` collects the
  safetensors shards and calls `super::mlx::load_safetensors(path_str)`.
- `crates/mlx-sys/src/mlx_bridge.cpp`: `mlx_load_safetensors()` forwards to
  MLX `load_safetensors(std::string(path))`.
- `infer/src/backend/metal.rs`: `MetalRuntimeLimits::apply()` only calls
  `set_wired_limit_bytes` when `wired_limit_bytes` is explicitly set.

### 2. The previous high RSS was caused by wired residency

The controlled A/B after the fix:

| case | ready RSS |
|---|---:|
| default, no auto wired limit | 5.71 GiB |
| explicit `--auto-wired-limit` | 18.61 GiB |

The opt-in run logged:

```text
auto wired_limit = 20 GiB (21475946095 bytes; model dir ...)
Metal runtime wired limit set to 21475946095 bytes (previous 0)
```

The default run has the same model load and warmup, but no wired-limit log.
That isolates the old ~18-20 GiB process RSS to the residency policy, not to a
second copy of the weights.

### 3. The README sweep stayed flat with prompt length

ARLE default, c=1, output 256:

| input | TTFT | TPOT | process RSS | system used |
|---:|---:|---:|---:|---:|
| 128 | 0.21 s | 11.6 ms | 2.21 GiB | 32.11 GiB |
| 4k | 4.59 s | 12.4 ms | 2.23 GiB | 33.70 GiB |
| 8k | 9.58 s | 13.1 ms | 2.23 GiB | 33.58 GiB |
| 12k | 16.64 s | 14.3 ms | 2.23 GiB | 32.96 GiB |

The flat RSS curve is expected once weights are not wired: c=1 prefill does not
retain full activations, and the retained KV for 256 generated tokens is small
relative to model weight files. Prompt length moves TTFT and some scratch /
system-used pressure, but it does not force all mmap-backed model pages to be
charged to process RSS.

## Why Inference Still Works

Loaded does not mean fully resident.

The MLX arrays remain valid handles to model weights. When a kernel needs a
page, macOS/Metal can fault or migrate the page into resident unified memory.
If the page is not wired, it can later be reclaimed or attributed differently.

Qwen3.6 also helps this RSS shape:

- It is a sparse MoE. Each token routes to a subset of experts, so one request
  does not necessarily touch every expert weight with equal frequency.
- The startup warmup is tiny (`prompt_tokens=8`, `completion_tokens=1`), so it
  does not intentionally sweep the full model into resident memory.
- The benchmark is c=1 with 256 output tokens, so the live KV and scheduler
  working set is small.

This is why the process can serve a 35B-A3B model while the process RSS sample
is much smaller than the on-disk model size.

## Why mlx-lm Shows About 10.8 GiB RSS

mlx-lm and ARLE both sit on Apple unified memory, but they do not expose the same
process RSS shape. In the README run, mlx-lm's process RSS stayed around
10.77-10.79 GiB while ARLE default stayed around 2.21-2.23 GiB. The system-used
numbers are much closer than the process RSS numbers.

That means the chart is best read as:

> ARLE's default no longer forces model weights into process-attributed wired
> RSS; mlx-lm still has a larger process-attributed resident set in this
> measurement.

It should not be read as:

> ARLE has proven the whole model consumes only 2.2 GiB total memory.

## Tradeoff

The tradeoff is p99 stability under memory pressure.

With no wired residency, cold or evicted expert pages can come back as latency
tail. That is why `--auto-wired-limit` still exists: it sets MLX's wired limit
to approximately model weight bytes plus 1 GiB, intentionally spending memory to
make pageout less likely.

Default posture after this fix:

| Mode | Memory posture | When to use |
|---|---|---|
| default | Low process RSS, OS can reclaim non-wired pages | local serving, mixed desktop workloads, README comparison |
| `--auto-wired-limit` | Higher RSS, model pages protected from pageout | latency-first / p99-sensitive dedicated serving |

## What Would Make This More SOLID

This doc explains the current evidence and kernel/accounting model. A full
memory-proof would add:

- `vmmap -summary <pid>` snapshots for default vs `--auto-wired-limit`.
- MLX allocator/cache/wired counters before load, after warmup, and during 12k
  generation.
- A memory-pressure A/B that reports p50/p95/p99 TTFT/TPOT with and without
  `--auto-wired-limit`.
- Higher-frequency RSS/system sampling, because the current sweep samples every
  0.5 s and can miss short transients.

Until then, the precise claim is:

> In the README sweep, ARLE's default process RSS during streaming is about
> 2.2 GiB because model weights are loaded through MLX mmap-backed unified-memory
> tensors and are no longer pinned/wired into process-attributed RSS by default.
> The model is still loaded; the number is a residency/accounting measurement,
> not total model memory.
