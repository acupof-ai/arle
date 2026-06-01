# Metal Low Process RSS Analysis

## TL;DR

The earlier README sweep showed about **2.2 GiB RSS**; the current README
essay-average chart shows about **2.5-4.1 GiB mean process RSS** for ARLE.
Neither number says the
35B-A3B model only needs that much memory, and neither says weights are
unloaded. Both are process-attributed RSS samples after ARLE stopped pinning MLX
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

The README sweeps sampled:

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

The README chart uses process RSS because that was the regression users saw. It
is not a replacement for a full `vmmap` / MLX allocator / memory-pressure trace.

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

### 3. The README essay retest stayed low with prompt length

ARLE default, c=1, output 256:

| input | TTFT | TPOT | process RSS | system used |
|---:|---:|---:|---:|---:|
| 128 | 0.27 s | 12.3 ms | 3.15 GiB | 30.13 GiB |
| 4k | 5.14 s | 14.2 ms | 3.97 GiB | 32.52 GiB |
| 8k | 11.53 s | 17.1 ms | 4.11 GiB | 34.25 GiB |
| 12k | 19.50 s | 32.3 ms | 2.46 GiB | 34.01 GiB |

The mean RSS curve stays far below the wired-weight footprint. It is not
monotonic because process RSS is page residency/accounting, and macOS can reclaim
non-wired pages between requests. c=1 prefill does not retain full activations,
and the retained KV for 256 generated tokens is small relative to model weight
files.

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

## Why mlx-lm Shows Higher RSS

mlx-lm and ARLE both sit on Apple unified memory, but they do not expose the same
process RSS shape. In the current README essay retest, mlx-lm's process RSS
stayed around 7.23-7.24 GiB while ARLE default averaged around 2.46-4.11 GiB.
The system-used numbers are closer than the process RSS numbers.

That means the chart is best read as:

> ARLE's default no longer forces model weights into process-attributed wired
> RSS; mlx-lm still has a larger process-attributed resident set in this
> measurement.

It should not be read as:

> ARLE has proven the whole model consumes only 2.5-4.1 GiB total memory.

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

> In the README essay-average retest, ARLE's default process RSS during
> streaming averages about 2.5-4.1 GiB because model weights are loaded through
> MLX mmap-backed unified-memory tensors and are no longer pinned/wired into
> process-attributed RSS by default. The model is still loaded; the number is a
> residency/accounting measurement, not total model memory.
