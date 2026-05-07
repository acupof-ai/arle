# M_e.5 — MLX multi-stream encode pipelining for c≥1 Metal decode

**Owner:** ckl · **Status:** designed 2026-05-07 (subagent), awaiting impl tick
**Track:** Metal scheduler · **Predecessor:** M_e.1 oMLX-C v3 (default-on),
  M_e.4 SwiGLU compile-fusion

## Goal

Encode step N+1's command buffer on a second `mlx::Stream` while step
N's GPU exec completes on the first. Target: convert each step's
wall-time from `encode_cpu_ms + gpu_exec_ms` (sequential) to
`max(encode_cpu_ms, gpu_exec_ms)` — predicted ~12 ms win per step on
Qwen3.6 35B-A3B if encode and GPU exec are balanced.

## 1. MLX stream API surface (verified)

`mlx/stream.h` (https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/stream.h):

```cpp
struct MLX_API Stream {
  int index;
  Device device;
};
MLX_API Stream  default_stream(Device d);          // per-thread
MLX_API void    set_default_stream(Stream s);      // per-thread
MLX_API Stream  new_stream(Device d);
MLX_API void    synchronize();                     // default stream
MLX_API void    synchronize(Stream);               // specific stream
MLX_API void    clear_streams();
```

`StreamContext` (in `mlx/utils.h`) is the RAII helper. Critical:
**`default_stream`/`set_default_stream` are per-thread**, so the same
thread can flip the default mid-function and subsequent ops bind to
the new stream — exactly what `ml-explore/mlx#3078`'s
"no-cross-thread-encoder-share" rule leaves open.

Each `Stream` has its own `StreamThread` worker (`mlx/scheduler.h`),
so dispatching to B does not block on A's encode.

## 2. Cross-stream dependency tracking — automatic

From `mlx/transforms.cpp::eval_impl`:

```cpp
} else if (in.event().stream() != stream) {
  // Use event to wait across async eval
  in.event().wait(stream);
}
```

Each `array` carries an `Event` keyed by the stream that produced it
(`mlx/event.h`: `void wait(Stream); void signal(Stream);`). When op
on stream B consumes an array produced on stream A, MLX inserts an
event-wait on B that blocks B's GPU queue until A signals. **No memory
copy** — unified memory; only a queue-side fence.

**Hidden cost called out**: that fence is real. If step N+1's compute
graph touches any array produced by step N (KV cache writes, running
token-id buffer, RoPE state), B will queue-wait on A and serialize at
the GPU. **The win materializes purely on the host: B's encode runs
while A's GPU is still draining.** The diagnosed bottleneck (95%
encode, 5% GPU) means even with full GPU serialization, encode-CPU
overlap is the prize.

## 3. Implementation recipe (~30 LoC, env-gated)

Inside `crates/mlx-sys/src/mlx_qwen35_model.cpp::qwen35_compiled_step_batch_packed`:

```cpp
// File-scope (per backend instance):
static thread_local mx::Stream s_a = mx::new_stream(mx::Device::gpu);
static thread_local mx::Stream s_b = mx::new_stream(mx::Device::gpu);
static thread_local int parity = 0;
static thread_local std::optional<mx::array> prev_sampled;
static thread_local mx::Stream prev_stream = s_a;

mx::array qwen35_compiled_step_batch_packed(/* …inputs… */) {
  if (!dual_stream_enabled()) return /* legacy path */;

  mx::Stream cur = (parity ^= 1) ? s_b : s_a;
  mx::StreamContext ctx(cur);                     // RAII: per-thread default

  // Encode step N+1 on `cur`. KV writes from prev step on `prev_stream`
  // are tracked via array events; MLX inserts a GPU-side fence
  // automatically.
  auto logits  = forward_decode(inputs, kv);
  auto sampled = sample(logits);
  mx::async_eval({sampled});                       // returns after encode

  if (prev_sampled) {
    mx::synchronize(prev_stream);                  // join only the prior stream
    auto out = std::move(*prev_sampled);           // host-readable now
    prev_sampled = sampled;
    prev_stream  = cur;
    return out;                                    // caller gets step N's token
  }
  prev_sampled = sampled;
  prev_stream  = cur;
  return sampled;                                  // first call still blocks once
}
```

Env gate: `INFER_METAL_DUAL_STREAM=1`. Off by default until matched
A/B confirms the win.

## 4. Composes with existing wins

- **oMLX-C v3** (host-pipelining via `prev_sampled`): the dual-stream
  recipe REPLACES the existing `decode_qwen35_packed_batch_pipelined`
  prev_sampled trick — both can't be active simultaneously since they
  fight over the same field. Choose one:
  - oMLX-C v3 alone: host overlap of step N+1 graph build with step N
    eval (already shipped, default ON, ~15% Qwen3.5 win).
  - M_e.5 alone: per-stream encoder overlap (additionally exploits
    StreamThread workers).
  - Combined: probably want oMLX-C v3's `prev_sampled` to live ON
    `prev_stream`, with `synchronize(prev_stream)` replacing the
    `eval(prev_sampled)` in the pipelined helper. Adds complexity.

  **v1 design choice**: keep oMLX-C v3 as is; M_e.5 is an alternative
  encode-pipelining path under a separate flag. Bench head-to-head.

- **Auto-wired-limit** (180e48b): orthogonal — memory pinning is
  unaffected by which stream the encoder uses.

- **M_e.4 SwiGLU compile-fusion**: orthogonal — primitive-count
  reduction shrinks the encode work per step, M_e.5 hides what
  remains.

## 5. Acceptance bench

`scripts/bench_*.sh` Qwen3.6 35B-A3B-4bit sweep, c=1 / c=4 / c=8.
Compare ITL p50/p99 (matched A/B per
`feedback_matched_ab_for_small_bench_effects.md`):

- **Win predicate**: ITL p50 drops by ~`min(encode_cpu_ms, gpu_exec_ms)`
  (most workloads will see ~6-12 ms of encode-vs-exec overlap, since
  diagnostics show 25 ms total step ≈ 25 ms encode + small GPU exec).
- **Disprove predicate**: ITL flat or worse. Most likely causes:
  (a) encode CPU thread already saturated by sampler/RoPE work;
  (b) GPU-side fences serialize tighter than expected because per-
  step graph touches more cross-step state than just KV.

Path probe (per `feedback_path_probe_before_perf_claim.md`): drop
`std::sync::Once` log at function entry confirming alternation
actually happens (`m_e_5_dual_stream_FIRED parity=0` then `parity=1`).

## 6. Risk register

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | Cross-stream fence forces GPU serialization on KV-cache reads → no GPU exec overlap, no win | Bench acceptance bench before defaulting on. Encode-CPU overlap is still the headline; even fully-serialized GPU is OK if encode is the bottleneck (it is). |
| R2 | `thread_local` static state surprises — multiple worker threads each get their own stream pair, doubling GPU memory for the per-stream `StreamThread` worker | Use a single global pair guarded by a mutex on first init, OR accept that ARLE's metal hot path is single-threaded (verified by current `prev_sampled` thread-local pattern). |
| R3 | `mlx::synchronize(prev_stream)` waits for the WHOLE prev stream, not just the prev sample | If prev stream has accumulated multiple unfinished outputs (KV writes), they all sync together. Acceptable: we want them all evaluated by the time we extract host tokens anyway. |
| R4 | First call still blocks (no prev to overlap with) | Documented; bootstrap cost is one step's worth, amortized over the full generation. Same shape as oMLX-C v3 bootstrap. |
| R5 | Combining with oMLX-C v3's `prev_sampled` field creates ambiguity over which stream owns it | v1: M_e.5 replaces oMLX-C v3 (mutually exclusive flags). v2: integrate (track stream alongside MlxArray). |

## 7. Implementation steps

1. Add file-scope `thread_local mx::Stream s_a`, `s_b`, `parity`,
   `prev_stream`. Use `static thread_local` so multiple TUs sharing
   this header would each get their own — but we only call from
   one TU, so single-threaded contention is the design.
2. Add `INFER_METAL_DUAL_STREAM` env probe (cached static int).
3. Path probe: `M_E5_DUAL_STREAM_PROBE` — `std::sync::Once`
   log::info! at first dual-stream entry.
4. Wrap the existing async_eval call in `qwen35_compiled_step_batch_packed`
   in the alternating-stream pattern from §3.
5. Bench c=1/c=4/c=8/c=16, compare against current head (M_e.4 baseline).
6. If win confirmed: matched A/B in second session, then flip default
   ON.
7. Wins entry under `docs/experience/wins/`.

## References

- `mlx/stream.h`, `mlx/utils.h::StreamContext`, `mlx/transforms.h`,
  `mlx/transforms.cpp::eval_impl` (cross-stream event-wait branch),
  `mlx/scheduler.h::StreamThread`, `mlx/event.h`
- ml-explore/mlx#3078 — confirms encoder is per-thread, ruling out
  worker-thread encoding without API change
- ml-explore/mlx#3485 — open PR adding `output_shapes` to GatherQMM
  (out of scope for this plan; would unblock alternative approach of
  full-MoE-forward `mx::compile`)
- Diagnosis source:
  [`docs/experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md`](../experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md)
- Predecessor host-pipelining design:
  [`docs/plans/M_e1-omlx-c-multi-step-pipelining.md`](M_e1-omlx-c-multi-step-pipelining.md)
