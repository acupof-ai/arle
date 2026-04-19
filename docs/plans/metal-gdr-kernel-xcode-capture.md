# Metal GDR Kernel — Xcode GPU Capture Runbook

**Date**: 2026-04-19
**Goal**: capture one decode step of `gated_delta_step` on Apple M4 Max,
attribute the 6.1 ms/row cost to compute / memory / dispatch so we can pick
the right optimization (kernel rewrite, memory layout, dispatch batching,
or none).
**Why now**: per `docs/experience/wins/2026-04-19-metal-qwen35-final-state.md`,
this is the only remaining lever with sized non-trivial upside (up to +50%
c=8 if 6.1 → 3 ms/row). All other /loop-reachable levers are exhausted.
**Owner**: ckl (interactive — Xcode GUI required)

---

## Prereqs (one-time)

- macOS 26.x with Xcode 16+ installed (`xcode-select -p` not empty)
- `xcrun --find metal-frameworkdebugger` resolves
- Local checkout of `mlx-community/Qwen3.5-4B-MLX-4bit` (≈4 GB)
- `metal_bench` built release: `cargo build --release --no-default-features --features metal,no-cuda --bin metal_bench`

---

## Step 1 — build a capture-friendly bench binary

`metal_bench` already supports `--use-step-driver` which routes through the
per-step FFI path (matches HTTP server). For capture we want a TIGHT loop
with no Tokio / scheduler overhead. Use:

```bash
MTL_CAPTURE_ENABLED=1 \
  ./target/release/metal_bench \
    --model /path/to/Qwen3.5-4B-MLX-4bit \
    --prompt-tokens 32 \
    --generation-tokens 16 \
    --warmup 3 \
    --runs 1 \
    --use-step-driver
```

`MTL_CAPTURE_ENABLED=1` is the env var Apple's Metal toolchain checks before
allowing programmatic / Xcode-attached GPU capture. Without it, capture
silently no-ops.

---

## Step 2 — attach Xcode GPU Frame Capture

Two ways. Pick one.

### 2a — Xcode "Attach to Process by PID" (recommended for CLI bins)

1. Open Xcode → **Debug → Attach to Process by PID or Name…**
2. Enter `metal_bench` → Attach. Xcode shows the running process.
3. In the toolbar, the GPU capture icon (camera-with-triangle) becomes
   active. Click it to capture the **next frame's** GPU work.
4. Re-run `metal_bench` (with the same env + args) — capture triggers on
   the first MTLCommandBuffer commit.
5. Xcode opens the .gputrace browser with all dispatched kernels, encoders,
   and command buffers from one decode step.

### 2b — Programmatic capture via `MTLCaptureManager` (if 2a flakes)

Modify `mlx_qwen35_model.cpp::qwen35_compiled_step_session` to wrap one
specific step in `MTLCaptureManager.startCapture/stopCapture`. **DON'T
COMMIT this** — it's a debugging-only diff. Save the .gputrace to
`/tmp/qwen35_step_<timestamp>.gputrace`. Open in Xcode.

Skeleton (Objective-C++ snippet, for reference — not committed):

```objc
#import <Metal/Metal.h>

void capture_one_step() {
    MTLCaptureManager* cm = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = MTLCreateSystemDefaultDevice();
    desc.destination = MTLCaptureDestinationGPUTraceDocument;
    desc.outputURL = [NSURL fileURLWithPath:@"/tmp/qwen35_step.gputrace"];

    NSError* err = nil;
    if (![cm startCaptureWithDescriptor:desc error:&err]) {
        NSLog(@"capture start failed: %@", err);
        return;
    }
    // … run one decode step here …
    [cm stopCapture];
}
```

Trigger it on the Nth decode step (skip warmup) via an env-gated counter:

```cpp
static int step_n = 0;
if (getenv("INFER_CAPTURE_STEP") &&
    step_n++ == atoi(getenv("INFER_CAPTURE_STEP"))) {
    capture_one_step();
}
```

---

## Step 3 — what to look at in the .gputrace

Open the capture in Xcode. Use the **Performance** tab (left sidebar).

For each `gated_delta_step` kernel dispatch (24 per layer, 24 GDR layers
per step = up to 24 dispatches per step, one per layer):

| Metric | Target | Notes |
|---|---|---|
| **GPU Time** | 6.1 ms/row baseline | If summed across all 24 GDR layers ≈ total GDR time |
| **Limiter** (Compute / Memory / Dispatch) | which? | Drives optimization choice |
| **ALU Utilization** | should be high if compute-bound | Low = memory-bound |
| **L2 Cache Miss Rate** | high = memory-bound | Suggests layout/coalescing issue |
| **Threadgroup Memory Used** | vs M4 limit (32 KB) | Spilling? |
| **Occupancy** | threads per SIMD group | Low = launch overhead, dispatch-bound |
| **Bandwidth** | GB/s read / write | Compare to M4 ~400 GB/s UMA |

In the **Counters** tab also check:
- `total_alu_inst` — raw compute volume
- `device_load_byte` / `device_store_byte` — memory volume
- `fragment_function_active` (should be 0 — this is a compute kernel)

---

## Step 4 — diagnosis decision tree

Read **Limiter** and **L2 Miss Rate** first.

| Reading | Diagnosis | Next action |
|---|---|---|
| Limiter = Compute, ALU > 70% | Kernel is at compute peak | Algorithm change or accept ceiling |
| Limiter = Memory, L2 miss > 30% | Memory layout / coalescing | Try transposing K/V state, padding, or restructuring `gdr_state_in` shape |
| Limiter = Dispatch, kernel runtime < 100µs | Per-launch overhead dominates | Batch the 24 layers' dispatches into one (this is the "GDR kernel batch dispatch" lever cited in the wins doc) |
| ALU < 30% AND Memory < 30% | Idle on synchronization | Look for `MTLCommandBuffer` waits, eval barriers |

The wins-doc-sized "+50% c=8" bet is on the **Dispatch** branch. Confirm
or refute first — that's the highest-leverage data point.

---

## Step 5 — close the loop

After capture, write up findings as
`docs/experience/wins/YYYY-MM-DD-metal-gdr-kernel-capture-findings.md`
with the table from Step 3 filled in plus the diagnosis from Step 4. If
the Dispatch hypothesis lands, open a follow-up plan for cross-layer
dispatch batching. If Memory is the limiter, open a plan for state-layout
restructuring. If Compute is the limiter, the wins doc's 6.1 → 3 ms/row
bet is dead and the project's terminal state truly holds.

---

## Out of scope for this runbook

- Full-attn layer capture (24 GDR is the bottleneck; 8 full-attn is
  linear-in-position and a smaller share of total time per Iter 11)
- mx::compile around `gated_delta_step` — won't reduce the kernel itself,
  only inter-op overhead which is already small
- KV quantization for full-attn layers — separately ruled out for Metal
  in `docs/research/kv-quantization-metal.md` (no FP8 HW on M-series)

---

## References

- Apple, [Metal GPU Capture](https://developer.apple.com/documentation/metal/frame_capture_debugging_tools)
- `docs/experience/wins/2026-04-19-metal-qwen35-final-state.md` — terminal state of the optimization arc
- `docs/experience/wins/2026-04-18-metal-qwen35-concurrent-decode-ceiling.md` — origin of the 6.1 ms/row figure
- `crates/mlx-sys/src/mlx_qwen35_model.cpp:87` — `gated_delta_kernel()` definition
- `crates/mlx-sys/src/mlx_qwen35_model.cpp:684-820` — full `gdr_step()` body
