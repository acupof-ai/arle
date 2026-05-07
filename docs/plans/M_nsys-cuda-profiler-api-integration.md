# M_nsys — cudaProfilerStart/Stop API integration (proper nsys fix)

> Per user 2026-05-07 directive "解决问题而不是绕过 nsys 必须得能用才行"
> — fix nsys reliably across all workload shapes, not just
> high-density high-conc.

## Priority & ROI

**Priority**: **P0** for diagnostic infrastructure(blocks future
nsys-driven optimizations on long-context / low-conc workloads).

**ROI basis**:
- Today: nsys reliably captures CUDA kernel data only at high-conc
  workloads (1k/256/c=64). Long-ctx + low-conc fails (0 kernel
  data in `.nsys-rep`) due to nsys 2025.6 + CUDA Graph + capture
  window timing edge cases (web search confirms upstream nsys
  limitation, no canonical fix in 2025.x).
- Fix: ARLE-side `cudaProfilerStart()` / `cudaProfilerStop()`
  calls triggered by SIGUSR1/SIGUSR2. Then `nsys profile
  --capture-range=cudaProfilerApi --capture-range-end=stop`
  captures EXACTLY the window between start/stop signals.
- **Workload-independent**: works at long-ctx 4k/c=4 the same as
  high-conc 1k/256/c=64. No more guessing `--delay`/`--duration`
  vs bench warmup timing.

**Negative case**: pure infrastructure/dev-tooling change. Zero
runtime cost (signal handler dormant when not nsys-tracing).

**Kill criteria**:
- If after wiring `cudaProfilerStart` calls, nsys still misses
  kernel data at long-ctx workloads → escalate to NVIDIA bug
  report; the ARLE-side wiring is correct
- LOC > 100 → reconsider scope (might mean integration is more
  invasive than expected — likely it's < 50 LOC)

## Background — why current nsys fails

Tested on 2026-05-07:
- ✓ Phase 1 trace(`fdb531b`,1k/256/c=64,45s):**19MB,full
  kernel data**
- ✓ Phase 1A v3 high-conc trace (`bench-output/2026-05-07-nsys-highconc-1av3/`,
  same workload): **93MB, full kernel data**
- ✗ Phase 1A v3 longctx 4k/c=4: **5.8MB, 0 kernel data**
- ✗ M_b.1 Phase B longctx attempts (multiple): empty
- ✗ Phase 0 longctx prefill trace attempts: empty

Pattern: **kernel-density-dependent**. nsys 2025.6 silently drops
CUDA tracing if kernel events are below some threshold during the
configured capture window (likely a CUPTI ring buffer / event
flush timing issue per SGLang issue #2776).

Web research:
- [SGLang #2776](https://github.com/sgl-project/sglang/issues/2776):
  same symptom on H100 + DeepSeek-V3, no resolution since Jan 2025
- [NVIDIA forum 315536](https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536):
  fix is `CuptiUseRawGpuTimestamps=false` for WSL2, NOT applicable
  to native Linux

The canonical workaround (vLLM official profiling guide, NVIDIA
sample code): **application calls `cudaProfilerStart()` /
`cudaProfilerStop()` to delimit capture window**. nsys with
`--capture-range=cudaProfilerApi` then ignores the global
delay/duration and captures exactly between the calls.

This bypasses the kernel-density issue because nsys starts
capture only when application explicitly says "now is the time".

## Design

### P0 — cudaProfiler signal handler (~30 LOC)

In `infer/src/main.rs`:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use signal_hook::{consts::{SIGUSR1, SIGUSR2}, iterator::Signals};

// Boot a background thread that listens for SIGUSR1/SIGUSR2
fn install_cuda_profiler_handler() -> Result<()> {
    let mut signals = Signals::new(&[SIGUSR1, SIGUSR2])?;
    std::thread::spawn(move || {
        for sig in signals.forever() {
            unsafe {
                match sig {
                    SIGUSR1 => {
                        cudarc::driver::sys::cuProfilerStart();
                        info!("cuProfilerStart fired");
                    }
                    SIGUSR2 => {
                        cudarc::driver::sys::cuProfilerStop();
                        info!("cuProfilerStop fired");
                    }
                    _ => {}
                }
            }
        }
    });
    Ok(())
}

// In main(), after CUDA context init:
install_cuda_profiler_handler()?;
```

### P1 — bench wrapper(~20 LOC)

`scripts/profile_nsys_workload.sh`:
```bash
#!/usr/bin/env bash
# Spawn ARLE under nsys, wait for ready, fire SIGUSR1, run bench,
# fire SIGUSR2, kill ARLE, return nsys-rep.
SERVER_CMD="$1"
BENCH_CMD="$2"
OUTPUT="$3"

nsys profile --output "$OUTPUT" --force-overwrite=true \
  --trace cuda,nvtx,osrt --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  $SERVER_CMD &
SERVER_PID=$!

# Wait for server ready
until curl -sm 1 http://localhost:8000/v1/stats >/dev/null; do sleep 1; done

# Find ARLE child PID
ARLE_PID=$(pgrep -P $SERVER_PID -f infer)

# Start capture
kill -USR1 $ARLE_PID

# Run bench
$BENCH_CMD

# Stop capture
kill -USR2 $ARLE_PID

# Cleanup
sleep 5
kill $SERVER_PID
wait $SERVER_PID
```

### P2 — verify (~0 LOC)

Run at the previously-failing longctx 4k/c=4 workload. Expect:
- `.nsys-rep` contains CUDA kernel data
- Kernel summary shows decode_attention + GEMM kernels (not just
  warmup outliers)

## Tasks

| # | Task | File | LOC | Owner |
|---|---|---|---|---|
| P0 | SIGUSR1/SIGUSR2 handler in main.rs | `infer/src/main.rs` + Cargo.toml signal-hook dep | ~30 | Codex (or Claude) |
| P1 | bench wrapper script | `scripts/profile_nsys_workload.sh` (new) | ~20 | Claude |
| P2 | Validate at long-ctx workload | bench artifact | 0 | Claude |

## Acceptance

- `cargo check --release -p infer --features cuda` passes
- `cargo test --release -p infer --features cuda --test e2e` passes
  (signal handler must not interfere with normal operation)
- nsys at long-ctx 4k/c=4 produces `.nsys-rep` with non-empty
  kernel data
- ARLE startup unaffected; if SIGUSR1 never fires, ARLE behaves
  identically to today

## Cross-references

- Discovery: 2026-05-07 multiple failed nsys traces at long-ctx
- Successful trace evidence (high-conc works):
  `bench-output/2026-05-07-nsys-highconc-1av3/arle-1av3.nsys-rep`
- vLLM canonical pattern (their docs):
  https://docs.vllm.ai/en/stable/contributing/profiling/
- NVIDIA cudaProfilerApi reference:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#profiler-control

## Rule

- **"Workaround vs proper fix"**: empirical workarounds (use
  high-conc workload that nsys can capture) are NOT acceptable
  long-term. Proper fix is application-side cudaProfilerApi
  integration so the capture window is application-controlled,
  not nsys-timing-controlled. Per user directive: 解决问题而不是绕过.
- This finding (nsys reliability gap) blocks the M3.9 Phase 1A v3
  diagnosis at long-ctx and any future low-conc trace work. P0
  for diagnostic infra.
