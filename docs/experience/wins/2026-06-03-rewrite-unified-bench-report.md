# Rewrite — unified bench + correctness report (Metal shipped, CUDA local-verified)

Goal deliverable for `完成全部的重构和正确性的验证并出新的bench报告` on branch
`arch/ideal-inference-engine`. Consolidates **both backends** of the clean rewrite
(device-neutral `infer-core` → host-only seam → thin executors). Supersedes the
Metal-only [`2026-06-03-rewrite-bench-report-metal-aipc.md`](2026-06-03-rewrite-bench-report-metal-aipc.md)
as the cross-backend view; that entry keeps the per-step Metal detail.

## Verdict (sharp)

- **Architecture thesis: proven on both backends' forward paths.** One device-neutral
  scheduler drives a real MLX forward (Metal) and a real cuda-kernels forward (CUDA),
  each a thin seam impl with **zero legacy `infer` dependency**.
- **Metal (AI-PC primary): shipped** — correctness-verified across 4 configs incl. the
  canonical Qwen3.6 MoE, benched faster than legacy, OS-citizen memory measured.
- **CUDA (server): refactored + local-verified; GPU numerical parity pending-remote**
  on pod infra (not a code gap — see Blocker).
- **Cutover (R5): in flight** — serving facade is the one blocker to deleting legacy
  `infer/`; consumer coupling measured shallow.

## Correctness

### Metal — all bit-identical vs legacy MetalBackend (independently re-verified at HEAD)

| config | result |
|---|---|
| Qwen3.5-0.8B single greedy token | `legacy=11751 new=11751` |
| Qwen3.5-0.8B full 16-tok sequence | identical |
| Qwen3.5-0.8B chunked prefill | identical |
| **Qwen3.6-35B-A3B-4bit MoE** (canonical) | identical |

Core suites green at HEAD: `infer-core 18/18`, `agent-bench 5/5`, infer-plan/seam.

### CUDA — local gate green, GPU parity pending-remote

- `CUDARC_CUDA_VERSION=12060 cargo check -p infer-cuda --features cuda,no-cuda` ✅
- `cargo clippy -p infer-cuda --features cuda,no-cuda -- -D warnings` ✅ (fixed `large_enum_variant` + `>= n+1` post-`71fab628`)
- Numerical greedy parity vs legacy CUDA on real GPU: **PENDING-REMOTE** (Blocker below).

## Performance (Metal, real hardware)

| metric | new engine | legacy | note |
|---|--:|--:|---|
| engine-core scheduler (CPU, mock) | **0.56 µs/tick** @ c=1 | — | scheduler ~free → OS-citizen |
| Qwen3.5-0.8B single-request decode | **188 tok/s** | — | 192 prompt + 128 gen |
| **Qwen3.5-0.8B multi-turn agent workflow** | **195.5 tok/s** | — | 3 turns, 144 tok, 736.6 ms |
| Qwen3.6-35B-A3B-4bit MoE (smoke) | **2.53 tok/s** | 1.97 | **new +28%**, canonical |

Multi-turn north-star (the AI-PC behavior, working):
```
turns=3 total_gen=144 total_wall=736.6ms tok_per_s=195.5  os_impact={samples:3, peak_rss_bytes:465780736}
  turn 0 prompt_len=288 gen=48 ttft_ticks=6 wall=305.0ms
  turn 1 prompt_len=368 gen=48 ttft_ticks=3 wall=215.4ms   <- prefix reuse
  turn 2 prompt_len=448 gen=48 ttft_ticks=3 wall=216.1ms   <- prefix reuse
```
TTFT **6→3 ticks** via cross-turn radix KV reuse. **Peak RSS ≈ 444 MiB** to serve the
whole workflow — measured (`mach_task_basic_info.resident_size_max`), not a stub.

## Code reduction (strict tooling — honest framing, NOT 167k→8.2k)

`167k → 8.2k` is **misleading**: the new stack is incomplete (no HTTP server, single
CUDA BF16 path, partial model coverage). The *fair, same-functionality* comparisons:

| subsystem | legacy | new | note |
|---|--:|--:|---|
| scheduler | `scheduler/cuda` **18.3k** (CUDA-welded) + separate Metal scheduler | `infer-core` **2.1k** | one device-neutral scheduler serves all backends — the cross-backend duplication is gone |
| CUDA Qwen forward | **~25.6k** coupled extraction closure | **1.78k** clean BF16 path | ~14×; deletes multi-quant/god-trait/GGUF/LoRA → one canonical path |
| new-stack clone density (`jscpd`, min-tokens 50) | — | **7.88%** (16 clones / 351 lines) | healthy |

So: yes, code volume and duplication drop **sharply on the parts rewritten** — and the
new stack still has serving surface to grow before it replaces all 167k.

## Blocker — CUDA GPU parity (pending-remote, infra not code)

Pod (H20) state probed 2026-06-03: nvcc 12.9 + cargo **1.92.0** present and working, but
(a) repo not synced, (b) no Qwen model cached, (c) `rust-toolchain.toml` pins **1.95.0**
→ cargo attempts a network download that stalls (Codex hit the same; `RUSTUP_TOOLCHAIN=stable`
then stalled at "Updating crates.io index").

**Unblock recipe (for when pod network/bring-up is available):**
1. Sync repo to pod (`tn push`).
2. `RUSTUP_TOOLCHAIN=1.92.0` to use the installed toolchain (edition-2024 works since 1.85) — avoids the 1.95.0 download.
3. `oniond` (on host via `tn exec`) to fetch a small BF16 Qwen3.
4. Build `infer-cuda --features cuda` (sm_90/sm_90a, CUDA 12.9) in a pod-side tmux (tunnel SIGKILLs detached children ~15-20s).
5. Greedy parity vs legacy CUDA path; report token ids.

## Remaining for full goal

- **CUDA GPU parity** (above) — the open correctness gate.
- **R5 cutover** — facade (OpenAI v1 over new `ServeHandle`, in flight via Codex) → flip 4 shallow consumers → delete `infer/src` → `cargo check --workspace --features metal,no-cuda` proves the new stack stands alone. Final delete gated on CUDA parity.

## Bench-rule note

New-crate measurement; runtime default unchanged (new engine beside old) → bench-exempt
for the default-flip rule (CLAUDE.md). This entry is the consolidated report artifact.
