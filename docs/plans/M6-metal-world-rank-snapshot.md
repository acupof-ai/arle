# M6 — Metal World-Rank Snapshot Runbook

> Sibling of [`m6-cuda-vllm-gap-followups.md`](m6-cuda-vllm-gap-followups.md);
> Apple-Silicon half of the M6 cell in
> [`backend-unification.md`](backend-unification.md) §M6.
> Reference snapshot for CUDA already landed:
> [`docs/experience/wins/2026-05-07-m6-world-rank-snapshot-cuda.md`](../experience/wins/2026-05-07-m6-world-rank-snapshot-cuda.md)
> ("Metal half: pending-remote, needs Apple Silicon runner").
> This plan tells the Apple Silicon runner exactly what to do.

## 0. Goal

Establish the first reproducible Metal-half world-rank snapshot for
ARLE-Metal vs the strongest Apple-Silicon native serving baselines on a
local M-series host. Without this, the M6 cell only carries CUDA evidence
and the unification roadmap cannot claim "world-#1 across both backends".

## 1. Acceptance gate

Win at least **4 of 8** Metal score cells (TTFT p50 + output tok/s ×
four workloads) against the best Apple-Silicon native baseline available
on the test host. Same gate as M6 CUDA. Use the matched-A/B protocol
(`feedback_matched_ab_for_small_bench_effects.md`) for any cell where
the delta is ≤10%.

## 2. Workloads

Same four shapes as the CUDA snapshot — keep them identical so the
cross-backend M6 matrix is a clean A/B by backend:

| ID | Prompt | Decode | Concurrency | Name |
|---|---|---|---|---|
| W1 | 4096 | 256 | 1 | prefill-heavy |
| W2 | 128 | 2048 | 1 | decode-heavy |
| W3 | 1024 | 256 | 16 | high-conc-mid |
| W6 | 32k | 256 | 4 | longctx-32k |

Skip W4/c=64 in the snapshot — use W3/c=16 as the high-conc cell
because the canonical scheduler runtime currently caps Metal hot-path
concurrency at single-digit slots until Tier B#1 paged-KV wires through
([recalibration §B1](../projects/2026-05-07-metal-world-first-recalibration-vs-unification.md)).
Skip W7/W8 (64k/128k) because mlx-lm currently does not ship a
matched-format public baseline at those lengths on M-series. They land
when M_e adds a custom long-context Apple-Silicon baseline.

## 3. Baselines (Apple-Silicon native)

Run whichever the host machine has. Pick the BEST cell per workload
when reporting score deltas.

| Name | Build | Notes |
|---|---|---|
| **mlx-lm** | `pip install mlx-lm==0.31.3` | Primary native baseline. Use `mlx_lm.server` on `--port 8000`. Same model path. |
| **llama.cpp Metal** | `brew install llama.cpp` (or `make GGML_METAL=1`) | Use `llama-server`. Match GGUF Q4_K_M to ARLE's exact-default GGUF path; do NOT use opt-in native-q4 (different format). |
| **vllm-mlx** | `pip install vllm-mlx` (waybarrios fork) | Best high-conc tok/s on M4 Pro per reports (1,150 tok/s c=32 DSV3-Q4). Optional; only if the test host has it installed. |
| **Ollama** | `brew install ollama` | Tertiary; reference for "casual user" baseline only — not a serious M-series competitor on tok/s. |

Drop TGI / TRT-LLM / SGLang from the Metal half — none ships an
Apple-Silicon backend.

## 4. Environment fields to record

Mirror M6 CUDA win's environment block. Required:

- **Host** — `system_profiler SPHardwareDataType | grep "Chip\|Memory"`
  (e.g. `Apple M4 Pro, 36 GB unified`).
- **macOS** — `sw_vers -productVersion`.
- **MLX version** — pin via `crates/mlx-sys` cmake FetchContent commit
  (currently 0.31.1 per `infer/src/backend/metal/AGENTS.md` build req).
- **ARLE commit** — `git rev-parse --short HEAD`.
- **ARLE feature set** —
  `cargo build --release -p infer --no-default-features --features metal`.
  Exact same flags everywhere; no `dflash` opt-in for the snapshot
  unless you are reporting a separate spec-decode cell.
- **Model** — `models/Qwen3.5-0.8B-MLX-4bit` for the W1/W2/W3 cells
  (state-of-the-art ARLE Metal baseline per current
  [`mlx-backend-roadmap.md`](../projects/mlx-backend-roadmap.md));
  `models/Qwen3-4B-MLX-4bit` for the W6 long-context cell so the
  comparison stays apples-to-apples with mlx-lm published numbers.
- **KV dtype** — Metal currently runs unquantized BF16 KV (Tier A#2 not
  landed); record this so future Q8/FP8 KV cells are cross-comparable.

## 5. Commands

ARLE build:

```bash
cargo build --release -p infer --no-default-features --features metal
```

ARLE server (W1 / W2 / W3 — short-context):

```bash
RUST_LOG=info \
target/release/infer \
  --model-path models/Qwen3.5-0.8B-MLX-4bit \
  --port 8000 \
  --max-seq-len 5120
```

ARLE server (W6 — longctx-32k):

```bash
RUST_LOG=info \
target/release/infer \
  --model-path models/Qwen3-4B-MLX-4bit \
  --port 8000 \
  --num-slots 4 \
  --max-seq-len 33792
```

mlx-lm server (W1 / W2 / W3):

```bash
mlx_lm.server --model models/Qwen3.5-0.8B-MLX-4bit --port 8000
```

mlx-lm server (W6):

```bash
mlx_lm.server --model models/Qwen3-4B-MLX-4bit --port 8000
```

llama.cpp server (each workload — same flags, GGUF model path):

```bash
llama-server -m models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --port 8000 -ngl 100 -c 5120
```

GuideLLM invocation: identical to M6 CUDA. Produce one
`docs/experience/wins/2026-MM-DD-bench-guidellm-metal-<workload>-<host>.md`
per cell using
[`TEMPLATE-bench-guidellm.md`](../experience/wins/TEMPLATE-bench-guidellm.md);
canonical wrapper is `scripts/bench_guidellm.sh <label>`.

## 6. Output

A single dated win entry per the M6 CUDA pattern:

```
docs/experience/wins/2026-MM-DD-m6-world-rank-snapshot-metal-<host>.md
```

That entry lists the 8 score cells in the same table shape as the CUDA
snapshot, the per-workload winner, and the cross-backend cell-by-cell
delta vs the CUDA snapshot at commit `48d31ace09f8`.

## 7. Stop conditions / non-goals

- This snapshot does NOT chase Tier A or Tier B Metal optimizations.
  It is a **measurement** of the current state. Tier work follows
  separately under M_d.1 / M5 ripples / Tier A#2 once those land.
- Do NOT run W7/W8 (64k/128k) without a matched Apple-Silicon
  baseline; partial cells without comparators are misleading. M_e adds
  the long-context cells later.
- Do NOT mix CUDA fp8 KV numbers with Metal BF16 KV numbers in the
  same cell. The cross-backend M6 matrix has a footnote column for KV
  dtype; respect it.

## 8. References

- M6 CUDA snapshot win:
  [`2026-05-07-m6-world-rank-snapshot-cuda.md`](../experience/wins/2026-05-07-m6-world-rank-snapshot-cuda.md)
- Master roadmap:
  [`backend-unification.md`](backend-unification.md) §M6
- Cross-vendor gauntlet (where this snapshot becomes the A4 row):
  [`M_e-world-first-bench-gauntlet.md`](M_e-world-first-bench-gauntlet.md)
- Metal current-state baseline (305.5 tok/s @ 1024/256 step-driver):
  [`mlx-backend-roadmap.md`](../projects/mlx-backend-roadmap.md)
- Bench-and-trace spec (mandatory report sections):
  [`bench-and-trace-spec.md`](../bench-and-trace-spec.md)
