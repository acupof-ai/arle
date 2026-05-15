<p align="center">
  <strong>ARLE</strong><br>
  <em>Pure-Rust runtime for serving, local agents, training, and evaluation. <code>infer</code> is the OpenAI-compatible serving binary; <code>arle</code> is the unified front door.</em>
</p>

<p align="center">
  <a href="https://cklxx.github.io/arle/"><img src="https://img.shields.io/badge/website-cklxx.github.io%2Farle-D97757?style=flat-square" alt="Website"></a>
  <a href="https://github.com/cklxx/arle/actions/workflows/ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/cklxx/arle/actions/workflows/cuda-ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/cuda-ci.yml/badge.svg" alt="CUDA CI"></a>
  <a href="https://github.com/cklxx/arle/actions/workflows/metal-ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/metal-ci.yml/badge.svg" alt="Metal CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/cklxx/arle/releases"><img src="https://img.shields.io/github/v/release/cklxx/arle?include_prereleases" alt="Release"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="docs/http-api.md">HTTP API</a> ·
  <a href="docs/support-matrix.md">Support Matrix</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="ROADMAP.md">Roadmap</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

---

## Quick Start

### 1. Install

**Apple Silicon — Homebrew (recommended):**

```bash
brew install cklxx/tap/arle
arle --doctor
```

**Apple Silicon or Linux x86_64 — one-line installer:**

```bash
curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh | sh
```

The script grabs the matching tarball from the latest GitHub Release,
SHA256-verifies it, and drops the binaries into `~/.local/bin` (override
with `INSTALL_DIR=...`). See [docs/install.md](docs/install.md) for the full
matrix, env-var overrides, and uninstall steps.

**Linux + NVIDIA — pull the published Docker image, no compile:**

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/Qwen3-4B:/model:ro \
  ghcr.io/cklxx/arle:latest \
  serve --backend cuda --model-path /model --port 8000
```

The `:latest` tag tracks the newest non-prerelease release image. Tagged
releases are published as `ghcr.io/cklxx/arle:X.Y.Z` (note: no `v` prefix -
the docker metadata-action strips it). For the current release:
`ghcr.io/cklxx/arle:0.1.5`.

**From source** (any backend; needed for `cpu`, CUDA/TileLang, or local hacking):

```bash
git clone https://github.com/cklxx/arle && cd arle
# Apple Silicon:
cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle
# Linux + NVIDIA:
cargo build --release --features cuda --bin arle
```

### 2. Serve a model

```bash
arle serve --backend metal \
  --model-path mlx-community/Qwen3-0.6B-4bit --port 8000   # Apple Silicon
arle serve --backend cuda \
  --model-path /path/to/Qwen3-4B --port 8000               # Linux + NVIDIA
```

### 3. Talk to it

```python
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Hello from ARLE"}],
).choices[0].message.content)
```

Or with curl: see [`examples/curl_chat.sh`](examples/curl_chat.sh).
More copy-paste paths: [`examples/`](examples/).

### 4. Run the local agent

```bash
arle                                                       # interactive REPL with built-in tools
arle --model-path /path/to/Qwen3-4B run --prompt "Summarize this repo"   # one-shot
arle --doctor --json                                       # self-check, machine-readable
```

CPU-only smoke build (no GPU required, source build):

```bash
cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle
./target/release/arle --doctor
```

---

## Status at a glance

| Backend | Platform | Status | Notes |
|---|---|:---:|---|
| **CUDA** | Linux + NVIDIA | **Stable** | Continuous batching, paged KV, radix-backed reuse, TileLang BF16 attention, custom CUDA quantized decode, CUDA Graph decode, packed paged-prefill for Qwen3 / Qwen3.5. **L4 / Qwen3-4B BF16 + FP8 paged KV (auto): 197 tok/s @ c=16 / 4096-in, peak_active=16 saturated.** |
| **Metal** | Apple Silicon | **Beta** | Live scheduler-backed serving, chunked prefill, replay-backed prefix reuse. Qwen3.5-0.8B MLX 4bit single-request step-driver reaches 305.5 tok/s on M4 Pro 20c; GGUF Q4_K_M exact default is 202.1 tok/s direct, with an opt-in native-q4 Metal load path at 236.7 tok/s direct / 239.8 tok/s step-driver on the matched 1024/256 profile. |
| **Metal DFlash** | Apple Silicon | **Beta — default-on** | Speculative decode for Qwen3 / Qwen3.5. Qwen3-4B bf16 achieves 5.9× decode speedup, Qwen3.5-4B-4bit maintains bit-identical parity, validated for c=1..8. |
| **CPU** | Portable | **Dev-only** | Smoke tests and request-path validation. DeepSeek V4 has a slow Rust reference path for 1B init correctness / HTTP smoke; not a serving-performance target. |

Models: **Qwen3 (0.6B – 72B)** and the **Qwen3.5 family** (including 0.8B GGUF Q4_K_M and 4B hybrid linear + full attention) are supported on CUDA and Metal according to the current matrix. **Qwen3.6 / Qwen3.5-MoE** has a narrow Metal Beta path; CUDA remains stubbed. Next-model priority queue: **DeepSeek V4 (#1, V4-only substrate + CPU reference smoke landed)** then **Qwen 3.6 (#2, planned)**; see [ROADMAP.md §Next-Model Priority Order](ROADMAP.md#next-model-priority-order). DeepSeek V2/V3/R1 support paths are intentionally not carried in the current runtime.

Authoritative matrix (HTTP API tiers, quantization, agent / train / eval surfaces): [docs/support-matrix.md](docs/support-matrix.md).
Stability tiers: [docs/stability-policy.md](docs/stability-policy.md).

---

## Why ARLE

In agent and RL workloads every turn pays a prefill tax: system prompt + history + tool results must be re-processed. As context grows, **prefill dominates latency**. ARLE treats this as the core problem in both serving and agent / RL loops:

- **Multi-turn KV reuse.** Slot-sticky reuse keeps prior-turn KV hot for the next turn. CUDA also includes a radix-backed tiered-KV path (`T0 GPU → T1 host pinned → T2 local disk → T3 cluster-shared`) for full-block reuse and staged readmission, so only the new user message requires prefill each turn when the prefix stays reusable.
- **Paged KV pool.** Main CUDA KV formats use `page_size=16` with direct GPU page attach and tail-page CoW on shared prefixes — predictable accounting, reusable full blocks, cheaper prefix sharing.
- **Shared runtime authority.** `infer`, `arle`, and the in-tree train / eval jobs resolve models and reuse the same Rust runtime / model contracts. Serving, local agent work, and RL tooling stay on one code path instead of drifting across separate stacks.

Architecture deep-dive: [docs/architecture.md](docs/architecture.md) · [docs/codebase-map.md](docs/codebase-map.md).
Latest benchmark snapshots (per change, dated): [docs/experience/wins/](docs/experience/wins/) · run your own with [`scripts/bench_guidellm.sh`](scripts/bench_guidellm.sh).

---

## Entry surfaces

`arle` is the single binary users interact with:

| Command | What it does |
|---|---|
| `arle` (no args) | Interactive agent REPL with built-in `python` and `shell` tools (sandboxed). |
| `arle run --prompt "…"` / `--stdin --json` | Script-friendly one-shot agent prompt. Use `--no-tools` to disable tool execution. |
| `arle serve --backend {cuda,metal,cpu} --model-path …` | Launch the OpenAI-compatible HTTP server through an ARLE-native backend. |
| `arle train {pretrain,sft,grpo,multi-turn,eval}` | In-tree training and RL workflows on the same runtime. |
| `arle data {download,convert}` | Dataset utilities. |
| `arle --doctor [--json] [--strict]` | Self-check: backend, hardware, HF cache, model resolution. CI-friendly. |

The REPL persists line history at `~/.arle-history` and exposes slash commands: `/help`, `/reset`, `/clear`, `/tools`, `/model`, `/stats`, `/models`, `/save`, `/load`, `/export`.

Operators who want only the native serving binary can use `infer` directly (`cargo build -p infer --release --features cuda` on Linux, `--features metal,no-cuda` on Apple Silicon) — same HTTP contract, without the agent / train / data surface.

---

## 📰 Latest Updates

<!-- Keep this list to the last 2 entries. Older history lives in CHANGELOG.md. -->

- **2026-05-15** — DSv4 DeepEP decode now has a default B=1 padded BF16
  reduce-scatter combine path. `ARLE_DSV4_COMBINE_REDUCE_SCATTER=1` folds the
  owner-rank return-side combine into NCCL `ReduceScatter` after each expert
  rank pre-sums padded route outputs by origin peer; set it to `0` to force the
  previous grouped SendRecv combine. Real 8xH20 validation against
  `/root/DeepSeek-V4-Flash` keeps normal streaming output and exact `410`
  arithmetic, with `decode64` at **12.05 post-first tok/s**. The matching
  single-token nsys trace moves the decode wave from **97.071 ms** to
  **94.923 ms**: return combine becomes `ReduceScatter` at **20.44 ms** per
  rank range and residual `SendRecv` falls to **3.26 ms**. The remaining
  bottlenecks are still local expert FP8/FP4 GEMV, AllReduce, attention/MHC,
  launch overhead, async alloc/free, and D2H readbacks. A follow-up
  `ARLE_DSV4_COMBINE_OVERLAP=1` experiment adds a dedicated communication
  stream and routed-output fence, but stays default-off: it returns exact
  `406` while regressing the single-token decode wave to **104.359 ms** due to
  all-reduce variance and cross-stream event overhead. The same binary with
  overlap disabled keeps the **12.05 post-first tok/s** decode64 baseline.
  Reusing incremental attention projection scratch for `c_q`, `c_q_normed`,
  `q_raw`, `kv_raw`, and `kv_normed` is a positive follow-up: the single-token
  nsys wave moves to **90.946 ms**, `cuMemAllocAsync` drops from **6,760** to
  **5,040** calls, `cuMemFreeAsync` drops from **3,048** to **1,328**, and the
  trace-off `decode64` smoke remains normal at **11.89 post-first tok/s**. A
  fresh direct single-token breakdown records **105.205 ms** for the same
  current default path and makes the ranked bottleneck explicit: **16,177**
  CUDA launches, **20.122 ms** reduce-scatter, **11.474/11.109 ms** FP8/FP4
  expert GEMV, **8.978 ms** all-reduce, attention/MHC/route kernels, and
  **347** D2H synchronization calls over only **44,044 B** of D2H activity.
  The opt-in route-grouped path now keeps grouped expert weight/scale pointer
  tables in layer-load-time caches, cutting its H2D activity from **1,918**
  calls / **374,752 B** to **440** calls / **7,808 B** and moving that trace
  from **105.808 ms** to **94.828 ms** while still returning `406`; it remains
  default-off until route-wise GEMV is replaced by true grouped GEMM/DeepGEMM.
- **2026-05-14** — DeepSeek V4 8xH20 serving now has committed decode and
  DeepEP-style MoE trace records against true `/root/DeepSeek-V4-Flash` with
  FP8 KV. The runnable TP=8/EP=8 layout returns normal multi-token math and
  writing output, while the 1,039-token prefill trace identifies return-side
  MoE combine exchange and local experts as the concrete blockers. A gated
  `ARLE_DSV4_COMBINE_DTYPE=fp8` experiment is functionally correct but remains
  opt-in because it is not faster than the BF16 combine default. Per-layer MHC
  scratch reuse raises the latest trace-off smoke throughput to **6.2-7.3
  tok/s** on short math/writing cases without changing output correctness. The
  gated grouped expert harness now caches per-layer weight pointer arrays and
  launches indexed active experts, improving the raw grouped prototype while
  keeping it default-off until real grouped GEMM/DeepGEMM replaces the current
  GEMV kernels. A follow-up pair GEMV kernel now fuses grouped `w1`/`w3`
  gate/up launches and is visible in DeepEP decode nsys, but it remains opt-in:
  the decode window is still dominated by NCCL send/recv plus alloc/free and
  launch churn. A route-wise grouped expert experiment also remains opt-in:
  it removes the local-count D2H readback, but real 8xH20 nsys regresses the
  single-token decode wave to **145.7 ms** because route-wise FP4 GEMV over
  fixed padded slots costs **35.9 ms** per rank range. A clean decode-only
  HTTP comparison also keeps pair GEMV default-off: default split expert GEMV
  reaches **11.79 post-first tok/s** on `decode64`, while
  `ARLE_DSV4_PAIR_EXPERT_GEMV=1` reaches **7.70 tok/s**; both paths return
  normal text and `410` for the arithmetic check. A full-write temporary
  allocation cleanup then switches selected decode buffers from zeroed to
  uninitialized allocations: HTTP `decode64` reaches **11.99 post-first tok/s**,
  math still returns `410`, `cuMemsetD8Async` drops **8,789 → 2,957 calls**, and
  the isolated single-token wave is **112.7 ms**. Per-layer DeepEP
  dispatch scratch reuse further raises default
  short math smoke to **7.7-7.8 tok/s** and cuts Nsight
  `cuMemAllocAsync`/`cuMemFreeAsync` calls in the 8-token window from 136,825 to
  111,531. A follow-up single-token nsys window now isolates one generated
  decode token at **266 ms wall**: `cuStreamSynchronize`, async allocation/free,
  launch/memset churn, and NCCL send/recv dominate before attention or GEMV.
  Reusing send-route token/slot buffers and deleting the unused expert-token
  pack output keeps short DeepEP smoke at **7.94-8.09 tok/s** while reducing
  single-token decode allocator calls by 883. Reusing recv/local route scratch
  for B=1 decode raises the latest short smoke to **8.24-8.79 tok/s** and cuts
  the isolated single-token nsys wave to **148 ms wall**, with decode-only
  `cuMemAllocAsync`/`cuMemFreeAsync` calls down to **9,480/9,488**. A further
  route-logits scratch cleanup lowers allocator calls again to **9,136/9,144**,
  though its single capture is a call-count cleanup rather than a confirmed
  wall-time win. A refreshed 2026-05-15 nsys run isolates the current
  single-token decode wave at **158 ms wall** and shows the concrete remaining
  stack: async allocation/free, launch/memset churn, D2H routing readbacks,
  NCCL SendRecv/AllReduce, local expert FP8/FP4 GEMV, then attention/MHC. That
  led directly to shared expert scratch reuse plus an in-place BF16 add kernel:
  short math/writing smoke now reaches **9.07-9.50 tok/s**, and the isolated
  single-token nsys wave drops to **140 ms wall** with allocator calls down to
  **7,416/7,424**. A current follow-up nsys run validates direct packed
  segment input for local expert `w1`/`w3`: the streaming output remains
  `霓虹`, decode-only `cuMemcpyDtoDAsync_v2` falls from **871 calls / 1.795 ms**
  to **613 calls / 1.240 ms** per rank range, and the same one-token wave is
  **145 ms wall**. Remaining targets are fewer host route readbacks,
  graph/lifetime cleanup, lower-latency/overlapped DeepEP exchange, and true
  grouped GEMM/DeepGEMM. Reusing per-layer hidden scratch for incremental HC
  pre-projection and RMSNorm temporaries then cuts alloc/free/memset calls by
  **1,376 each** and moves the same single-token wave to **135 ms wall**; the
  current top costs are launch/runtime overhead, D2H route readback, NCCL
  SendRecv/AllReduce, and local expert FP8/FP4 GEMV. Reusing the default
  AllGather count matrix for both send and receive counts removes the
  redundant 32-byte send-count D2H readback, cutting decode-only D2H calls
  **887 → 543** and the same wave to **130 ms wall**. The next count-side
  target was the remaining 256-byte all-rank count matrix readback; the B=1
  padded dispatch path now ships by default, skips its unused send-count kernel,
  removes that readback plus the count AllGather, and moves the same single
  token wave to **124 ms wall** with decode-only D2H calls **543 → 344**.
  The return-side combine path now also pre-sums valid padded route rows into
  one BF16 row per origin peer before the return exchange, reducing returned
  combine rows by **8×** and moving the wave to **112 ms wall**; `SendRecv`
  time falls **25.211 → 23.329 ms** per rank range. The B=1 dispatch side now
  also fuses hidden rows and route metadata into one BF16 payload by appending
  the 3xI32 metadata as raw 16-bit words, reducing SendRecv launches
  **1,032 → 688**, raising HTTP `decode64` to **12.22 post-first tok/s**, and
  recording the latest isolated nsys wave at **119.0 ms** while keeping the
  `霓彩` and `410` checks correct. Remaining hard blockers are
  NCCL SendRecv/AllReduce, launch/runtime and allocator/memset/free churn, the
  local-count D2H, and local expert FP8/FP4 GEMV. A default-path single-expert
  `w1`/`w3` pair GEMV experiment is now available only as
  `ARLE_DSV4_PAIR_EXPERT_GEMV=1`: the 8xH20 trace kept output correct but
  showed the FP4 pair kernel is slower on B=1 decode, so it stays default-off
  while the main compute target remains true grouped GEMM/DeepGEMM. The
  route-wise grouped expert path now also has a pair route GEMV follow-up:
  output remains `霓彩` and the isolated wave is **117.9 ms**, but nsys shows
  the token is still dominated by `ncclDevKernel_SendRecv`
  (**50.3 ms/rank-range**), FP4 route pair GEMV (**19.6 ms**), FP4 route
  `w2` GEMV (**10.5 ms**), FP8 GEMV (**9.4 ms**), plus allocation and launch
  overhead, so the path stays opt-in. A subsequent default-path incremental
  stream scratch recycle lowers the warmed single-token nsys wave
  **128.1 → 111.8 ms** and cuts allocator/free calls
  **8,453/6,048 → 7,757/5,352**, while trace-off HTTP `decode64` stays flat at
  **11.48 tok/s**. Reusing GPU compressor projection scratch cuts allocator/free
  calls again to **6,765/4,360** but does not improve HTTP throughput; the
  end-to-end blocker remains NCCL plus D2H synchronization and local expert
  GEMV. Reusing B=1 incremental attention scratch then cuts warmed decode
  free calls to **3,048** without retaining prompt-sized prefill buffers, and
  the isolated single-token nsys wave is **97.0 ms**; the profiler shows the
  token is still dominated by
  NCCL SendRecv/AllReduce, D2H route-count synchronization, launch/runtime
  overhead, local FP8/FP4 expert GEMV, and attention/MHC kernels rather than
  sampler.
  Evidence:
  [`docs/trace-artifacts/2026-05-14-dsv4-deepep/`](docs/trace-artifacts/2026-05-14-dsv4-deepep/),
  [`docs/trace-artifacts/2026-05-15-dsv4-deepep/`](docs/trace-artifacts/2026-05-15-dsv4-deepep/)
  and
  [`docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md`](docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md).
- **2026-05-10** — 🎉 W4-hybrid prefill graph capture **closes 4k/c=4 SGLang +76.6% gap** via Path B.2 bucketed allocation key (`a56b7a9`/`c44788f`). Engine-side TTFT p50 **2000ms → 150ms = -92.5%** improvement on RTX 4070 Ti SUPER 16GB (server-side `/v1/stats engine_ttft_us` ground truth; client-side guidellm 0.6.0 broken — bench tool bug isolated). Throughput **+632%** in 60s window. Bucketed `page_indices_len` (64-entry) + `prefix_token_rows_len` (128-row) reduce capture key churn from 388 unique → **7 unique** with **98.5% LRU dominant key reuse**. Codex's "second-order bucketing" insight (captured scalar launch parameters use bucket capacity, not exact dim from first capture) was load-bearing; new anti-pattern in skill v1.7.0 catalog. Opt-in via `INFER_PREFILL_GRAPH=1` + `INFER_HYBRID_W4A8_PREFILL=1`. Plus **RoPE scaling support** (YARN / Linear / NtkAware) wired through qwen3-spec + qwen35-spec + `precompute_rope_with_scaling`. Evidence: [`docs/experience/wins/2026-05-10-bench-40-pathB2-tier1-strong-proceed.md`](docs/experience/wins/2026-05-10-bench-40-pathB2-tier1-strong-proceed.md), [`docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md`](docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md).

Full history: [CHANGELOG.md](CHANGELOG.md). Next up: [ROADMAP.md](ROADMAP.md).

---

## Documentation map

- [docs/http-api.md](docs/http-api.md) — HTTP route contract, streaming behavior, boundary guarantees
- [docs/support-matrix.md](docs/support-matrix.md) — backend / model / quant / API support tiers
- [docs/stability-policy.md](docs/stability-policy.md) — stability levels and compatibility posture
- [docs/architecture.md](docs/architecture.md) — package boundaries and dependency direction
- [docs/codebase-map.md](docs/codebase-map.md) — workspace layout and main execution paths
- [docs/environment.md](docs/environment.md) — environment variables and runtime knobs
- [docs/troubleshooting.md](docs/troubleshooting.md) — common build / runtime errors and fixes
- [docs/comparison.md](docs/comparison.md) — how ARLE compares to vLLM / SGLang / mistral.rs / llama.cpp
- [docs/release-checklist.md](docs/release-checklist.md) · [docs/perf-and-correctness-gates.md](docs/perf-and-correctness-gates.md)
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor setup, validation, release expectations
- [SECURITY.md](SECURITY.md) — vulnerability reporting policy
- [examples/](examples/) — copy-paste smoke paths (curl, OpenAI SDK, Docker, Metal, train fixtures)
- [docs/index.md](docs/index.md) — maintainer-facing PARA index, plans, and experience logs

---

## License

[MIT](LICENSE)
