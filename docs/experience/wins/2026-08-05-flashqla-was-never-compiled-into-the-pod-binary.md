# FlashQLA was never compiled into the pod binary: TTFT 31.08 → 25.01 s — CUDA, 2026-08-05

> Status: Shipped

## Goal

Explain the W8A16 TTFT gap. The
[decode budget](2026-08-04-w8a16-decode-step-kernel-budget.md) put decode at
parity with SGLang and left prefill 1.48× behind (31.08 s vs 21.03 s), unmoved
for three days while nine optimizations landed on decode.

## Parameters

- H20 GPU 6, Qwen3.6-27B W8A16 vs SGLang 0.5.13 on the mechanically repacked
  GPTQ twin (identical int8 values, same `gptq_marlin` kernel)
- `bench-agent-32k-64.jsonl`, c=1, 16 requests × 256 tokens, temp 0, seed
  20260416, 33000 prompt tokens, two reps per arm
- P/D separated: `prefill tok/s = prompt_tokens / TTFT`, `decode tok/s = 1 / ITL`
- Prefill ledger: one cold 33K request at `max_tokens=1`, `nsys
  --cuda-graph-trace=node` (prefill is not captured into a graph)

## The prefill knob is not the gap

SGLang ran `--chunked-prefill-size 8192`, ARLE its 2048 default — an unmatched
parameter. Sweeping both eliminates it:

| arm | TTFT p50 | prefill tok/s | ITL p50 | decode tok/s |
|---|---:|---:|---:|---:|
| ARLE c=2048 | 31.14 / 31.08 s | 1068 | 16.67 ms | 60.0 |
| ARLE c=4096 | 31.07 / 31.02 s | 1068 | 16.68 ms | 60.0 |
| SGLang c=8192 | 21.04 / 21.03 s | 1568 | 17.18 ms | 58.2 |
| SGLang c=4096 | 21.10 / 21.10 s | 1563 | 17.19 ms | 58.2 |

Doubling the chunk moves neither stack. ARLE's explicit values also clamp to
`[128, 4096]` (`infer-api/src/loaded.rs:2085`), so the 8192 arm is unreachable
here regardless.

## The ledger names it

| bucket | ARLE | SGLang | Δ |
|---|---:|---:|---:|
| Marlin GEMM (8448 launches each) | 18.660 s | 18.675 s | −0.015 |
| full attention (FA3) | 1.632 | 1.529 | +0.103 |
| **linear attention + conv1d** | **7.231** | **0.314** | **+6.917** |
| other | 0.361 | 0.422 | −0.061 |
| **GPU idle** | **3.877** | **0.190** | **+3.687** |
| wall | 31.76 | 21.13 | +10.63 |

The two rows in bold account for the entire gap with no residual. Quantized
GEMM is not involved: same kernel, same launch count, 15 ms apart.

ARLE spent 7.23 s in `gated_delta_rule_prefill_recurrent_kernel` — 864 launches
at 7.86 ms each, a serial scan — where SGLang's FLA chunked decomposition
(`kkt_solve` + `h_blockdim64` + `fwd_kernel_o` + `recompute_w_u`) cost 0.31 s.

## Root cause

serve logged it every boot:

```
qwen35.rs:826  FlashQLA chunked GDR unavailable (stub build or non-sm90);
               using the recurrent scan
```

`--qwen35-gdr-chunked` defaults true and FlashQLA has been default-on since
`c2eb5de9e`, but `scripts/pod-build-env.sh` set only `ARLE_CUDA_ENABLE_FA3=1`.
Without `ARLE_CUDA_ENABLE_FLASHQLA_GDR`, `build.rs:1637` skips every
flashqla-gated TileLang row, the runtime probe finds no kernel, and the path
falls back silently. **Every W8A16 prefill number measured 2026-08-02 through
2026-08-04 ran on a binary missing a shipped, licensed, default-on
optimization.**

Two more failures hid behind it, both invisible while the rows were never
generated:

1. `/host/arle-build` was a mixed tree — git at `b7fecaa5d`, individual files
   overwritten by single-file `tn push`. It lacked `4b85750e4`, which targets
   the flashqla rows at `sm_90a`; without it ptxas rejects `setmaxnreg`.
   Replaced wholesale from `git archive HEAD`, with a
   `.arle-source-receipt` since `.git` still reports the old commit.
2. tilelang 0.1.12 renames a kernel parameter that collides with a C++ keyword
   (`do` → `do_1`), which failed `gdr_fq_bwd` codegen. The scalar path already
   stripped that suffix; tensors now do too (`6e3f68fac`).

## Results

| | before | after | Δ |
|---|---:|---:|---:|
| TTFT p50 | 31.08 s | **25.01 s** | **−19.5%** |
| prefill tok/s | 1068 | **1329** | **+24.4%** |
| ITL p50 | 16.68 ms | 16.69 ms | +0.1% |
| decode tok/s | 59.9 | 59.9 | 0.0% |

Two reps per arm, agreeing to 0.08 s TTFT. Decode is untouched, as a
prefill-only kernel should be.

Prefill ledger after:

| bucket | before | after |
|---|---:|---:|
| linear attention + conv1d | 7.231 s | **0.441 s** |
| other (incl. the FlashQLA prep/cumsum rows) | 0.361 | 1.108 |
| GPU busy | 27.884 | **21.871** |
| GPU idle | 3.877 | 3.967 |
| wall | 31.761 | **25.838** |

Against SGLang, TTFT 1.48× → **1.19×**, and GPU-busy time is now within 0.93 s
(21.87 vs 20.94). **The remaining prefill gap is 3.8 s of GPU idle** — ARLE
idles 3.97 s during prefill against SGLang's 0.19 s. That is now the largest
single item in TTFT and it is not a kernel problem.

## Correctness

`GATE_PROFILE=generic lever_gate.sh`, needle ladder
115/300/2000/8000/16000/32000 × 3, `RAW=1 TEMPLATE=qwen3_nonthink
NEEDLE_MAX_TOKENS=256`. Every rung `exact=3 partial=0 miss=0 DET`,
`correctness PASS: summaries=6`. Self-consistency only — no baseline arm,
because this restores an already-licensed default rather than flipping a new
one. FlashQLA's license (chat GSM 95/100 zero disagreements, chat MMLU 80 v 81,
needle 9/9; raw-completion knife-edge flips the named trade) was adjudicated
2026-08-02.

## Learnings

**Rule: a feature flag that defaults true proves nothing about the binary.**
The CLI said on, the code path said on, the docs said default-on since
`c2eb5de9e` — and the kernel was not in the build. Only the serve log knew.
Before attributing a gap to design, grep the boot log for the fallback line of
every optimization the number depends on.

**Rule: a silent fallback is a measurement bug, not a robustness feature.**
`unavailable → use the recurrent scan` kept the server correct and made three
days of prefill numbers meaningless. A fallback on a default-on fast path
belongs at `warn!` at minimum, and a bench harness should record which fast
paths were live.

**Rule: single-file `tn push` creates a tree that no commit describes.** The
pod tree passed `git status --short` with two entries while being a day stale
in the files that mattered. Sync whole trees; leave a receipt when the `.git`
metadata cannot be trusted.

Related: [[feedback_path_probe_before_perf_claim]],
[[feedback_always_sync_latest_delete_stale_pod_trees]],
[[feedback_flag_silent_noop_passes_exit0_smoke]].
