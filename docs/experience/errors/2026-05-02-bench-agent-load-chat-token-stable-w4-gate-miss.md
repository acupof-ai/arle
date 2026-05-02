# Chat Token-Stable W4 Gate Miss - agent-load benchmark, agent-w4-tool-resume, arle-cuda, 2026-05-02

## Context

After A6 showed canonical W4 still attaching only 32 tokens, this tranche tested fork B: make chat history serialization token-stable at the assistant tool-call boundary. The code change was intentionally scoped to `crates/chat/` and did not touch scheduler, prefix-cache, admission, or KV-tier logic.

The local protocol fix removed the extra blank line before `<tool_call>` when an assistant message has empty `content`, so a generated warmup continuation shaped as `<tool_call>...` can be byte-stable with the later resume history rendering.

## Hypothesis

If the W4 miss was caused only by chat-template newline drift, canonical W4 should move from the A6/A3 range of `matched_prefix_tokens=32` and `avoided-prefill ~= 0.35-0.38%` toward the one-session control result (`matched_prefix_tokens ~= 8256`, `avoided-prefill ~= 96.2%`).

Gate target: canonical W4 `avoided-prefill >= 90%`; partial success threshold from the task was `>= 50%`.

## Params

- GPU: NVIDIA L4 24GB.
- Model: `models/default -> Qwen3-4B`.
- Server:
  - `./target/release/infer --model-path models/default --port 8000 --num-slots 8 --max-seq-len 12288 --kv-cache-dtype fp8 --mem-fraction-static 0.85 --t1-host-pinned-high-water 0.98 --t1-host-pinned-low-water 0.95 --t1-host-pinned-capacity-mb 32768 --t1-host-pinned-min-prompt-tokens 4096`
- Env:
  - `CUDA_HOME=/usr/local/cuda`
  - `CARGO_HOME=/tmp/cargo-home-local`
  - `PEGAINFER_CUDA_SM=89`
  - `LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64`
  - `ZIG=/root/.local/lib/python3.12/site-packages/ziglang/zig`
  - `INFER_TILELANG_PYTHON=/usr/bin/python3`
- Trace:
  - `python3 scripts/bench_agent_trace.py --workload agent-w4-tool-resume --server http://127.0.0.1:8000 --label chat-token-stable-w4 --out bench-output/2026-05-02-agent-load-chat-token-stable-w4/results.json --trace-out bench-output/2026-05-02-agent-load-chat-token-stable-w4/trace.jsonl`

## Results

The gate missed. The run did not complete cleanly because the server was killed by the Linux OOM killer after 189 admitted requests, but the prefix signal was already stable and below threshold.

| metric | e577d670 baseline | A3 03253745 | A6 fbb39407 | chat token-stable run |
| --- | ---: | ---: | ---: | ---: |
| matched prefix tokens | 32 | 32 | 32 | 16-32 across 182 attach events |
| avoided-prefill ratio | 0.35% aggregate / 0.37% latest | 0.38% latest | ~0.35-0.38% | 0.377% mean over attach events |
| resume prefill tokens | 8539 latest | 8456 latest | ~8.2k-8.5k | 8.1k-8.6k |
| successful scored turns | 78 / 128 | 76 / 128 | gate miss | 54 / 128 before OOM |
| resume TTFT p50 / p99 | 4340.3 / 26279.7 ms | 2430.6 / 13542.5 ms | gate miss | 9254.7 / 25658.4 ms from successful scored turns |

Server log samples:

- `Request 150 -> slot 3 (prompt=8496 tokens, radix_gpu_attach=32, queue=0)`
- `Request 189 -> slot 2 (prompt=8480 tokens, radix_gpu_attach=32, queue=0)`
- Parsed server log: `attach_count=182`, `matched_min_max=16..32`, `prompt_min_max=8167..8604`, mean avoided-prefill `0.3766%`.

The client summary reported:

- `turns OK: 182 / 256`
- `scored turns OK: 54`
- `tokens total: 13590`
- `TTFT p50/p99: 9254.7 / 24229.2 ms`
- `W4 scored resume: resume turns=54 TTFT p50/p99=9254.7/24229.2 ms`
- `stats_after: unavailable`

The server was killed by host OOM:

```text
Out of memory: Killed process ... (infer) ... anon-rss:51311468kB
```

## Root Cause

The blank-line boundary was a real local serialization instability, but it was not the canonical W4 binding constraint.

The remaining mismatch is semantic: the warmup request caches the actual assistant continuation generated after `<|im_start|>assistant\n`; the resume request later injects a canonical synthetic assistant tool-call history plus tool result. Token-walk radix lookup can only follow exact token IDs, so it still stops at the first divergent continuation rather than selecting the session's warmup transcript as the resume base.

This is consistent with the reference design read before the change: SGLang's streaming-session path saves and restores a session slot by session identity, bypassing raw prompt-token equality for the append-only session state. vLLM's prefix cache hashes token blocks, so it has the same requirement: exact token sequence equality is needed unless a higher-level session/KV connector maps the resume to prior state.

The OOM is a separate A6 substrate problem under the 32GiB host-pinned retention setting. It made this particular bench incomplete, but it did not hide a partial B win: before OOM, every observed resume attach still stayed at 16-32 tokens.

## Fix

Implemented and tested the narrow chat fix anyway:

- `crates/chat/src/protocol.rs`: shared assistant tool-call rendering helper.
- Empty assistant tool-call history now renders as `<|im_start|>assistant\n<tool_call>\n...` instead of inserting a second blank line.
- Added a unit test that builds a warmup generation prompt, appends the expected `<tool_call>` block, and asserts that the resulting completed warmup prefix is a byte prefix of the resume prompt.

Verification:

- `cargo fmt -p chat --check`
- `cargo test -p chat` (`22 passed`)
- `cargo build --release -p infer`
- Canonical W4 trace run above; gate failed.

## Rule

Do not spend more cycles on chat-template-only fixes for canonical W4 unless a local tokenizer diff proves the generated warmup assistant continuation and injected resume assistant history are identical past the assistant header.

Next architectural move should be fork A: session-id keyed lookup/readmission that attaches the prior warmup KV by session state, then prefill only the tool-output delta. B alone stayed under the 50% partial-success threshold.

