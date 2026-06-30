# DSv4 fixed-band page-attn and TP-safe slot snapshot store

## Context

DSv4 could not treat the KV tier as a serialized whole-slot snapshot forever. The
engine/radix/tier layer owns page identity, while FlashMLA and DSA read their own
fixed-band device layouts. The missing connection was a backend-owned page
metadata path that lets DSv4 lower the host slot page table into FlashMLA without
pretending DSv4 is a sequential paged cache.

## What Worked

- `HostPagedKvPool` now has fixed-band allocation semantics: a DSv4 slot draws the
  complete FlashMLA logical band once, and `truncate_slot` only rewinds the logical
  cursor. It does not release tail band pages after MTP reject or prefix restore.
- `KvBatchDescriptor` carries both live prefix pages (`flat_page_ids`) and the
  complete slot page table (`flat_slot_page_ids`). Existing models keep using the
  live prefix table; DSv4 consumes the full slot table.
- `Dsv4KvAdapter::prepare_kv_batch` mirrors the host slot page table into each
  layer's `TokenKVPool` with `mirror_band`, then advances the FlashMLA cursor.
- Whole-slot and position-0 prefix restore receive the host slot page table from
  `infer-core`, mirror it first, then restore `Dsv4SlotSnapshot` payloads into those
  physical pages.
- TP is supported by deterministic rank-local storage: every rank stores/restores
  its own shard under the same engine key. Hit length, demote room, snapshot fit,
  insert, read/parse/restore success all go through TP min-reduce, so any rank miss
  or failure makes every rank take the same branch.

## Verification

Local gates:

```
cargo test -p infer-seam --release --lib
cargo test -p infer-core --release --lib
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
```

Results:

- `infer-seam`: 34 passed.
- `infer-core`: 83 passed, 1 ignored benchmark.
- `infer-api` CUDA/no-cuda typecheck: passed.

`cargo test -p infer-cuda --features cuda,no-cuda --lib ...` reaches the linker
on macOS and fails on missing CUDA symbols, which is the existing local no-cuda
test limitation. The CUDA crate remains covered locally by the `infer-api`
typecheck; runtime validation must run in the H20 pod.

Pod gates (TP=4, H20, container `sglang-test`, commit `63b59b0c`):

```bash
CARGO_TARGET_DIR=/host/arle-build/target-nccl-dsv4 \
CARGO_NET_OFFLINE=true CUDARC_CUDA_VERSION=12090 CUDA_HOME=/usr/local/cuda \
RUSTC_WRAPPER= CARGO_BUILD_RUSTC_WRAPPER= \
cargo build --release --no-default-features --features cli,cuda,nccl --bin arle
```

Serve:

```bash
RUST_LOG=info CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090 \
DG_JIT_CACHE_DIR=/host/deepgemm-warm \
ARLE_DEEPGEMM_ROOT=/host/arle-build/crates/cuda-kernels/vendor/deepgemm \
ARLE_DEEPGEMM_LIBRARY_ROOT=/host/arle-build/crates/cuda-kernels/vendor/deepgemm/deep_gemm \
ARLE_DSV4_EXPERT_BACKEND=deepgemm ARLE_DSV4_MOE_BACKEND=allreduce \
ARLE_DSV4_INCREMENTAL_KV=1 ARLE_DSV4_DECODE_PHASE_TIME=1 \
INFER_TP_SIZE=4 INFER_CUDA_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
INFER_DSV4_MAX_SEQ_LEN=16384 \
/host/arle-build/target-nccl-dsv4/release/arle serve \
  --backend cuda --model-path /host/DeepSeek-V4-Flash-FP8 \
  --port 18207 --bind 127.0.0.1 --max-running-requests 4 \
  --max-prompt-tokens 16384 --max-total-tokens 20000 \
  --dram-fraction 0.5 --kv-ssd-path /host/arle-kv-ssd-page-tier \
  --slot-oversubscription
```

Results:

- Build: passed in 1m47s. Build log shows DeepGEMM native auto-enabled from
  sm_90 + vendored source; no `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE` is required.
- Binary/source hygiene: tracked tree and binary contain no
  `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE`, `ARLE_DSV4_FP8_LINEAR_DEEPGEMM`, or
  `dsv4_deepgemm_cache` strings. The binary still contains the NCCL coordinator
  marker (`minted NCCL unique id`).
- Startup: 4 workers entered lockstep; each rank built with `TokenKVPool`
  `PackedBytes { bytes_per_token: 584 }`, `12977` pages, and the adapter ledger
  reports `layers(kv_pool+dsa_cache)=19933MB`, `prefill_linear=68MB`.
- Correctness smoke: raw completions returned valid text for 3/4 short prompts:
  `"hello"`, `"4"`, and `"CUDA is a parallel computing platform..."`. The exact
  prompt `Say exactly: hello` can stop after one special token and decode to empty;
  that is a tokenizer/template stop caveat, not a NaN or forward failure.
- Position-0 prefix snapshot: repeated 2168-token prompt went from `2.229s` to
  `0.877s`; `/v1/stats` after the second run reported
  `prefix_cache.hits=1`, `hit_tokens=2168`, `hit_rate=0.1667`.
- c=4 perf sample, 4 x 1456-token prompt + 32 decode tokens:
  wall `9.836s`, `5824` input tokens, `128` output tokens,
  `prefill_tok_s=592.1`, `decode_tok_s=13.0`, `total_tok_s=605.1`.
- c=4 decode-count sample with `ignore_eos=true`, 4 x 676-token prompt + 48 decode
  tokens: wall `12.592s`, `2704` input tokens, `192` output tokens,
  `decode_tok_s=15.25`, `total_tok_s=230.0`. Text decoded empty for all four
  responses despite the 192 generated-token accounting; treat this as the next
  API/tokenizer attribution item, not a forward-kernel PASS.
- `/v1/stats` still reports `kv_tier.available=false` in multiproc:
  `not_available_reason="multiproc coordinator: kv_tier/kv_system not yet relayed"`.
  Prefix counters are relayed; slot/page tier counters are not yet visible through
  the coordinator.
- `ARLE_DSV4_DECODE_PHASE_TIME=1` is present in the coordinator and all four worker
  process environments, but `[decode-phase]` lines were not emitted in the service
  log for these HTTP runs. Do not use phase split numbers from this run.

## Problems

- Whole-slot demote/promote is wired with TP scalar consensus, but the multiproc
  coordinator still does not relay `kv_tier`/`kv_system`, so L2/L3 slot counters
  cannot be observed through `/v1/stats` yet.
- Some DSv4 raw completions produce generated tokens that decode to an empty string
  under the HTTP tokenizer path. That must be attributed at token-id level before
  using text-level c=4 runs as correctness evidence.

## Rule

For DSv4 TP, never let a single rank decide prefix or slot-tier reuse. Store bytes
rank-locally, but reduce the decision to a scalar consensus before the scheduler
branches. Page identity must come from the host slot table, and DSA remains a
fixed-band sidecar until it is page-addressable at arbitrary radix boundaries.
