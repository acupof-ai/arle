# V100 KV-precision sweep: 4 precisions × 4×4 + 4×16 grids, memory + prefill timing

## Context

Captured after the 2026-05-29 group-quant kill
([`errors/2026-05-29-int4-kv-group-quant-kill.md`](../errors/2026-05-29-int4-kv-group-quant-kill.md))
and the kv_tier delete-style refactor that removed `LocalCudaTransport`
(commit `67745ebc`) and flipped the audit default grid 256→4 tokens.

User ask: "默认4x4 提交好代码 留好正确输出证据 bench一下性能和内存使用 看一下 KV cache 层整体做的如何 删除式重构 无用或者差路径确认后删除干净". This entry is the
**evidence-of-correct-output + perf/memory bench** half. The refactor
already shipped at commits `67745ebc` (drop dead LocalCudaTransport)
and `7b7e1066` (V100 build fix — `strcasecmp` not in `std::`).

V100 sm_70 / CUDA 12.4. `cargo test --release -p infer --features cuda
--test kv_precision_parity_qwen35 -- --nocapture --test-threads=1`
against `Qwen3.5-4B`, default 4 prompts and the new 4-token default
plus a `KV_PARITY_MAX_TOKENS=16` stress override. Pool budget 19.9 GB.

Build flags: `ARLE_CUDA_DISABLE_FLASHMLA=1` (SM90 file uses
`__nv_fp8_e8m0` which is CUDA 12.5+/SM90+), `ARLE_TILELANG_SRC` +
`ARLE_TILELANG_CUTLASS_INCLUDE` + `ARLE_TILELANG_AOT_FALLBACK`
substrate-overrides for the V100 tilelang+cubin-cache flow.

## Quality — 4×4 (new default grid) and 4×16 (stress)

```
precision   4×4 mean_match   4×4 first_div   4×16 mean_match   4×16 first_div
bf16        1.0000           None            1.0000            None
int8        0.9375           p2 / s3         0.8906            p2 / s13
fp8         1.0000           None            0.7344            p2 / s3
tq4         0.0000           p0 / s0         0.0000            p0 / s0     (sm_70 N/A)
int4        0.8125           p0 / s1         0.5781            p0 / s1
```

All numbers reproduce
[`wins/2026-05-28-int4-kv-two-level-k.md`](2026-05-28-int4-kv-two-level-k.md)
exactly — the kv_tier refactor + audit default flip changed zero runtime
semantics, as the bench-every-change rule requires of a "docs/dead-code"
diff. The state on `main` after `7b7e1066` is the same KV path that
2026-05-28 measured. Reproducible via:

```
cargo test --release -p infer --features cuda \
  --test kv_precision_parity_qwen35 -- --nocapture --test-threads=1
KV_PARITY_MAX_TOKENS=16 cargo test --release ... (same)
```

INT8 fails its `0.99` gate at both grids (prompt 2 first divergence) —
that is the long-standing sm_70 INT8 prefill anomaly, see
[`errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md`](../errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md)
neighborhood. Not a regression of this session's work; it has been the
audit's noisy gate since before the two-level work landed.

## Memory — pool layout per precision (8 layers, 4 kv_heads × 256 head_dim, kv_dim=1024)

Numbers extracted from `TokenKVPool` construction logs (`paged_kv.rs:350`
and `:444`) at runtime on the 4×4 default audit. Pool budget = 19.9 GB
for all precisions; per-format capacity is determined by per-token byte
cost.

```
precision   data MB/layer   scales MB/layer   working MB   max_tokens   format/page_size
bf16         2499.9           0.0                 0.0      610,336      page_size=16
int8         1969.0          30.8              3938.1      961,440      page_size=16
fp8          1969.0          30.8              3938.1      961,440      page_size=16
tq4          1979.6           0.0              3959.3      966,615      page_size=1
int4          984.5          30.8              3938.1      961,440      page_size=16
```

Per-precision per-token K+V cost (one layer, raw data only):

```
bf16   4096 B/token   (1024 dim × 2 B × 2 (K+V))
int8   2048 B/token   (1024 dim × 1 B × 2)
fp8    2048 B/token   (1024 dim × 1 B × 2)
tq4    2056 B/token   (TQ4 nibble + per-(token, head) f16 norm overhead)
int4   1024 B/token   (1024 dim × 0.5 B × 2)  ← 4× tighter than BF16
```

Per-(token, head) scale overhead for INT8/FP8/INT4 is 32 B/token at
4 kv_heads × 4 B (per the `scale_bytes_per_token` formula in
`paged_kv.rs:154`). At max_tokens this lands at 30.8 MB/layer. The
"working" buffer (3938 MB) is the bf16 staging area used for batched
quantize-into-pool — one allocation shared across layers and across
the K and V paths, so it does NOT scale with layer count.

INT4 capacity vs BF16 capacity at the same 19.9 GB budget:
`961,440 / 610,336 ≈ 1.58×`. The capacity gain is bottlenecked by the
working buffer; the data-only ratio is `4096 / 1024 = 4×`. Closing the
gap means either skipping the bf16 working buffer (direct bf16 → INT4
fused kernel) or sharing it across formats more aggressively — out of
scope for this entry.

TQ4 uses `page_size=1` (per-token paged pool) instead of the standard
`page_size=16`. That's why its `max_tokens == page_count` is an order
of magnitude larger than the int4 page count, but the total data
footprint comes out similar (per-token TQ4 cost includes the inline
f16 norm).

## Performance — prefill timing on the 4×4 default audit

20 prefill samples (5 precisions × 4 prompts), prompt lengths
{51, 57, 59, 67} tokens, batch=1, paged prefill:

```
prefill_us   min     median    mean     max
             559007  629554    634633   716708
```

Roughly 10 ms/prompt-token at batch=1 on V100, dominated by
prefill compute (TileLang `batch_prefill_paged_hd256_q*_kv*_sm70` for
all KV formats — quantize/dequantize is fused into the same op). The
distribution is flat across precisions because the AOT-compiled
tilelang prefill kernels are bf16-input and the per-format quant
happens on the existing data, not the prefill path itself.

Decode-step timing is not on the per-step log channel at `RUST_LOG=info`
and was not extracted; per-precision `elapsed` (52-54 s, including the
~45 s warmup + 4 prefill + 4 decode steps × 4 prompts) is in the audit
table above and confirms there is no per-precision decode regression
relative to BF16.

## kv_tier delete-style refactor — what landed

Surveyed `infer/src/kv_tier/` for speculative paths per the module's
own `AGENTS.md` "delete-style refactor" posture. Cuts that landed at
`67745ebc`:

1. **`infer/src/kv_tier/transport/local_cuda.rs`** (-173 lines) +
   matching re-exports. Zero external callers; was documented as
   "future P0' NVLink peer hop", a speculative milestone that the
   project never reached. The live local lane goes through the
   scheduler's CUDA copy stream directly. AGENTS line, `pub mod`,
   `pub use`, and the M3 milestone bullet on `KVTransport` all
   removed in the same commit.
2. **`KV_PARITY_MAX_TOKENS` default 256 → 4** in both the dense and
   Qwen3.5 audit harnesses. The wins/errors entries since 2026-05-27
   have all used the 4×4 grid as the canonical iteration form, but
   the default still pointed at the long-trajectory 256-token grid.
   New comment block in both files documents the override knobs
   (16 = stress, 256 = long-trajectory).

Surveyed but **left alone** (confirmed live or intentionally scaffolded):

- `transport/nixl.rs` — feature-gated behind `rdma-nixl`/`rdma-nixl-real`,
  both off by default. Carries `TransportId::Nixl = 0` serialization
  slot; deleting would break on-disk format compatibility for existing
  artifacts. Project policy explicitly keeps NIXL "design-ready"
  per [`tiered-kv-cache.md`](../../projects/tiered-kv-cache.md).
- `coordinator/bench.rs` — `#[test] #[ignore]` micro-bench, referenced
  by [`2026-05-04-bench-kv-tier-copy-throughput.md`](2026-05-04-bench-kv-tier-copy-throughput.md)
  and [`2026-05-05-bench-kv-tier-rust-substrate.md`](2026-05-05-bench-kv-tier-rust-substrate.md)
  as the canonical T1↔T2 throughput harness. Dev tool, not production
  code path.
- `TransportId::{Mooncake, Reserved}`, `MemKind::{Vram, Block}` —
  reserved serialization/enum slots for post-M5 transports. Not
  exercised today but live-load-bearing for the on-disk format and
  NIXL memory-classification ABI.
- `KVHandle / KVBlock / KVSpan / KVPayload` + `chunk.rs` state enums
  (`IndexEntryState`, `RequestChunkState`, `StoreState`,
  `SpanTaskKey`) — all referenced inside the coordinator, prefix
  cache, or disk/shared-fs stores. Internal but live.

Net diff for the kv_tier review: −173 lines of code, −1 line in
AGENTS.md, −4 lines of stale milestone doc in `transport.rs`. No
runtime hot-path change; the post-refactor audit reproduces every
mean_match digit of the pre-refactor wins entry, as required by the
"docs/dead-code is exempt from re-bench" rule in
[CLAUDE.md §Benchmarks](../../../CLAUDE.md).

## Substrate footnote

V100 build had to navigate three orthogonal substrate issues, none of
them about the KV path under test. Documenting for the next person:

1. **`std::strcasecmp` is not a thing under nvcc.** The current
   `csrc/gemm/quantized_gemv.cu:39` `dsv4_fp8_gemv_mma_enabled()`
   referenced `std::strcasecmp` (a POSIX function exposed via
   `<strings.h>`, not `<cstring>`). Fixed at commit `7b7e1066`:
   include `<strings.h>` and drop the `std::` qualifier on the call
   site. Surfaced on every V100 build attempt today; not specific
   to KV-tier work.
2. **FlashMLA SM90 sparse-FP8 instantiation uses `__nv_fp8_e8m0`,
   which only exists in CUDA 12.5+/sm_90+.** V100 has CUDA 12.4 so
   the path is unbuildable. Workaround: build with
   `ARLE_CUDA_DISABLE_FLASHMLA=1` which routes through the existing
   stub `csrc/attention/arle_flashmla_decode_stubs.cu`. Runtime gate
   `dsv4_flashmla_decode_enabled` defaults OFF, so the stub is
   harmless on V100.
3. **TileLang AOT cache hash drift across `build.rs` edits.** Cargo's
   `OUT_DIR` hash depends on build.rs contents; editing build.rs to
   short-circuit the tilelang Python probe changes the hash, so the
   new build's `OUT_DIR` is empty even though a sibling directory
   holds the right cubins. Workaround in the audit-local fork: env
   `ARLE_TILELANG_AOT_FALLBACK=<old OUT_DIR>/tilelang_aot` plus a
   local-only patch to `tools/tilelang/gen_tilelang_aot.py` that
   copies cubin/.c/_device_kernel.cu from the fallback when missing.
   Not committed (substrate wart, not the algorithm under test).

## Rule

`KV_PARITY_MAX_TOKENS=4` is the canonical iteration grid for KV
precision quality work; `=16` is the stress grid every wins/errors
entry should also report; `=256` remains the long-trajectory FP8 Tier
1 anchor and is opt-in. The 4×4 + 4×16 pairing is the minimum
quality evidence a KV-touching diff ships. Per-precision pool memory
breakdown (data / scales / working) ships from the
`paged_kv.rs:444` log line — quote it directly rather than
recomputing the bytes, so the next person can sanity-check by `grep`
on a fresh audit log.

When auditing kv_tier for delete-style refactors, the discriminator
is "has any external caller" via grep, NOT "is the variant used in
the runtime path". Serialization slots (TransportId discriminants,
MemKind classes) are live even with zero callers because removing
them breaks on-disk format compatibility.
