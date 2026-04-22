# Metal DFlash Qwen3.6/Qwen3.5 — dflash-mlx alignment plan

**Status**: active, execution started 2026-04-22  
**Scope**: Apple Silicon Metal backend, `Qwen3.6-35B-A3B` / `Qwen3.5-35B-A3B`
single-request DFlash hot path, plus shared verify kernels used by packed
verify.  
**Baseline reference**: local same-machine serial runs currently show
`Qwen3.6-35B-A3B` DFlash below baseline on Metal, while
[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx) reports
`2.20x` at `1024` generated tokens on `Apple M5 Max`.

---

## 1. Goal

Close the largest implementation gaps between this repo's Metal DFlash path and
`dflash-mlx`, in descending order of expected end-to-end impact:

1. Keep single-row DFlash `drafted / posterior / acceptance` GPU-resident.
2. Add a true single-row sampled verify path for `Qwen35CompiledModel` that
   uses scalar `cache_pos`, not the packed `cache_pos_arr` path with `B=1`.
3. Wire the already-landed `mlx_batched_sdpa_2pass` kernel into the Qwen3.6
   verify hot path for long-context exact verify.
4. Add `M=16` verify-specialized quantized linear dispatch
   (`verify_qmm` / `verify_linear`) for the target verify pass.

The success condition is not "claim parity with upstream numbers". The success
condition is:

- one clean canonical implementation path per stage,
- no leftover experiment branches,
- serial local validation after every runtime change,
- and a final local benchmark showing whether each stage improved, regressed, or
  stayed flat.

---

## 2. Current gap summary

The current local implementation already has:

- GDR tape recording and tape replay rollback
- varlen tape replay for packed rollback
- single-row and batched sampled verify APIs
- draft sink/window cache compaction

The current local implementation is still missing the following
`dflash-mlx`-level behaviors:

### 2.1 GPU-resident acceptance

`dflash-mlx` keeps `drafted`, `posterior`, and prefix-acceptance computation on
the GPU and only materializes what the host truly needs at the end of the
cycle. This repo still materializes:

- draft suffix rows to `Vec<u32>`
- verify posterior rows to `Vec<u32>`

That creates a per-block GPU->CPU->GPU roundtrip in the speculative hot path.

### 2.2 Native single-row verify

This repo's single-row DFlash path currently reuses packed sampled verify with
`B=1`. That is simple, but it routes through the packed full-attention update
path and pays `cache_pos_arr` handling designed for true batched verify.

`dflash-mlx` single-row verify uses the plain target cache path and avoids the
packed-row machinery entirely.

### 2.3 Long-context exact verify for full attention

This repo already ships `mlx_batched_sdpa_2pass` in `crates/mlx-sys`, but the
Qwen3.5/Qwen3.6 target verify path never calls it. Long-context verify still
falls through stock MLX SDPA in the compiled C++ model.

### 2.4 Verify-specialized quantized linears

`dflash-mlx` swaps eligible `nn.QuantizedLinear` modules for
`VerifyQuantizedLinear` and dispatches `m == 16` verify matmuls through custom
Metal kernels. This repo still sends verify matmuls through the generic
`quantized_matmul` path.

---

## 3. Implementation slices

Each slice must land as one complete refactor unit with:

- code,
- docs update,
- local validation,
- bench entry under `docs/experience/wins/`,
- and a commit.

### Slice A — GPU-resident single-row acceptance

**Objective**

Remove the Rust-side whole-block CPU materialization of draft and posterior
tokens from the single-row Qwen3.5/Qwen3.6 DFlash path.

**Primary files**

- `infer/src/backend/metal/dflash.rs`
- `infer/src/backend/metal/request_state.rs`
- `docs/resources/metal-dflash.md`
- `docs/experience/wins/2026-04-22-bench-guidellm-qwen36-dflash-gpu-acceptance.md`

**Deletion / refactor target**

- delete the remaining single-row "sample to `Vec<u32>` first" flow
- converge on one canonical path for single-row block acceptance

**Acceptance**

- no whole-block CPU posterior materialization in single-row DFlash
- `cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact`
- `cargo +stable clippy -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench --bin metal_request -- -D warnings`
- `metal_request` smoke exits `0`
- one serial `baseline -> dflash` benchmark on the same machine window

### Slice B — native single-row sampled verify

**Objective**

Add a dedicated `Qwen35CompiledModel` single-row sampled verify API and switch
single-row DFlash to it.

**Primary files**

- `crates/mlx-sys/src/lib.rs`
- `crates/mlx-sys/src/mlx_qwen35_model.cpp`
- `infer/src/backend/metal/qwen35.rs`
- `infer/src/backend/metal/dflash.rs`
- `docs/resources/metal-dflash.md`
- `docs/experience/wins/2026-04-22-bench-guidellm-qwen36-dflash-single-row-verify.md`

**Deletion / refactor target**

- delete the `B=1 packed verify` dependency from the single-row path
- keep packed sampled verify only for actual packed verify

**Acceptance**

- single-row DFlash no longer routes through `cache_pos_arr`
- serial validation as in Slice A
- same-session serial benchmark recorded before/after

### Slice C — long-context `batched_sdpa_2pass` hookup

**Objective**

Use the already-landed `mlx_batched_sdpa_2pass` kernel for long-context verify
on the Qwen3.5/Qwen3.6 full-attention path when the shape contract matches.

**Primary files**

- `crates/mlx-sys/src/lib.rs`
- `crates/mlx-sys/src/mlx_qwen35_model.cpp`
- `infer/src/backend/metal/mlx.rs`
- `infer/src/backend/metal/qwen35.rs`
- `docs/resources/metal-dflash.md`
- `docs/experience/wins/2026-04-22-bench-guidellm-qwen36-dflash-sdpa-2pass.md`

**Deletion / refactor target**

- do not add another parallel SDPA policy layer in Rust
- keep the routing decision in one place, nearest to the compiled verify path

**Acceptance**

- `q_len == 16`, `D == V`, and long-prefix verify can hit 2-pass SDPA
- serial long-generation benchmark recorded with exact same command before/after

### Slice D — verify-specialized quantized linear dispatch

**Objective**

Add an `M=16` verify-specialized quantized matmul path and route eligible
verify-only linear calls through it.

**Primary files**

- `crates/mlx-sys/src/lib.rs`
- `crates/mlx-sys/src/mlx_bridge.cpp`
- `infer/src/backend/metal/mlx.rs`
- `infer/src/backend/metal/ops.rs`
- `infer/src/backend/metal/qwen35.rs`
- `crates/mlx-sys/AGENTS.md` or `docs/resources/metal-dflash.md` if interface docs need refresh
- `docs/experience/wins/2026-04-22-bench-guidellm-qwen36-dflash-verify-qmm.md`

**Deletion / refactor target**

- avoid adding a "try verify qmm then fall back then special-case again"
  ladder in multiple places
- one verify-linear dispatch helper, one fallback policy

**Acceptance**

- verify path can hit specialized `M=16` quantized matmul
- non-verify / non-eligible paths still use stock `quantized_matmul`
- serial benchmark recorded before/after

---

## 4. Validation protocol

Every runtime slice uses the same minimum validation set:

```bash
cargo +stable fmt --all --check
```

```bash
cargo +stable test -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench tests::baseline_compare_default_draft_qwen36_a3b -- --exact
```

```bash
cargo +stable clippy -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench --bin metal_request -- -D warnings
```

```bash
cargo +stable run -p infer --release --no-default-features --features metal,no-cuda --bin metal_request -- --model /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 --prompt hi --raw-prompt --max-new-tokens 2 --dflash-draft-model z-lab/Qwen3.6-35B-A3B-DFlash
```

Benchmark protocol for throughput claims:

- strict serial order: baseline, then DFlash
- same binary, same machine window
- no parallel benchmark processes
- prefer `prompt_tokens=20`, `generation_tokens=1024`, `warmup=1`, `runs=3`
- if the machine is noisy, discard the run and repeat

---

## 5. Stop / revert conditions

Stop and reassess before proceeding to the next slice if any of these occur:

- correctness regression in sampled tokens or rollback tests
- `metal_request` smoke fails
- `clippy -D warnings` requires layering temporary compatibility shims
- a slice cannot be landed without keeping both old and new hot paths alive

If a slice regresses throughput but materially simplifies the implementation, it
may still be kept temporarily only if:

- the old path is deleted,
- the regression is documented in the wins entry,
- and the next slice directly targets the measured regression source.

---

## 6. Commit plan

Planned commit sequence:

1. `docs(metal): plan qwen36 dflash mlx alignment`
2. `perf(metal): keep qwen36 dflash acceptance on gpu`
3. `perf(metal): add qwen36 single-row sampled verify`
4. `perf(metal): route qwen36 verify through sdpa 2pass`
5. `perf(metal): add qwen36 verify qmm path`

The exact subjects may tighten during implementation, but each slice must stay
commit-sized and independently validated.
