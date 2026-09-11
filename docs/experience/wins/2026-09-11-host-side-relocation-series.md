# Host-side relocation series — seam/quant/Metal dead-code cleanup, no runtime delta

Date: 2026-09-11. PRs #263, #293, #307, #341, plus the load-time capability
registry #289 documented in its own section.

> Status: Shipped. Every change here is behaviour-preserving by construction
> (module moves, a deleted dead trait, visibility narrowing, dead-code
> adjudication). The gate is compile + the cited unit tests; no GPU A/B
> applies. The one W2 change with a possible per-step cost — the `Box<dyn
> BackendExecutor>` dispatch of #252 — is covered separately by a matched
> Metal c=1 A/B recorded in this entry.

## Group 1 — behaviour-preserving relocations

### #263 — weight-layout decisions move into `infer-quant`
Commit `669c1ac18` ("refactor(quant): decide weight layouts in infer-quant,
execute them in infer-cuda"). Before, the format/shape/SM-tier gates for a
weight matrix lived in the CUDA loader and launcher paths. After, one host
function `plan_weight_layout(query, caps, policy)` in
`crates/infer-quant/src/layout.rs` makes the decision with zero device calls;
the repack functions in `cuda-kernels` execute unconditionally for a matrix
whose source buffers are present. NVFP4 stays Required (hard error below
sm_80 or on a Marlin shape that is not instantiated); W8A16 and per-channel
FP8 stay Optional and demote to the scalar/dequant path; the DeepGEMM
preflight enters as a host bool.

Files: `infer-quant/src/layout.rs` (+214, new), `infer-quant/tests/layout.rs`
(+195, new), `infer-quant/src/lib.rs`, `cuda-kernels/src/tensor/device_matrix.rs`,
`infer-cuda/src/{dsv4/load.rs,loader.rs,ops.rs,ops/quant_linear.rs,
ops/quant_linear_fp4.rs,ops/quant_linear_fp8.rs}`.

Why behaviour is preserved: this is a host-side decision move — the same
format/shape/SM booleans are computed from the same inputs, only the crate
boundary changes; no kernel call, dtypes, or repack bytes change. The same PR
also restored `metal_kv_memory_probe` as a `crates/cli` example; that is dev
tooling and was exempted in the commit body.

Gate (host): the ten `infer-quant` layout tests pin every decision and
demotion — `nvfp4_hopper_gets_marlin_required_and_sfb`,
`nvfp4_below_sm80_is_a_hard_error`,
`nvfp4_wrong_group_size_is_a_hard_error`,
`nvfp4_sfb_needs_hopper_and_native_bridge`,
`w8a16_fit_is_optional_and_demotes_below_sm80`,
`fp8_per_channel_gets_marlin_and_prefill_flag`,
`dsv4_decode_cache_follows_the_flag`,
`formats_without_a_layout_decision_keep_their_checkpoint_layout`.

### #293 — `KvSlotAccounting` folded back into `KvAllocator`; `infer-util` dropped
Commit `1de6adcfe` ("refactor(seam): fold KvSlotAccounting into KvAllocator;
drop unused infer-util"). `KvSlotAccounting` had split off from
`KvAllocator` only to narrow what `BackendExecutor::submit` could write; #276
removed the `&mut dyn KvSlotAccounting` submit argument, leaving the trait
with no purpose. `alloc`/`truncate_slot` moved back onto the `KvAllocator`
impls for `HostPagedKvPool`, `HipKvPool`, and `VulkanKvPool`; the trait and
the `infer-api → infer-util` dependency (zero references) were deleted.
L2/L3 helpers (`evict_slot_page`/`reinstate_slot_page`) are unchanged.

Files: `infer-seam/src/{allocator.rs,kv.rs,kv_batch.rs,lib.rs}`,
`infer-hip/src/kv_pool.rs`, `infer-metal/src/executor.rs`,
`infer-vulkan/src/{executor.rs,kv_pool.rs}`, `infer-api/Cargo.toml`.

Why behaviour is preserved: a dead trait is deleted and its two methods are
inlined onto the allocator the executor already held; allocation/truncation
bodies are byte-identical moves. Gate: the existing allocator suites compile
and pass — infer-core 18 + infer-seam 8 tests, plus the infer-metal executor
and infer-vulkan executor tests whose imports the PR repaired.

### #307 — remaining plain `dead_code` allows adjudicated
Commit `a2457c5a1` ("chore: adjudicate remaining plain dead_code allows").
The genuinely-dead items were deleted (`cli` `test_env_lock`, train
`TEST_LR`); feature-only allows became `cfg` gates (kv-gds Linux helpers,
autograd qwen36/test_backend imports); structural allows were documented
with a one-line reason (serde wire fields, the train shared test-util
module). The only serving-crate file touched was
`infer-server/src/anthropic.rs`: `budget_tokens` was already read by the
coordinator, so its stale `#[allow(dead_code)]` was removed and the doc
comment corrected; the `metadata` field keeps its allow because it is a
wire field that must deserialize but is never read.

Why behaviour is preserved: attribute/comment edits and deletions of items
the compiler proved unreferenced; serde field names and parse behaviour are
untouched. Gate: `cargo clippy --all-targets -D warnings` across the
affected crates; request deserialization tests cover `anthropic.rs`.

### #341 — `infer-metal` visibility narrowed to `pub(crate)`
Commit `8558283b1` ("refactor(metal): narrow infer-metal internal visibility
to pub(crate)"). Items with no caller outside the crate were `pub`, which
also hid them from the dead-code lint; each is now the tightest scope the
compiler accepts (`pub(crate)` or private). Two crate-root re-exports with
no external use were removed (`MetalInflight`, `LoadedMetalDeepseekOcr`);
the OCR struct itself stays `pub` because `src/backends/metal.rs`
destructures its fields. `mlx-sys` had no narrowable surface.

Files: `infer-metal/src/{dflash.rs,executor.rs,kv_ssd.rs,lib.rs,mlx.rs,
resource.rs,slot.rs}` (7 files, +34/−27).

Why behaviour is preserved: visibility changes cannot alter runtime
behaviour; nothing was deleted (clippy reported no newly-dead item). Gate:
clippy -D warnings on the metal build, the no-metal feature stub, and the
root metal build.

## #289 — load-time capability gates collected into one registry

Commit `b02cf7d29` ("refactor(api): declare load-time backend capabilities
in one registry"). This is kept out of Group 1 because it governs when a
model load is refused.

Before: the MTP and `--kv-disk` load gates were copied textually into four
backend builders (`src/backends/{cpu,hip,vulkan,metal}.rs`). MTP was refused
by cpu/hip/vulkan/metal with "only supported by the CUDA backend". `--kv-disk`
was refused by cpu/hip/vulkan ("has no KV tier store"); the Metal builder
had **no** kv-ssd gate at all — the generic Metal executor spills to an L3
page tier, and a `--kv-disk` request to Metal was allowed through with no
builder check. A new backend opted into a feature simply by not copying the
check.

After: each backend declares a `BackendCapabilities { mtp_spec_decode,
kv_ssd_tier }` once at `register_backend`, and one
`check_load_capabilities(backend, caps, config)` runs at both load dispatch
points in `infer-api/src/loaded.rs` before any executor is built. Declared:
cuda `{true, true}`, metal `{false, true}`, hip/vulkan/cpu `NONE`. The
model-kind-conditional DeepSeek-OCR VLM tier limit stays inside the Metal
builder (it depends on the resolved model, not the backend). Runtime
per-step capabilities are untouched (they remain `BackendExecutor`
accessors).

Allow/refuse deltas, configuration by configuration:

| request | before | after | flip? |
|---|---|---|---|
| MTP on cuda | allowed | allowed | no |
| MTP on cpu/hip/vulkan/metal | refused | refused | no |
| `--kv-disk` on cuda | allowed | allowed | no |
| `--kv-disk` on cpu/hip/vulkan | refused | refused | no |
| `--kv-disk` on metal (generic executor) | unchecked, allowed through | declared supported, allowed | no (formalized) |
| `--kv-disk` on metal DeepSeek-OCR VLM | builder model-kind limit | same builder limit | no |
| neither feature, any backend | allowed | allowed | no |

No configuration flipped from allowed to refused or from refused to
allowed; the Metal kv-ssd path was already functional and is now declared
rather than silently ungated. The only externally visible difference is
the error text: a refusal names the selected backend instead of hard-coding
"the CUDA backend".

Gate: four unit tests in
`infer-api/src/loaded.rs::capability_tests` —
`none_requested_passes_for_any_backend`,
`mtp_gated_by_declared_capability`,
`kv_ssd_gated_by_declared_capability`,
`rejection_names_the_backend_not_a_specific_other_backend`.

## #252 — `Box<dyn Any>` inflight + runtime-dyn dispatch: per-step cost audited

Commit `ff8f3b9ce` ("refactor(seam): object-safe BackendExecutor,
de-genericize Engine"), PR #252. The backend changed from a compile-time
type parameter to `Box<dyn BackendExecutor>`, and the inflight handle from
an associated type `E::Inflight` to `Box<dyn Any + Send>`. Both touch the
submit/poll path, so this one was measured rather than grouped.

Static audit (file:line at #252 head):

- New per-step heap allocation: yes, one. Every backend's `submit` now
  returns a boxed handle — Metal
  (`crates/infer-metal/src/executor.rs:394`) does
  `Ok(Box::new(MetalInflight::Ready(…)))` on the synchronous placeholder
  path, and the real executor boxes at
  `crates/infer-metal/src/executor.rs:399`
  (`.map(|i| Box::new(i) as Box<dyn Any + Send>)`). Before #252
  `MetalInflight` was returned by value (`type Inflight = MetalInflight`).
  One malloc per scheduler step.
- New indirect calls: two per step. `Engine::step` calls
  `self.executor.submit(...)` and later `self.executor.poll(...)` through a
  trait object (`crates/infer-core/src/lib.rs:895`, submit site); before,
  both were statically monomorphized calls. `poll` additionally does one
  `Any::downcast` on the handle (Metal
  `executor.rs:407`), a type-pointer comparison with no allocation.
- The `Box<dyn KvPool>` is passed by reference into `submit`
  (`&mut *self.kv`); it adds no allocation, and kv calls on the hot path
  were already virtual through the old `KvPool` generic in the same shape.
- Nothing per-request beyond the per-step box: the handle is created once
  in `submit`, moved into `Engine.inflight`, and consumed once in `poll`;
  there is no per-token allocation (the handle holds the whole step's
  `StepOutput`).

c=1 decode is GPU/kernel-bound (per-step host work is tens of microseconds
against a ~44 ms TPOT on the 4B), so a wash was expected; the matched A/B
confirms it.

### Matched Metal c=1 A/B (2026-09-11)

Binaries built in an isolated target dir (`CARGO_TARGET_DIR=/tmp/ab252-target`,
`--release --features metal,no-cuda,cli`) — the shared lane target/ poisoned
an earlier build attempt with mixed-revision artifacts, per the shared-target
mtime rule. Baseline `e650b104f` (#252's merge-base; the single-commit
squash point `ff8f3b9ce` does not compile without its PR siblings, so the PR
head `66508ae69` is compared against the merge-base):
base sha256 `24cb7c90…`, head sha256 `a3bc77b3…`.

Model `mlx-community/Qwen3.5-4B-MLX-4bit` (same hybrid family/kernel geometry
as the canonical 35B; the 35B needs ~38 GiB on this box and was also deferred
in #332), P=543/N=128/K=6, alternating B,L,L,B single-user serves
(`--max-running-requests 1 --system-reserve-bytes 1GiB
--memory-budget-bytes 8GiB --allow-swap`, `scripts/bench_local_metal.py`):

| order | arm | TPOT ms | decode tok/s | TTFT ms |
|---|---|---:|---:|---:|
| 1 | base | 46.65 | 21.4 | 919.2 |
| 2 | head | 43.47 | 23.0 | 866.5 |
| 3 | head | 44.32 | 22.6 | 858.1 |
| 4 | base | 42.91 | 23.3 | 890.1 |

Median across each arm's two serves: TPOT base 44.78 → head 43.90 (**−1.98%**),
TTFT base 904.7 → head 862.3 (−4.7%). The base arm's own between-serve spread
is 8.7% (46.65 vs 42.91); both deltas are well inside the ±10% matched-A/B
envelope and smaller than the arm's run-to-run noise. A wash, as the static
audit predicted.

Correctness — needle ladder (`scripts/lever_gate.sh`,
`GATE_PROFILE=metal`, `mlx-community/Qwen3.5-0.8B-MLX-4bit`, lengths
115/300/446, 2 runs each, same protocol as #332): both arms report
`exact=2 partial=0 miss=0` at all three lengths — 6/6 exact each, identical
output. The dyn dispatch preserves generated tokens.

CUDA confirmation is pending-remote. The CUDA per-step handle is
`CudaInflight { output: StepOutput }` (already synchronous), so the only
new cost there is the same box plus dyn calls against a much longer GPU
step; the matched command is:

```
scripts/bench_throughput.py <url> <model> --seconds-per-concurrency <...>
# baseline: release binary at e650b104f, treatment: 66508ae69 (or current main)
# c=1, same model, alternating B/L/L/B; TPOT delta inside ±10% closes it
```

