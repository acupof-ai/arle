# M_quant Hybrid Phase 1b — Loader storage augmentation directive

**Trigger:** `b6502f7` Phase 1a checkpoint merge tool ready
**Owner:** codex(Rust impl)
**Effort:** ~0.5d / ~100-150 LOC
**Blocks:** Phase 2(Linear dispatch)

## §1 Why this is ready

Phase 0 reconnaissance(`1959a21`)confirmed:
- ✅ Scheduler boundary clean(`StepPlan` enum at `execution.rs:411`)
- ✅ Memory cost 45% acceptable at c≤8

Phase 1a tool ready(`b6502f7`):
- `scripts/merge_w4_hybrid_checkpoint.py` produces hybrid checkpoint
- Output naming:`marlin_qweight` + `marlin_scales`(W4A16) **AND**
  `marlin_w4a8_qweight` + `marlin_w4a8_s_channel` + `marlin_w4a8_s_group`(W4A8)
  per Linear
- `config.json` `quantization_config.quant_type = "marlin_w4_hybrid"`

Loader currently has NO awareness of hybrid format。Phase 1b adds it。

## §2 Concrete loader patches needed

### 2.1 Detection at `weight_loader.rs:514`

```rust
// BEFORE
Ok("marlin_w4a8" | "w4a8_marlin")

// AFTER
Ok("marlin_w4a8" | "w4a8_marlin" | "marlin_w4_hybrid")
```

Also extend `LoaderConfig` struct(line ~446)to add:
```rust
pub(crate) marlin_w4_hybrid: bool,
```

Plus update all `LoaderConfig` initialization sites(lines 463, 470, 483,
490, 497, 504, 520):set `marlin_w4_hybrid: true` when detected,else `false`。

When `marlin_w4_hybrid` flag is on,**both `marlin_w4a8` AND legacy
W4A16 marlin flags should also be true** so existing tensor-loading
branches engage(see §2.2)。

### 2.2 Tensor reading at `:663-715`

Current code reads either `marlin_*` (W4A16) or `marlin_w4a8_*`(W4A8)
based on flags。For hybrid,need to read BOTH:

```rust
// BEFORE (per-format branches)
if config.marlin_w4a8 {
    let packed_name = name.replace(".weight", ".marlin_w4a8_qweight");
    let channel_scales_name = name.replace(".weight", ".marlin_w4a8_s_channel");
    let group_scales_name = name.replace(".weight", ".marlin_w4a8_s_group");
    // ...
    return DeviceMatrix::from_marlin_w4a8(...);
}

// AFTER (hybrid reads both,then chooses runtime)
if config.marlin_w4_hybrid {
    // Read W4A16 tensors
    let w4a16_qweight_name = name.replace(".weight", ".marlin_qweight");
    let w4a16_scales_name = name.replace(".weight", ".marlin_scales");
    let w4a16_qweight = read(...);
    let w4a16_scales = read(...);

    // Read W4A8 tensors
    let w4a8_packed_name = name.replace(".weight", ".marlin_w4a8_qweight");
    let w4a8_channel_name = name.replace(".weight", ".marlin_w4a8_s_channel");
    let w4a8_group_name = name.replace(".weight", ".marlin_w4a8_s_group");
    let w4a8_qweight = read(...);
    let w4a8_s_channel = read(...);
    let w4a8_s_group = read(...);

    return DeviceMatrix::from_hybrid_w4_marlin(
        w4a16_qweight, w4a16_scales,
        w4a8_qweight, w4a8_s_channel, w4a8_s_group,
        ...
    );
}
```

### 2.3 New DeviceMatrix variant or fields

Two implementation options:

**Option A**(recommended,simpler):extend `DeviceMatrix` to carry
both side-tensors:
```rust
struct DeviceMatrix {
    // existing fields...

    // Hybrid extension (None for non-hybrid Linear):
    hybrid_w4a8_qweight: Option<...>,
    hybrid_w4a8_s_channel: Option<...>,
    hybrid_w4a8_s_group: Option<...>,
}
```

For non-hybrid loads,these are None。For hybrid loads,populated。
Linear dispatch(Phase 2)checks hybrid_*。is_some() to route to W4A8。

**Option B**:new `HybridLinear { w4a16: DeviceMatrix, w4a8: DeviceMatrix }`
struct with separate matrices。More structural but ~50 extra LOC and
requires `LinearWeight` enum or similar。

Phase 0 recommended **Option A**(`1959a21`)— ~50 LOC simpler。Use it。

### 2.4 New `from_hybrid_w4_marlin` constructor

Add to `crates/cuda-kernels/src/tensor.rs` or wherever `from_marlin_w4a8`
lives:
```rust
pub fn from_hybrid_w4_marlin(
    w4a16_qweight: ...,
    w4a16_scales: ...,
    w4a8_qweight: ...,
    w4a8_s_channel: ...,
    w4a8_s_group: ...,
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> Self {
    // Construct DeviceMatrix with both side-tensors populated
}
```

## §3 Validation

After Phase 1b lands:
1. **Build check**:`cargo build --release -p infer --features cuda`
2. **Loader unit test**:add test case loading hybrid checkpoint,verify
   `DeviceMatrix.hybrid_w4a8_qweight.is_some()` for Linear layers
3. **Existing tests pass**:`cargo test --release -p infer --features
   cuda --test e2e` and `--test greedy_consistency`
4. **No regression on legacy paths**:verify W4A16-only and W4A8-only
   checkpoints still load correctly(should be unchanged path,but
   verify with existing test_w4a8_vs_bf16_token_diff)

## §4 Phase 2 preview(after Phase 1b)

Linear dispatch:add `phase: PhaseHint` enum to `run_linear`,route
based on hybrid_w4a8_qweight presence + phase:
```rust
match (weight.hybrid_w4a8_qweight.is_some(), phase) {
    (true, PhaseHint::Prefill) => run_marlin_w4a8_linear(...),
    (true, PhaseHint::Decode)  => run_marlin_w4a16_linear(...),
    (false, _)                  => existing single-format path
}
```

Caller(scheduler step plan)passes hint:
- `StepPlan::Decode` → `PhaseHint::Decode`
- `StepPlan::Prefill | Mixed | Split` → `PhaseHint::Prefill`(per Phase 0
  Option A)

## §5 KILL criteria

- **Phase 1b**:if loader changes break existing W4A16 or W4A8 single-
  format tests → revert,re-evaluate Option B(separate HybridLinear)
- **Phase 2**:if hybrid greedy gate fails → dispatch logic bug,debug
- **Phase 4**:if hybrid bench shows < 5% E2E improvement vs W4A16-only →
  per master plan KILL,not worth memory cost

## §6 Cross-references

- Hybrid plan main: [`9754aca`](M_quant-w4a16-w4a8-hybrid-prefill-decode.md)
- Phase 0 reconnaissance: [`1959a21`](../research/...)
- Phase 1a merge tool: [`b6502f7`](.../merge_w4_hybrid_checkpoint.py)
- Concurrency sweep: [`8588f6a`](../experience/wins/...)
- Loader: `infer/src/weight_loader.rs:446-715`
- Linear FFI: `infer/src/ops/linear.rs:777-859`(W4A8) + W4A16 path
- Scheduler StepPlan: `infer/src/scheduler/cuda/execution.rs:411`

## §7 Status

- ✅ Phase 0 reconnaissance(`1959a21`)
- ✅ Phase 1a checkpoint merge tool(`b6502f7`)
- ⏳ **Phase 1b loader storage augmentation**(this directive)
- ⏳ Phase 2 Linear dispatch
- ⏳ Phase 3 E2E test
- ⏳ Phase 4 Bench + ship wins entry

Codex,when picked up:start with §2.1 detection patch as smallest
verification step。If it builds + existing tests pass,proceed to §2.2
tensor reading + §2.3 DeviceMatrix extension。
