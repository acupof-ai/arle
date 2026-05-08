# Phase 1.A recipe — scoping fix for nvtx wrap

> Per `2fafa9e` Phase 1.A recipe attempt at applying the 5-LOC nvtx scope
> diff,Claude found a scoping issue:`lookup` variable is used at lines
> 213/219/225/227/228/237/241/243+ AFTER the proposed scope close。
> Block-wrap as written would put `lookup` out of scope。This brief
> records the correct minimal pattern for tomorrow's pickup。

## Issue

`2fafa9e` recipe wraps prefix lookup body 181-220 in a block:
```rust
{
    use crate::scheduler::cuda::nvtx_scopes::nvtx_scope;
    nvtx_scope!("step_admission_prefix_lookup");
    let mut lookup = if let Some(session_id) = session_id
        && let Some(session_lookup) = ...
    {
        session_slot_hold = Some(session_lookup.hold);
        session_lookup.lookup
    } else {
        self.prefix_cache.lookup_or_stage(prompt_tokens, heuristics)
    };
    // ... session_resume_tokens follow-up logic
}
```

Variable scoping problem:
- `lookup` declared inside block → **NOT visible** to outer code at lines 213+
- Lines 213(`lookup.blocks`),219(`matched_sealed_lookup_blocks(&lookup.blocks)`),
  220(`lookup.matched_len`),etc. all need `lookup` in outer scope
- Block-wrap as written would **break compilation**

## Correct minimal pattern

Use **block-as-rvalue**:
```rust
let mut session_slot_hold = None;
let mut lookup = {
    use crate::scheduler::cuda::nvtx_scopes::nvtx_scope;
    nvtx_scope!("step_admission_prefix_lookup");
    if let Some(session_id) = session_id
        && let Some(session_lookup) =
            self.lookup_session_slot_or_stage(session_id, prompt_tokens.len(), heuristics)
    {
        session_slot_hold = Some(session_lookup.hold);
        session_lookup.lookup
    } else {
        self.prefix_cache.lookup_or_stage(prompt_tokens, heuristics)
    }
};
// _nvtx_scope drops here when block exits, range_pop() called
// `lookup` is now in outer scope, visible at lines 213+
```

Differences from `2fafa9e` recipe:
- `let mut lookup = { ... }` instead of `{ let mut lookup = ...; }`
- Inner block returns the value via tail expression(no `;` after the if-else)
- Scope coverage:lines 182-190 only(initial `lookup_or_stage` call)
- **Session prefix matching at lines 191-209 is NOT in this scope** —
  considered a separate concern。Could add a second `nvtx_scope!("step_admission_session_match")`
  if needed,but Phase 1.A's goal is "prefix::lookup" specifically(not
  "all admission logic")

## Why minimal scope is correct for Phase 1.A goal

Per `d2c2c17` strategic brief,Phase 1.A targets 4-phase decomposition:
- prefix::lookup ← `step_admission_prefix_lookup`(this scope)
- prefill::compute ← existing `step_prefill_kernel_launch`
- first_decode::compute ← existing `step_decode_kernel_launch`
- scheduling::overhead ← `step_total - step_admission - prefill - decode`

`step_admission` already covers session prefix matching + queue manipulation。
The missing piece is just the **prefix lookup itself**(which is line 182-190
in admission.rs)。Minimal scope is correct;wider scope would conflate phases。

## Phase 1.A actual LOC

`2fafa9e` estimated "~5 lines wrap"。Correct minimal:**~3 lines**:
- `use crate::scheduler::cuda::nvtx_scopes::nvtx_scope;`(or full-path call)
- `nvtx_scope!("step_admission_prefix_lookup");`
- `let mut lookup = { ... };` re-formatting(0 net LOC,just punctuation)

→ Phase 1.A scope-add LOC:**3 net lines**(within original 5 estimate)。

## Validation

After applying minimal scope diff:
1. `cargo check --release -p infer --features cuda` should pass(unchanged behavior)
2. Behavior preservation:`cargo test --release -p infer --features cuda scheduler::`
   should still report 182 PASS(no functional change,only NVTX instrumentation)
3. nsys output should now show `step_admission_prefix_lookup` scope distinct
   from `step_admission`(parent)

## Why this matters

`2fafa9e` recipe is otherwise excellent — concrete commands,decision matrix,
SOLID gates。Just the variable-scoping wrap pattern needed correction。Tomorrow's
codex/Claude pickup can apply minimal pattern above without compilation-error
detour。

§0 SOLID rule:**audit recipes themselves before adopting**。Codex's recipe
caught my earlier P0.3 hypothesis-inheritance gap;this brief catches codex's
recipe scoping gap。Bidirectional audit cycle continues — neither side ships
unverified prescriptions。

## Cross-references

- Recipe being fixed:`2fafa9e` `2026-05-09-eod85-p0.0-phase1a-nvtx-decomposition-recipe.md`
- Strategic priority:`9e964c9` P0.0 Phase 1 evidence decomposition
- Architectural source:`d2c2c17` post-B3 strategic next-axis ROI
- B3 Step 2 LANDED:`b85929b` feat(scheduler): wire prefix-aware CUDA admission gate
- nvtx_scope macro:`infer/src/scheduler/cuda/nvtx_scopes.rs:29-35`
- admission.rs prefix lookup site:`infer/src/scheduler/cuda/runtime/admission.rs:182-190`(post-`b85929b`)

## Status

Recipe scoping fix recorded。Tomorrow's pickup applies the **block-as-rvalue**
pattern,not the original block-wrap pattern。3 net LOC change(within
`2fafa9e` 5-LOC estimate)。

This is the **10th commit in tonight's bidirectional audit cycle**,extending
the chain into Phase 1.A territory。
