# Recipe-itself audit — skill v1.8.0 anti-pattern #21 candidate

> Per `43b2115` decision to defer skill v1.8.0 codification(5 anti-patterns
> added today already,avoid noise),this brief pre-stages anti-pattern #21
> alongside the already-deferred #20。Both fire together when v1.8.0
> triggers。
>
> **Evidence**:`b55bfcd` caught a real Rust scoping bug in my `2fafa9e`
> Phase 1.A recipe within ~5 min of attempted application,saving tomorrow's
> Phase 1.A pickup from compilation failure。

## Anti-pattern definition

**Name**:Recipe-itself audit gap

**Definition**:Recipe-style briefs(copy-paste-ready code diffs / shell
commands / step-by-step procedures)inherit the SAME hypothesis-vs-evidence
trap as any other prescription:**writing a recipe is NOT the same as having
a working recipe**。

A recipe-style brief that hasn't been **dry-run audited or actually applied**
is a hypothesis,not evidence,regardless of how authoritative the prose
sounds。

**Skill rule**:after writing a recipe-style brief,run an audit pass
specifically targeting:
1. **Syntax correctness**(does the diff compile?does the shell command parse?)
2. **Scoping correctness**(do bindings live where later code references them?
   are guards / closures dropped at the right time?)
3. **Tool / file existence**(does the script the recipe calls actually exist?
   with the expected interface?)
4. **Side effects**(does any temporary instrumentation contaminate state?
   require revert?)

## Concrete evidence — `b55bfcd`(2026-05-09 EOD+86)

### Bug pattern

My `2fafa9e` Phase 1.A nvtx scope diff:
```rust
// Phase 1.A recipe (incorrect):
{
    nvtx_scope!("step_admission_prefix_lookup");
    let mut lookup = if ... { ... } else { ... };
}
// `lookup` out of scope here, but used at lines 193+ → COMPILATION FAILURE
```

### Catch + correction

`b55bfcd` block-as-rvalue fix:
```rust
// Correct pattern:
let mut lookup = {
    nvtx_scope!("step_admission_prefix_lookup");
    if ... { ... } else { ... }  // tail expression
};
// `lookup` lives in outer scope, _nvtx_scope guard dropped at `};`
```

### Why missed

When writing the recipe,I read `admission.rs:181-205` to identify the
block boundaries,but **didn't trace where `lookup` is used DOWNSTREAM**。
A simple grep `grep -n 'lookup\.' admission.rs:200-300` would have caught
the binding-survives-block requirement。

### Cost / saved

- Attempted application time:~2 min(diff syntax check + grep downstream uses)
- Compilation failure time would have been:~5-10 min(cargo build cycle +
  error trace + revert)
- Net savings:~3-8 min per occurrence,but more importantly **pickup
  worker confidence**(applying a "verified ready" recipe vs hitting
  unexpected breakage)

## Generalization beyond Rust scoping

The same trap applies to ALL recipe types:

### Shell command recipes(e.g. `nsys profile ... --capture-range=none ...`)

**Audit checklist**:
- Run `<cmd> --help` to verify all flags exist
- Run `<cmd> --dry-run` if available
- For destructive commands,run with `--noop` first
- Verify file paths the command reads/writes exist or are creatable

### Bench script recipes(e.g. `python3 scripts/bench_X.py http://localhost:8000 model`)

**Audit checklist**:
- `ls -la <script>` confirms existence
- `head <script>` reads docstring for arg semantics
- `grep -E '^def main|sys.argv|argparse' <script>` confirms arg parsing
- Verify expected interface matches recipe usage

### Config / flag recipes(e.g. `--admission-policy=prefix-aware --cold-headroom=N`)

**Audit checklist**:
- `grep -E "argument|flag|option" <main.rs / impl_args>` confirms flag is
  actually parsed
- Run binary with `--help` to see actual flag listing
- Verify default values match recipe assumptions

## Bidirectional audit cycle codification

Today's session demonstrated **bidirectional methodology** at scale:

1. `1fdd763` Phase 0 audit(audit code claim)
2. `c076aae` audit-of-audit(audit the audit's hypothesis chain)
3. `8b1a913` adopt gates(integrate finding back to plans)
4. `3456f8f` cheap-experiment recipe(prescribe verification)
5. `43b2115` cite recipe(integrate prescription back to plans)
6. `d2c2c17` strategic ROI(audit prescription's effectiveness vs goal)
7. `9e964c9` P0.0 priority(integrate strategic finding)
8. `ec5c37c` LICENSE staleness gate(audit the empirical evidence's currency)
9. `183bd60` annotate gate(integrate gate)
10. `b85929b` codex re-bench LANDS(close 1-9 cycle with verified code)
11. `b55bfcd` recipe-scoping audit(THIS antipattern's evidence — audit recipe itself)
12. **(this brief)** pre-stage codification

→ **11 commits,bidirectional rigor at every layer**:
- Audit claims about code → audit hypothesis chains → audit recipes → audit
  empirical evidence currency → audit recipes' application
- Each layer has a §0 SOLID gate; none ships unverified

This is the methodology to **codify in skill v1.8.0**。Not "write more docs"
— it's "audit at every prescription layer including recipes themselves"。

## Skill v1.8.0 batch — proposed contents

When triggered(per `43b2115` deferral rule:wait for next surprise lesson
to justify version bump):

| # | Anti-pattern | Pre-staged evidence |
|---|--------------|---------------------|
| #20 | Phase 0 root-cause hypothesis inheritance | `c076aae` SOLID gap |
| #21 | Recipe-itself audit gap | `b55bfcd` block-wrap fix |
| #22? | (pending next surprise) | (TBD) |

Bumping for both #20 + #21 makes a coherent v1.8.0 batch about
**audit-at-every-prescription-layer**。Single coherent theme = better
skill memorability than scattered single-anti-pattern bumps。

## §0 SOLID first principle in action

This brief = §0 applied to the methodology itself:
- **推断 ≠ evidence**:writing a recipe is recipes — verifying it works is
  evidence
- **混淆变量必须隔离**:recipe form(prose+diff)vs recipe content
  (compilable code)— different layers,different audit needs
- **Root cause 假设也要 license-or-kill**:"recipe is correct because
  prose is correct" is hypothesis;cargo build is the kill criterion
- **80% SOLID 不够**:my 2fafa9e was 80% SOLID(syntax,scope name,
  insertion site verified)but missed the 20%(downstream `lookup` uses
  invalidate the block-wrap)

## Cross-references

- `2fafa9e` original Phase 1.A recipe(had bug)
- `b55bfcd` recipe scoping fix(EOD+86)
- `nvtx_scopes.rs:29-37` — macro creates `_nvtx_scope` guard at current
  scope
- Skill v1.7.0 latest(`c768b70`)
- v1.8.0 deferral decision(`43b2115`)
- §0 first principle(CLAUDE.md "求真务实,追求极致")

## Status

Anti-pattern #21 candidate codified + pre-staged。Triggers on next skill
v1.8.0 bump。Coherent batch with #20 around "audit-at-every-prescription
-layer" theme。

This brief itself is not exempt from §0 — it's a **claim about methodology**
that needs evidence。The evidence is `b55bfcd`(real bug caught,real cost
saved)。Without that,this would be hypothesis。With it,skill rule is
SOLID-grounded。
