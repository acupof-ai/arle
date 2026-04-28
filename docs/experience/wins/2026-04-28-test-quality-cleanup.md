# Test Quality Cleanup — 2026-04-28

## Context

Goal: review the workspace test population, delete shallow / tautological
tests that don't exercise full call-chain output, and add coverage for
error-prone scenarios that were gaps. Triggered by the user's directive
to identify "AI coding 不需要的测试" (AI-noise unit tests) and replace
them with hostile-input / boundary / failure-mode tests.

Scope: workspace-wide (`infer/src/`, `crates/*/src/`, `crates/*/tests/`,
root `tests/`). ~1255 tests at start.

## What Worked

### 1. Audit-first, delete-second

Wrote `docs/reviews/2026-04-28-test-quality-audit.md` (544 lines)
classifying every `#[test]` into A/B/C/D/E/F/G/U + EXEMPT before
touching any code. This was the single highest-leverage decision:

- The first-pass rough estimate (485 deletion candidates) was off by
  **27×**. The careful body-by-body audit landed at 9 A + 6 B + 0 C +
  3 D = 18 candidates, of which 7 were further demoted to EXEMPT
  during execution (NCCL FFI ABI pins, public layout strings).
- The audit's bias toward **false-negative on delete** (when in doubt,
  keep) prevented contract-test loss. Specifically: `BlockFingerprint`
  persistence stability, `UNIQUE_ID_BYTES` rendezvous wire size, and
  `WeightFormat::kernel_alignment` layout strings would all have been
  deleted by a less careful pass.

### 2. Codex review caught two audit-quality bugs early

Running `codex review --commit <audit-doc-commit>` flagged:
- `unique_id_size_constant` was misclassified A — actually a wire
  contract (NCCL `ncclUniqueId` 128-byte size).
- The audit inventory was generated against a dirty working tree, so
  the root `tests/` directory and a couple of file-level counts were
  off.

Both findings were P2 but high signal. The `--commit` form (review one
specific commit) was the right shape because the diff contained a
self-contained, reasoned document.

### 3. Surgical deletion + EXEMPT discipline shrank the blast radius

Net change after Phase 2: **10 deletions + 1 refactor** across
`infer/src/{types,kv_tier/tier,scheduler/policy,quant,model_registry}.rs`
and `crates/{autograd/src/optim,train/src/cli_args}.rs` (88 LOC removed,
8 LOC added for the AdamW `is_device_backed()` getter that replaced
the brittle `format!("{optim:?}").contains("device_backed: true")`
assertion).

Each phase was its own small commit + cargo-test-on-affected-crate +
async codex review. No phase touched > 5 files.

### 4. Phase 3 added 9 failure-mode tests for known-weak surfaces

| File | New tests | Surface |
|------|-----------|---------|
| `infer/src/block_manager.rs` | 1 | OOM rejection + cross-pool isolation + recovery |
| `infer/src/sampler.rs` | 5 | all-equal logits, single-vocab, count-vector mismatch, empty inputs, top_p=0 ≠ greedy |
| `infer/src/tokenizer.rs` | 3 (ignored without weights) | invalid UTF-8 via lossy, ChatML injection envelope-blindness, ZWJ/long-grapheme |

The tokenizer tests are gated `#[ignore = "requires model weights"]`
to match the existing pattern in that file; they pass when run with
the Qwen3 fixture present.

No mocks. No new dev-deps. No new public API surface (the one new
getter `AdamW::is_device_backed` replaced a Debug-string lock and is
already used by callers via `step()`).

### 5. AI-noise test patterns (concrete catalog)

For future audits in this codebase:

- `assert_eq!(struct.field, value_just_set)` — Class A
- Pure derive round-trips (`Copy`, `Hash`, `Eq`, `Ord`) — Class A
- `assert_eq!(literal_array.len(), N)` — Class A
- `Foo::new(...); assert!(...)` testing only ctor inputs — Class B
- `format!("{x:?}").contains("...")` for non-public-API debug — Class D
- `mem::size_of::<EnumWithReprI32>()` — depends:
  - **EXEMPT** if `<Enum>` crosses an FFI boundary (NCCL, CUDA driver, etc.)
  - **Class A** if it's an internal-only enum

Patterns that LOOK like Class A but are actually contracts:
- `BlockFingerprint::compute_*` — persistence stability (KV-tier on disk)
- `UNIQUE_ID_BYTES == 128` — wire ABI
- `pub WeightKernelAlignment.weight_layout` literal strings — kernel-dispatch routing
- ChatML role-token byte sequences in `crates/chat/src/protocol.rs`
- Tensor-name strings in `crates/qwen{3,35}-spec/`

## Rule

**Audit before deletion; classify by reading bodies, not names.** A
careful body-by-body audit will reduce the "obvious deletion list" by
roughly an order of magnitude. The right metric is *information
density per test*, not *test count per file*.

**Bias toward keep on ambiguity.** Class U exists for a reason. The
codebase has many "tests that look trivial but are actually contract
pins for FFI / wire / persistence / public-API surfaces"; these are
the smoke-list 11 tests in the audit that, if accidentally deleted,
would cost the most.

**Phase test deletions A → B → D → C (cheapest-to-revert first).** A
is pure-derive removals (no behavior loss); B is constructor-coverage;
D requires a refactor (replace Debug-string with a getter); C requires
a real-backend / real-axum replacement, which is why we found zero
candidates here in the first place — the codebase already mocks
narrow boundaries while testing real internals.

**Add tests where the existing population is thinnest, not where it's
already thick.** `prefix_cache.rs` (55 tests) and `http_server/tests.rs`
(54) don't need more coverage; `block_manager.rs` and `sampler.rs`
edge cases (vocab=1, all-equal logits, count-vector mismatch) did.

## Verify

- `cargo test --release -p infer --lib --no-default-features --features metal`
  → **503 passed / 0 failed / 24 ignored** (Phase 3 commit `f4e3a71`)
- `cargo test --release -p autograd --lib` → **5 passed / 0 failed / 0 ignored**
- `cargo test --release -p train --lib --no-default-features --features metal`
  → **75 passed / 0 failed / 1 ignored**

Bench-exempt: test-only changes; no runtime hot path edits. Per
`CLAUDE.md` Verify-phase rule, this is in the docs/test/comment-only
exemption set, not the in-scope set that requires a `bench_guidellm`
entry.

## Cross-links

- Audit doc: [`docs/reviews/2026-04-28-test-quality-audit.md`](../../reviews/2026-04-28-test-quality-audit.md)
- Phase 1 commit (audit doc): `3757be1`
- Phase 2A+B commit (10 deletions): `e67c905`
- Phase 2D commit (AdamW getter): `69b8429`
- Audit doc addendum (codex corrections): `526012e`
- Phase 3 commit (9 scenario tests): `f4e3a71`
