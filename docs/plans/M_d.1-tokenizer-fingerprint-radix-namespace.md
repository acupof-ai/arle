# M_d.1 — Tokenizer fingerprint as RadixCache namespace

> Closes the silent-corruption hole documented in
> [`2026-05-07-radix-cache-no-tokenizer-fingerprint.md`](../experience/errors/2026-05-07-radix-cache-no-tokenizer-fingerprint.md).
> Refines the original 5-step sketch down to **3 steps** after a P0
> survey of the actual surface — Steps 2 (NFC) and 3 (chat-template fp)
> are unnecessary in ARLE for the reasons noted below.

## P0 survey findings (2026-05-07; refined post codex review)

First-pass `grep` + targeted reads of `infer/src/tokenizer.rs`,
`infer/src/prefix_cache.rs`, `infer/src/http_server/`, and
`crates/chat/src/` exposed three simplifications. Codex review of
HEAD `78833f7` then surfaced two additional production-surface
corrections (folded into Steps 2 + 3 below); the sketch is now an
accurate map of every cache instance that needs the namespace.

| Original sketch step | P0 finding | Decision |
|---|---|---|
| 1. SHA-256 fingerprint at `Tokenizer::from_file` | `Tokenizer { inner: HfTokenizer }` has no fingerprint field; `from_file` reads `tokenizer.json` raw | **Keep — implement** |
| 2. NFC normalization at `Tokenizer::encode` entry | `tokenizer.rs:322-323` already documents that the HF tokenizer's pre-tokenizer applies NFC; double-normalize is a regression risk | **Drop — already done by HF config** |
| 3. Chat-template version into namespace | `crates/chat/src/protocol.rs` hardcodes ChatML; ARLE does NOT read model `chat_template` JSON. Template version = `crates/chat` source revision = compile-time constant = already namespaced by deploy | **Drop — implicit via build** |
| 4. Block token-segment composition | `RadixCache::lookup_or_stage` walks one prefix; no cross-request token-segment composition exists in tree | **Already holds — keep documented as invariant** |
| 5. Fingerprint as RadixCache namespace prefix | `RadixCache` has TWO instantiation sites: CUDA `scheduler/cuda/core.rs` and Metal `backend/metal/prefix_cache.rs:38`. Threading per-call would touch every `lookup_or_stage` call site (>10) | **Refine — store namespace on `RadixCache` itself, not per-call key** |

Net result: original 5-step sketch → **3 actionable steps + 1 test**,
with the namespace becoming an instance-level constant rather than a
per-call key prefix. This is structurally cleaner: one RadixCache
instance per `(tokenizer_fp, build_revision)` pair; mid-run tokenizer
swap is a server restart, not a runtime invalidation.

## Design

### 1. `Tokenizer::fingerprint() -> [u8; 32]`

`infer/src/tokenizer.rs`:

```rust
pub struct Tokenizer {
    inner: HfTokenizer,
    fingerprint: [u8; 32],
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let path = Path::new(path);
        let tokenizer_path = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let bytes = std::fs::read(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to read tokenizer file: {}", e))?;
        let fingerprint = sha2::Sha256::digest(&bytes).into();
        let inner = HfTokenizer::from_bytes(&bytes)
            .map_err(|e| anyhow!("Failed to parse tokenizer: {}", e))?;
        Ok(Self { inner, fingerprint })
    }

    pub fn fingerprint(&self) -> &[u8; 32] { &self.fingerprint }
}
```

Cost: pulls `sha2` into `infer/Cargo.toml` (already a transitive dep
via `cudarc` and `tokenizers`). One `std::fs::read` of `tokenizer.json`
(typically ~7 MB for Qwen3) at boot — adds <50 ms to startup.

### 2. `RadixCache` namespace field — close the public-derive bypass

`infer/src/prefix_cache.rs`:

```rust
// Remove `#[derive(Serialize, Deserialize)]` from `RadixCache` —
// public derive is a bypass: any caller can do
// `serde_json::from_str::<RadixCache>(s)` and skip the namespace
// check. Replace with custom impls.

pub struct RadixCache {
    nodes: Vec<Node>,
    free_nodes: Vec<usize>,
    block_size: usize,
    clock: u64,
    soft_pin_keepalive_ticks: Option<u64>,
    block_index: HashMap<BlockId, usize>,
    /// Compile-time + tokenizer namespace. Two RadixCache instances
    /// with different namespaces MUST NOT be merged or share blocks.
    /// On disk snapshots include this; loading rejects mismatch.
    namespace: [u8; 32],
}

impl RadixCache {
    pub fn new(block_size: usize) -> Self {
        Self::new_with_namespace(block_size, [0; 32])  // legacy/test default
    }

    pub fn new_with_namespace(block_size: usize, namespace: [u8; 32]) -> Self { ... }

    pub fn with_soft_pin_keepalive_namespaced(
        block_size: usize,
        soft_pin_keepalive_ticks: u64,
        namespace: [u8; 32],
    ) -> Self { ... }

    pub fn namespace(&self) -> &[u8; 32] { &self.namespace }

    /// THE ONLY snapshot-load API. Custom-implements serde so the
    /// derive can be removed; takes the expected namespace and rejects
    /// any snapshot whose stored namespace differs.
    pub fn load_snapshot(json: &str, expected_namespace: &[u8; 32])
        -> Result<Self> { ... }

    /// Snapshot serialization stays public via custom Serialize impl
    /// (writes `namespace` field). No restore-side bypass remains.
    pub fn save_snapshot(&self) -> String { ... }
}
```

The four existing `serde_json::from_str::<RadixCache>` call sites in
`infer/src/prefix_cache/tests.rs` (lines 800, 837, 861, 910) MUST migrate
to `RadixCache::load_snapshot(&json, &expected_namespace)`. After the
migration `RadixCache: !Deserialize` is the type-system guarantee that
no future code can bypass the namespace check.

### 3. Server boot wiring — corrected to actual call sites

Codex review (HEAD `78833f7` → review feedback) corrected the original
draft: the production cache surfaces are NOT what was named.

| Surface | Actual call site | Action |
|---|---|---|
| CUDA scheduler RadixCache | `infer/src/scheduler/cuda/core/construction.rs:269` — `RadixCache::with_soft_pin_keepalive(...)` | Swap to `with_soft_pin_keepalive_namespaced(..., derive_namespace(&tokenizer))` |
| Metal RadixCache (Qwen3 path) | `infer/src/backend/metal/prefix_cache.rs:38` — `RadixCache::new(block_size)` | Swap to `RadixCache::new_with_namespace(block_size, derive_namespace(&tokenizer))` |
| Metal Qwen3.5 prefix runtime | `infer/src/backend/metal/runtime.rs:416-418` — `MetalQwen35PrefixRuntime { entries: HashMap<Vec<u32>, _>, disk_entries: HashMap<Vec<u32>, _> }` — independent cache, NOT a `RadixCache` instance | Add `namespace: [u8; 32]` field; SSD disk format includes namespace header; `reconcile_disk_entries` rejects on-disk entries whose namespace ≠ runtime namespace; `entries` lookup is in-memory only and inherits namespace via instance ownership |

`derive_namespace` = `sha256(tokenizer.fingerprint() ++ env!("CARGO_PKG_VERSION") ++ build_git_sha)`.
The same helper feeds all three surfaces — single source of truth.

Server boot logs the namespace at INFO once per active backend:
`"prefix-cache namespace: surface=<cuda|metal-qwen3|metal-qwen35> tokenizer=<hex16> build=<hex8>"`.
Operators who hot-swap `tokenizer.json` will see the namespace change
after restart and know the cache cleared. Silent swap mid-run is not
defended against (the loaded `Tokenizer` instance keeps its old
fingerprint until restart) — that's the operator-policy half of the
contract, not code.

### Test (T1) — namespace isolation

`infer/tests/tokenizer_fingerprint_radix_isolation.rs`:

1. Create RadixCache A with namespace `[0xAA; 32]`, insert blocks for
   token sequence `[1, 2, 3, 4]`.
2. `let json = A.save_snapshot();`
3. `assert!(RadixCache::load_snapshot(&json, &[0xBB; 32]).is_err())` —
   wrong-namespace load rejected.
4. `let restored = RadixCache::load_snapshot(&json, &[0xAA; 32])?;` —
   matching-namespace load succeeds and returns the original cache.
5. Create RadixCache C fresh with namespace `[0xBB; 32]`, lookup
   `[1, 2, 3, 4]`, assert MISS (no shared state with A — different
   instance, different namespace).
6. **Compile-time guard test**: a `compile_fail` doctest or `trybuild`
   case asserts `serde_json::from_str::<RadixCache>(_)` no longer
   compiles after the derive removal. This is the structural guarantee
   that the bypass cannot regress.
7. Metal Qwen3.5 SSD case lives as a `#[cfg(feature = "metal")] mod tests`
   inline in `infer/src/backend/metal/runtime.rs` (NOT in
   `infer/tests/`) — `MetalQwen35PrefixRuntime` is private + cfg-gated
   to `metal`, so it can only be tested intra-module. Test:
   write a fake disk entry with namespace header `[0xAA; 32]`,
   instantiate `MetalQwen35PrefixRuntime` with namespace `[0xBB; 32]`,
   call `reconcile_disk_entries`, assert the `[0xAA]` disk entry is
   rejected (logged + dropped) and not surfaced in `disk_entries`.
   Runs only under `cargo test --release --no-default-features --features metal`.

The default-feature CPU-only portion (steps 1-6) lives in
`infer/tests/tokenizer_fingerprint_radix_isolation.rs` and runs in
`cargo test --release` with no extra features (~30 ms). The Metal SSD
test (step 7) gates on the `metal` feature and is exercised on
Apple-Silicon CI.

## Tasks

| # | Task | File | LOC est. | Owner |
|---|---|---|---|---|
| 1 | Add `fingerprint: [u8; 32]` to `Tokenizer`, compute in `from_file`, expose `fingerprint()` | `infer/src/tokenizer.rs` | ~15 | Claude |
| 2 | Add `namespace` to `RadixCache`; remove public `#[derive(Serialize, Deserialize)]`; custom `save_snapshot` / `load_snapshot(json, &expected_namespace)` API; add `with_soft_pin_keepalive_namespaced` constructor | `infer/src/prefix_cache.rs` | ~80 | Claude |
| 2b | Migrate **5** in-tree `serde_json::from_str::<RadixCache>` (and the type-annotated form on `:198`) call sites to `load_snapshot` so the type-system bypass-guard actually compiles | `infer/src/prefix_cache/tests.rs` (lines 198, 800, 837, 861, 910) | ~25 | Claude |
| 3 | Server boot wiring — CUDA `with_soft_pin_keepalive_namespaced` + Metal Qwen3 `new_with_namespace` + Metal Qwen3.5 runtime namespace field | `scheduler/cuda/core/construction.rs:269`, `backend/metal/prefix_cache.rs:38`, `backend/metal/runtime.rs:416-554` (incl. `reconcile_disk_entries` mismatch reject), `main.rs` boot log | ~70 | Claude |
| 4 | Test: namespace isolation (in-memory + snapshot-mismatch + SSD-mismatch + compile-fail bypass-guard) | `infer/tests/tokenizer_fingerprint_radix_isolation.rs` (new) | ~150 | Claude |

**Total: ~335 LOC, ~1 day.** Larger than the first draft because
codex review caught two missed surfaces (Metal Qwen3.5 runtime + the
public derive bypass); both are required for the contract to actually
hold.

## Acceptance

- `cargo test --release -p infer --test tokenizer_fingerprint_radix_isolation` passes.
- `cargo test --release` full sweep still passes (no regression on
  existing prefix-cache tests — they pass `[0; 32]` legacy namespace).
- `cargo test --release --no-default-features --features metal` still
  passes (Metal RadixCache instantiation goes through the same
  `new_with_namespace` path).
- Server boot log line `"prefix-cache namespace: surface=<cuda|metal-qwen3|metal-qwen35> tokenizer=<hex16> build=<hex8>"`
  appears once per active backend surface (so a CUDA boot prints one
  line; a Metal Qwen3.5 boot prints two — one for the bridge RadixCache
  if instantiated, one for the runtime).
- Manual smoke: change `models/Qwen3-4B/tokenizer.json` by one byte,
  restart server, confirm logged namespace tokenizer prefix differs.

## Out of scope

- **Mid-run tokenizer hot-swap defense.** Operator-policy contract
  (silent swap is forbidden; restart to clear cache) is sufficient.
  Adding a file-watcher with auto-restart is over-engineering for a
  problem that policy + boot-time logging covers.
- **Per-tenant / per-session tokenizer namespacing.** Single-namespace
  per RadixCache instance suffices; multi-tenant requires per-tenant
  RadixCache instances, which is a Tier-KV M_d.2 question, not M_d.1.
- **Chat template fingerprint.** Already implicit via `crates/chat`
  build. If ARLE ever adopts model-side `chat_template.jinja` rendering
  (not currently planned), the namespace becomes
  `sha256(tokenizer_fp ++ chat_template_fp ++ build)` — trivial extension.
- **NFC normalization in `Tokenizer::encode`.** HF tokenizer config
  already does this for Qwen3; double-normalize is a regression risk
  with no benefit. Verify per-model when adding new tokenizers.

## Bench gate

This is a correctness-only change. Per CLAUDE.md §Benchmarks "regression-check
minimum" — one `scripts/bench_guidellm.sh` run after landing to confirm
no perf regression from the boot-time `std::fs::read` + sha256.

## References

- Errors entry that motivated this:
  [`2026-05-07-radix-cache-no-tokenizer-fingerprint.md`](../experience/errors/2026-05-07-radix-cache-no-tokenizer-fingerprint.md)
- Parent plan (where M_d.1 sits in the combo roadmap):
  [`M_d-tier-kv-spec-decode-coordination.md`](M_d-tier-kv-spec-decode-coordination.md)
- HF tokenizer NFC anchor:
  `infer/src/tokenizer.rs:322-323` (test `test_encode_unicode_zwj_and_long_grapheme_no_truncation`)
- Hardcoded ChatML rendering:
  `crates/chat/src/protocol.rs:218` and `:295`
