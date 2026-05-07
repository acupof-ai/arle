# M_d.1 — Tokenizer fingerprint as RadixCache namespace

> Closes the silent-corruption hole documented in
> [`2026-05-07-radix-cache-no-tokenizer-fingerprint.md`](../experience/errors/2026-05-07-radix-cache-no-tokenizer-fingerprint.md).
> Refines the original 5-step sketch down to **3 steps** after a P0
> survey of the actual surface — Steps 2 (NFC) and 3 (chat-template fp)
> are unnecessary in ARLE for the reasons noted below.

## P0 survey findings (2026-05-07)

`grep` + targeted reads of `infer/src/tokenizer.rs`,
`infer/src/prefix_cache.rs`, `infer/src/http_server/`, and
`crates/chat/src/` exposed three facts that simplify the original sketch:

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

### 2. `RadixCache` namespace field

`infer/src/prefix_cache.rs`:

```rust
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
        Self::new_with_namespace(block_size, [0; 32])  // legacy default
    }

    pub fn new_with_namespace(block_size: usize, namespace: [u8; 32]) -> Self { ... }

    pub fn namespace(&self) -> &[u8; 32] { &self.namespace }
}
```

Snapshot load (`from_serde`) MUST verify namespace match before
returning the deserialized cache; mismatch → return error, not silent
acceptance. This is the contract that prevents the cache-hit-on-stale-tokenizer
silent corruption.

### 3. Server boot wiring

Two call sites to update:

- `scheduler/cuda/core.rs` — wherever `RadixCache::new(block_size)` is
  called for the CUDA scheduler. Replace with
  `RadixCache::new_with_namespace(block_size, derive_namespace(&tokenizer))`
  where `derive_namespace` = `sha256(tokenizer.fingerprint() ++ env!("CARGO_PKG_VERSION") ++ build_git_sha)`.
- `backend/metal/prefix_cache.rs:38` — same swap.

Server boot logs the namespace at INFO:
`"radix-cache namespace: tokenizer=<hex16> build=<hex8>"`. Operators
who hot-swap `tokenizer.json` will see the namespace change after
restart and know the cache cleared. Silent swap mid-run is not
defended against (the loaded `Tokenizer` instance keeps its old
fingerprint until restart) — that's the operator-policy half of the
contract, not code.

### Test (T1) — namespace isolation

`infer/tests/tokenizer_fingerprint_radix_isolation.rs`:

1. Create RadixCache A with namespace `[0xAA; 32]`, insert blocks for
   token sequence `[1, 2, 3, 4]`.
2. Snapshot A to bytes via serde.
3. Try to load the snapshot into RadixCache B with namespace
   `[0xBB; 32]` — assert error.
4. Create RadixCache C fresh with namespace `[0xBB; 32]`, lookup
   `[1, 2, 3, 4]`, assert MISS (no shared state with A).

This is a CPU-only test (~10 ms), runs in `cargo test --release` no
features needed. Catches future refactors that bypass the namespace
check.

## Tasks

| # | Task | File | LOC est. | Owner |
|---|---|---|---|---|
| 1 | Add `fingerprint: [u8; 32]` to `Tokenizer`, compute in `from_file`, expose `fingerprint()` | `infer/src/tokenizer.rs` | ~15 | Claude |
| 2 | Add `namespace: [u8; 32]` to `RadixCache`, `new_with_namespace`, snapshot-load mismatch check | `infer/src/prefix_cache.rs` | ~30 | Claude |
| 3 | Server boot wiring (CUDA + Metal call sites + INFO log) | `scheduler/cuda/core.rs`, `backend/metal/prefix_cache.rs`, `main.rs` | ~20 | Claude |
| 4 | Test: namespace isolation + snapshot-mismatch reject | `infer/tests/tokenizer_fingerprint_radix_isolation.rs` (new) | ~80 | Claude |

**Total: ~145 LOC, half-day.** Smaller than the original sketch
because two of the five mitigations were already in place (NFC via HF
config, chat template via build).

## Acceptance

- `cargo test --release -p infer --test tokenizer_fingerprint_radix_isolation` passes.
- `cargo test --release` full sweep still passes (no regression on
  existing prefix-cache tests — they pass `[0; 32]` legacy namespace).
- `cargo test --release --no-default-features --features metal` still
  passes (Metal RadixCache instantiation goes through the same
  `new_with_namespace` path).
- Server boot log line `"radix-cache namespace: tokenizer=<hex16> build=<hex8>"`
  appears once per backend.
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
