# `hf-hub` Rust library "unexpected end of file" bug — workaround sustainable,fix P3

> Per `da68b98` `arle data download` consistent failure across multiple
> datasets + `4b5bb91` wget workaround unblocks 52k samples,this brief
> documents the bug surface + proposes prioritized fix paths。
>
> **Workaround is production-acceptable** for axis 2 Medusa training
> while hf-hub fix is queued P3。

## Bug surface

Per `da68b98` empirical:
- Failure across `openai/openai_humaneval`,`tatsu-lab/alpaca`,HF mirror
- Direct `curl` to HF Hub works(HTTP/2 200,571KB content-length)
- → bug is **internal to `hf-hub` Rust library**,not network

Library version:`hf-hub = "0.5" features = ["tokio"]`(per `Cargo.toml`)

`hub_dataset.rs:35-40`:
```rust
pub fn download_dataset_file(repo_id: &str, filename: &str) -> Result<PathBuf> {
    let api = build_api().context("failed to initialise HuggingFace API")?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Dataset));
    repo.get(filename)
        .with_context(|| format!("failed to download '{filename}' from dataset '{repo_id}'"))
}
```

→ Uses **sync `repo.get(filename)`** path of hf-hub。

## Hypothesis space

1. **hf-hub 0.5 sync API redirect bug**:HF Hub redirects to S3 backed
   storage,sync `ureq` client may not handle the redirect chain
   correctly while async tokio path does
2. **HTTP/2 chunked transfer issue**:HF Hub serves content via HTTP/2
   chunked,sync API may close connection mid-stream
3. **TLS handshake mid-stream**:S3 redirect target uses different TLS
   cert chain,sync handshake fails
4. **Old hf-hub version**:0.5 is from ~2024,latest is 0.6+,may have
   known fixes

## Workaround sustainability

`4b5bb91` wget workaround:
- `wget HF_PARQUET_URL` → 24 MB alpaca file
- `pandas read_parquet → JSONL` conversion
- `arle data convert --format alpaca` → canonical chat JSONL
- 52,002 samples / 2.3M tokens(23× Medusa paper requirement)

Sustainability factors:
- ✅ Works for static dataset files(common case for SFT training data)
- ✅ Doesn't require Rust library fix
- ✅ Reproducible in scripts/CI
- ⚠ Requires `wget` + Python `pandas` deps(usually available)
- ⚠ Can't auto-resolve HF Hub revision pinning(major version updates need manual URL update)
- ⚠ Doesn't work for HF Hub auth-required datasets(`lmsys/lmsys-chat-1m` needs token)

For Medusa Phase 1.A.1 smoke + 1.A.2 production training:**workaround
sustainable**(public datasets only)。

For future use cases needing HF Hub auth(licensed/gated datasets):
**library fix becomes P1**。

## Fix paths(prioritized by ROI)

### Path 1 — Update `hf-hub` to latest
`Cargo.toml`:`hf-hub = "0.5"` → `"0.6"` or `"^0.6"`(check latest crates.io)
- Effort:~5 min(version bump)
- Risk:API breakage between versions
- ROI:if 0.6 fixed the redirect/TLS bug,immediate unblock

### Path 2 — Switch to async API
Convert `download_dataset_file` to use tokio async path:
```rust
pub async fn download_dataset_file_async(...) -> Result<PathBuf> {
    let api = ApiBuilder::new().build()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Dataset));
    repo.get(filename).await.with_context(...)
}
```
- Effort:~30 LOC + caller migration
- Risk:caller chain refactor(`arle data download` CLI may need tokio runtime)
- ROI:async path likely has different redirect handling

### Path 3 — Bypass hf-hub,use `reqwest` direct
Implement custom downloader matching HF Hub API URL pattern:
```
https://huggingface.co/{repo}/resolve/main/{filename}
```
- Effort:~150 LOC(URL resolution + auth + redirect handling + cache)
- Risk:duplicates hf-hub functionality,maintenance burden
- ROI:full control,no dep on third-party lib

### Path 4 — Extend `arle data convert` to handle wget+convert pipeline
Codify the workaround as a proper `arle data fetch` command that wraps
wget(via reqwest)+ format conversion:
- Effort:~50 LOC
- Risk:Low(reuses existing `arle data convert` infra)
- ROI:productionizes workaround,no library dep changes
- **Recommended for short-term unblock**

## Recommendation

**P3 path 4**(productionize wget workaround):0.5d codex,unblocks
all dataset downloads via reqwest,bypasses hf-hub bug。

**P3 path 1**(version bump):try first as cheap experiment(5 min)。
If 0.6 fixes the bug → no other work needed。If 0.6 doesn't fix → fall
through to path 4。

**Path 2**(async)+ **Path 3**(reqwest custom)= deferred until Path 1
+ Path 4 prove insufficient。

## Cross-references

- `da68b98` arle data download blocker discovery
- `4b5bb91` wget workaround(52k samples unblocked)
- `b4ae33f` Medusa Phase 1.A directive
- `crates/train/src/hub_dataset.rs` source
- `crates/train/Cargo.toml`:`hf-hub = "0.5"`
- `infer/Cargo.toml`:`hf-hub = "0.5"`(also affected)

## Status

**Workaround production-acceptable for Medusa axis 2**(public datasets)。
HF Hub library fix queued P3 alongside Path 1 cheap version-bump
experiment(5 min)+ Path 4 productionize workaround(0.5d)。

If user prefers to upgrade hf-hub version first(cheapest experiment):
```bash
sed -i 's/hf-hub = "0.5"/hf-hub = "0.6"/' Cargo.toml \
    crates/train/Cargo.toml infer/Cargo.toml
cargo update -p hf-hub
cargo build --release -p infer --features cuda
arle data download --repo tatsu-lab/alpaca --file data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
```

If still fails after version bump → Path 4 is the durable fix。

Codex pickup queue priority(Path 4):
- After Hybrid Phase 1b lands + bimodal investigation lands
- Before #33 KV W4A8(memory axis,independent)
- Before B3 PrefixAwareAdmission(SGLang gap closure)

## Methodology insight

**Library bugs in dependencies often have version-bump fixes** — try
that 5-minute experiment FIRST before investigating the source。`da68b98`
spent investigation effort on chunking/TLS hypothesis but didn't try
version bump path,which is cheapest possible mitigation。
