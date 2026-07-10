---
title: #34 HF Hub blocker — Phase 0 audit + fix path matrix
date: 2026-05-09
type: research
status: parallel-axis
depends_on:
  - docs/research/2026-05-08-medusa-phase1a-hf-download-blocker.md
---

# #34 HF Hub `arle data download` blocker — Phase 0 audit + fix path matrix

> Source-grep + reproduce 2026-05-08 errors entry blocker。Confirmed still
> blocked。Identified fix path matrix。Parallel axis to #24 prefill graph
> chain — 不冲突 codex 当前 WIP。

## 1. Phase 0 Source audit

### Implementation 当前

`crates/train/src/hub_dataset.rs:35-40`:

```rust
pub fn download_dataset_file(repo_id: &str, filename: &str) -> Result<PathBuf> {
    let api = build_api().context("failed to initialise HuggingFace API")?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Dataset));
    repo.get(filename)
        .with_context(|| format!("failed to download '{filename}' from dataset '{repo_id}'"))
}
```

→ 使用 `hf-hub` `api::sync` 接口,`repo.get(filename)` 是单 call entry。

### 依赖版本 audit

| Crate | Version | Features | Status |
|-------|---------|----------|--------|
| `hf-hub` workspace lock | **0.5.0** | tokio | locked |
| crates.io 最新 | **1.0.0-rc.1** | -- | major upgrade |
| 0.5 semver bound 内最新 | 0.5.0(无 patch)| -- | dead-end |

3 crates use `hf-hub = "0.5"` with `tokio` feature:
- `crates/cli/Cargo.toml:26`
- `crates/train/Cargo.toml:25`(本 download path)
- `infer/Cargo.toml:35`

### Reproduce 验证

```bash
$ timeout 30 ./target/release/arle data download \
    --repo openai/openai_humaneval \
    --file openai_humaneval/HumanEval.jsonl.gz
[download_dataset] fetching ...
# 30s timeout 后无 output(2026-05-08 是 "request error: io: unexpected end of file"
# 现在直接 hang 30s) → blocker 仍存在,failure mode shift
```

→ Blocker confirmed active 2026-05-09。可能 hf-hub 0.5 + 当前 HF 服务端 chunked
encoding 行为不兼容。

## 2. 三种 fix 路径对比

### Path A — `hf-hub` 升级 0.5 → 1.0-rc(API rewrite)

| 维度 | 评估 |
|------|------|
| LOC | 100-200(3 crates 的 import + API call site 改)|
| Risk | 中(API changed: builder pattern, async/sync 分离)|
| Wall-clock | 2-4 hours codex |
| 依赖稳定 | 1.0-rc.1 是 release candidate,**可能继续变** |
| 可逆 | 易(revert lock + Cargo.toml)|
| 解决根因 | ✓ if 1.0 fixed the chunked encoding bug |

### Path B — `hf-hub` 中间版本(0.6 OR 0.7 OR 0.8)

需先 web search 看是否 stable 0.6+ 存在 + changelog 是否含 fix。
- **risk**:中间版本可能也 broken
- **LOC**:50-150(API drift 较小,vs 0.5)
- **wall-clock**:1-2 hours

### Path C — 替换 `hf-hub` 用 `reqwest` 直接 call HF API

| 维度 | 评估 |
|------|------|
| LOC | 200-400(reqwest call + JSON parse + cache path resolution + auth header)|
| Risk | 低(全自控)|
| Wall-clock | 4-6 hours codex |
| 依赖稳定 | reqwest 极稳 |
| 可逆 | 中(全 rewrite,revert 是 git revert)|
| 解决根因 | ✓ trivially(不依赖 hf-hub)|
| Bonus | 同 model.safetensors 直 download 路径 unify(infer 也用 hf-hub)|

### Path D(临时)— 用 `huggingface-cli download` shell out

| 维度 | 评估 |
|------|------|
| LOC | 50-100(spawn command + parse path output)|
| Risk | 低(huggingface-cli is python,稳)|
| Wall-clock | 1 hour Claude |
| 依赖外部 | requires `pip install huggingface-hub`(已有 in `.venv`)|
| 可逆 | 易 |
| 解决根因 | ✗ workaround,不修 hf-hub |
| Bonus | 立即解锁 Medusa #28 dataset download |

## 3. 推荐路径

**Phase 0 立即 unblock — Path D shell-out workaround**(1 hour Claude):
- 解锁 #28 Medusa Phase 1a dataset download 立即
- Bonus:`infer` 端 model download 仍走 hf-hub 0.5(HF Hub 本体可能能下 model files,只是 dataset path 失败)

**Phase 1 后续永久 fix — Path A `hf-hub 1.0` upgrade OR Path C reqwest 替换**(codex pickup):
- Path A 简单但依赖 unstable(rc)
- Path C 慢但全可控
- License threshold:`arle data download <repo> <file>` 5 个不同 repos 全 PASS
  + Medusa Phase 1.A.1 step 2 PASS

## 4. License-or-kill criteria

| 维度 | PASS | KILL |
|------|------|------|
| 5 个 dataset 测试(含 large file > 100MB)| 全成功 | 任一 io 错误 |
| 已下文件 SHA256 vs HF 官方 | 匹配 | 不匹配 |
| Cache path 命中 | 第二次 call 0 网络 IO | 重 download |
| `cargo test --workspace` | PASS | regression |

## 5. 不在本 axis(避免 scope creep)

- ❌ 重写 `infer::hf_hub`(model download 路径,目前 working,scope creep)
- ❌ 加 retry / exponential backoff(blocker 是 protocol bug 不是 transient,加 retry 不解决)
- ❌ 加新 HF endpoint mirrors 全 enumerate(deferred)

## 6. Action items

1. **Path D shell-out workaround Claude tick 内做**(1 hour est,unblocks Medusa Phase 1a 立即)
2. **Path A/C 根 fix codex pickup**(after #24/#37 chain done OR parallel,200-400 LOC)
3. **本 brief commit + push** 作 #34 fix work 时 reference

## 7. 状态

#34 HF Hub blocker confirmed active 2026-05-09(failure mode shift from io error
→ silent hang)。`hf-hub 0.5.0` deadlocked semver 内,无 patch available。Path D
shell-out workaround Claude 立即可做(1 hour),Path A/C 永久 fix codex pickup
(2-6 hours)。完全 parallel #24/#37 prefill graph chain。

## Cross-references

- 上游 errors:`docs/research/2026-05-08-medusa-phase1a-hf-download-blocker.md`
- 当前 impl:`crates/train/src/hub_dataset.rs`
- 3 dependent crates:`crates/{cli,train}/Cargo.toml` + `infer/Cargo.toml`
- 影响 task:#28 Medusa Phase 1a dataset download(blocked)
