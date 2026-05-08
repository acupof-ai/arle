# Medusa Phase 1.A — `arle data download` HF Hub blocker discovered

> Per `b4ae33f` Phase 1.A directive,attempted to validate Phase 1.A.1
> step 2(`arle data download --repo tatsu-lab/alpaca --file data/train.json`)
> on production `target/release/arle` binary。
>
> **Result:consistent failure** with `request error: io: unexpected
> end of file`,**3 different repos/files**(`openai/openai_humaneval`,
> `tatsu-lab/alpaca`,both with and without `HF_ENDPOINT` mirror set)。
> HF Hub itself is reachable(curl 200 response)。**`hf-hub` Rust
> library or HTTP client issue blocks Phase 1.A pipeline**。

## Empirical evidence

### Test 1 — `openai/openai_humaneval`
```
$ arle data download --repo openai/openai_humaneval --file openai_humaneval/HumanEval.jsonl.gz
[download_dataset] fetching 'openai_humaneval/HumanEval.jsonl.gz' from dataset 'openai/openai_humaneval'
[download_dataset] error: failed to download ... request error: io: unexpected end of file
```

### Test 2 — `tatsu-lab/alpaca`(codex `b4ae33f` recommended)
```
$ arle data download --repo tatsu-lab/alpaca --file data/train.json
[download_dataset] error: failed to download ... request error: io: unexpected end of file
```

### Test 3 — with `HF_ENDPOINT=https://hf-mirror.com`
```
$ HF_ENDPOINT=https://hf-mirror.com arle data download --repo tatsu-lab/alpaca ...
[download_dataset] error: failed to download ... request error: io: unexpected end of file
```

### Direct connectivity verified

```
$ curl -sI https://huggingface.co/datasets/tatsu-lab/alpaca
HTTP/2 200
content-type: text/html; charset=utf-8
content-length: 571568
```

HF Hub itself reachable;curl HTTP/2 200。Issue is library-internal,
not network-blocking。

## Hypothesis

The `hf-hub` crate(or its HTTP client `ureq`)may be:
- Not handling HF Hub's redirect chain correctly
- Mismatching expected protocol(HTTP/1.1 vs HTTP/2)
- Failing on TLS handshake mid-stream
- Timing out before completing chunked transfer

## Implication for Medusa Phase 1.A

Phase 1.A.1 step 2 is BLOCKED until codex fixes `arle data download`。
Either:
- Path 1:codex investigates `hf-hub` Rust crate(`crates/train/src/hub_dataset.rs`)
- Path 2:fall back to manual `wget`/`curl` download → place file in HF cache dir
  → `arle data convert` reads cached file
- Path 3:add Python-side fallback via `huggingface_hub`(Python lib)

Path 2 is the quickest workaround:user runs:
```bash
mkdir -p ~/.cache/huggingface/datasets/tatsu-lab--alpaca
cd ~/.cache/huggingface/datasets/tatsu-lab--alpaca
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train.json
arle data convert --input ~/.cache/.../train.json --format alpaca --output /tmp/alpaca.jsonl
```

## Codex pickup recommendation

This is a substrate fix(library integration),**codex own**:
- Investigate `crates/train/src/hub_dataset.rs:download_dataset_file`
- Check `Cargo.toml` for `hf-hub` version,maybe upgrade
- Test with `RUST_LOG=debug ./target/release/arle data download ...`
- Add `--proxy` flag if needed for proxy-only environments

Effort:0.5d codex(library debugging + retry logic + maybe upgrade)。

## Phase 1.A pickup chain refinement

Per this blocker:
- Phase 1.A.1 step 1(HF login):user-side(unblocked)
- Phase 1.A.1 step 2(download):**BLOCKED**(this entry)
- Phase 1.A.1 step 3(tokenize):depends on step 2
- Phase 1.A.1 step 4-5(train/eval Medusa):depends on Phase 1.B(codex)

**Net**:Phase 1.A pickup needs codex first to:
- (a)Fix `arle data download` 
- (b)Implement `arle train medusa` + `arle eval medusa`(Phase 1.B per `afdddec`)

OR Claude can use Path 2 manual workaround now if user permits。

## Cross-references

- Phase 1.A directive: `b4ae33f`(`docs/plans/M_medusa-phase1a-dataset-directive.md`)
- Phase 1.A.1 inventory: `74bde06`(584 tokens vs 100k+ target)
- Medusa Phase 0: `afdddec`
- Source: `crates/train/src/commands/download_dataset.rs:52`
- Library: `crates/train/src/hub_dataset.rs::download_dataset_file`

## Status

- ❌ `arle data download` blocked at `hf-hub` library level
- ⏳ Codex pickup:fix `arle data download`(0.5d substrate)
- ⏳ Or Claude workaround:manual wget(0.25d but bypasses ARLE infra)

## Rule

**When Phase 1.A.1 cannot proceed via existing infrastructure,document
the blocker as a substrate gap rather than working around it silently**。
A library-level HF Hub download bug affects all future dataset-loading
features(SFT,Medusa,xgrammar)— silent workaround leaves substrate
fragile。

For ARLE specifically:`hf-hub` integration must be production-grade
before any axis 2 training axis can deploy。Codex pickup priority:
P1(blocking Medusa Phase 1.A,dependency on multiple future axes)。
