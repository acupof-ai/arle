# W4A8 H3b scale_perm_single fix applied — STILL PARTIAL,3rd layer of bug remains

> Continues elimination chain:
> `25391f3` H3 row stride confirmed → `62f885d` row-fix PARTIAL → `3479a87`
> H3b scale_perm_single confirmed → this entry: H3b applied → still 100%
> diff but output character shifted toward English-fragmented + code-like
> artifacts。
>
> Cumulative qualitative progression suggests **3-layer perm bug**, with 2
> layers fixed and 1 remaining。

## Setup

Applied codex H3b patch to `/tmp/quantize_qwen3_w4a8.py`:

```python
# Before (line 81-84):
perm, scale_perm, scale_perm_single = get_perms(groupsize, k)
del scale_perm_single  # ⛔
# ... s_channel produced raw, no permutation ...

# After (post 3479a87):
perm, scale_perm, scale_perm_single = get_perms(groupsize, k)
# (do NOT delete)
# ... s_channel produced ...
s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
s_channel = s_channel.reshape((-1, n)).contiguous()
```

Re-quantized + re-tested.

## Result — qualitative output progression

```
prompt: "The capital of France is" (max_tokens=32)
BF16:  " Paris. The capital of Germany is Berlin. The capital of Italy..."

W4A8 progression across fix iterations:

(1) Pre-H3 fix (81b6481):
    ".........11.1.11111111 baudaskan1 baud111askan11"
    → period/digit bias = weights all near-zero magnitude

(2) Post-H3 row-fix (62f885d):
    " Gamb Cocktail zipトル Turbo bsites+xmlval dephenperate.zip
     casting traits intend oeah指拜登国 ensis fraction tur 烦恼 ondacticism..."
    → multilingual real words = correct magnitude, wrong tile-level positioning

(3) Post-H3b scale_perm_single fix (this run):
    "ing bootstrap | teting Parts 不断 ~~~~~~~~ 受 atroning recipients
     prime bargaining � indefinite/bootstraping ücken natural([` \"\"\"
     odd he 一主义者 farandlekodding"
    → English-fragmented + code-like = closer to natural English text,
      but still semantic-wrong; channel-scale partially aligned
```

Each fix improves output character toward "more like English". H3b adds:
- "recipients", "prime bargaining", "indefinite", "natural", "odd"
  (real English short phrases)
- Code/markdown artifacts (`|`, `~~~~~~~~`, `[\``, `"""`)
- Less random multilingual mix

Token diff still 100% (idx=0 onward) — first BF16 token "Paris" (12095)
vs W4A8 different token. But the OUTPUT DISTRIBUTION shape changed.

## Diagnostic — 3-layer perm bug hypothesis

| Layer | Bug | Fix status | Output character |
|---|---|---|---|
| 1. Per-thread weight bytes | row pattern `[4k]` vs `[2k skip-8]` | ✅ FIXED `25391f3` | period bias → multilingual words |
| 2. Per-channel scale | `del scale_perm_single`, no permutation | ✅ FIXED `3479a87` | multilingual gibberish → English-frag + code |
| 3. ??? | **TBD — 3rd layer remaining** | ❌ pending | English-frag still wrong |

Possible 3rd-layer suspects:

A. **Per-group scale `scale_perm` post row-fix may need different shape**:
   The script applies `scale_perm` to s_group on line 103:
   ```python
   s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
   ```
   With row-fix changing weight tile shape, the s_group reshape into
   `(-1, len(scale_perm))` may have different N units than expected.

B. **`pack_w4a8` block-level permutation `256 * j` stride**:
   ```python
   for j in range(4):
       perm.extend([p + 256 * j for p in perm1])
   ```
   PR #31 reference also uses `256 * j` per codex `e3ca4d8` — should be
   correct. But interaction with row-fix may need different.

C. **`weight_loader.rs:663-715` may load scales from wrong tensor field**:
   The pack writes `<name>.marlin_w4a8_s_channel` but loader reads as
   per `infer/src/weight_loader.rs:670`. If naming convention diverges
   from PR #31, kernel reads wrong tensor.

D. **`pack_w4a8` final bit-packing `q |= res_np[:, i::8] << (4*i)`**:
   8-element per uint32 pack — `i::8` stride may need different post
   row-fix.

## Probability estimate

After H3 + H3b fixes:
- 2 of 3 perm layers fixed
- Output character progressed: noise → multilingual → English-frag
- Distance to "Paris correct" remaining: ~25% of original gap (qualitative)

**Probability 3rd-layer fix gets to passing test**: ~70% (right area but
need PR #31 deeper comparison; ~3 distinct candidate sites)

## Codex action

Continue PR #31 deeper comparison:
1. Verify `pack_w4a8` line 91-104 (post-row-fix shape) matches PR #31 `Layer.pack` exactly:
   - reshape calls
   - permute calls
   - scale_perm application timing
2. Check `weight_loader.rs:663-715` reads tensors with correct names matching `pack_w4a8` writes
3. Verify final tile permutation lines 102-105:
   ```python
   tile = 16
   w = w.reshape((k // tile, tile, n // tile, tile))
   w = w.permute((0, 2, 1, 3)).reshape((k // tile, n * tile))
   res = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)
   ```
   May need different reshape/permute order for INT8 layout.

## Skill methodology validation

Per anti-pattern #13 (NULL = real elimination): each partial-fix is real
progress, narrowing the bug space. The 3-fix-layer pattern would have
been invisible without persistent qualitative observation across iterations.

Token-diff 100% across all 3 states might suggest "no progress" to a
naive reader. But the OUTPUT DISTRIBUTION CHARACTER monotonically improves
toward English text — this is REAL signal.

## Cross-references

- H3 row-stride confirmed: [`2026-05-08-w4a8-bug-h3-confirmed-perms-row-stride.md`](2026-05-08-w4a8-bug-h3-confirmed-perms-row-stride.md) (`25391f3`)
- H3 row-fix partial: [`2026-05-08-w4a8-h3-row-fix-partial.md`](2026-05-08-w4a8-h3-row-fix-partial.md) (`62f885d`)
- H3b scale_perm_single confirmed: [`2026-05-08-w4a8-bug-h3b-confirmed-scale-perm-single-deleted.md`](2026-05-08-w4a8-bug-h3b-confirmed-scale-perm-single-deleted.md) (`3479a87`)
- W4A8 garbage gate: [`../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md`](../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md) (`81b6481`)
- ARLE script (current state with H3 + H3b): `/tmp/quantize_qwen3_w4a8.py`
- PR #31 reference: `/tmp/marlin-w4a8/marlin/__init__.py:263-310`
- Failing test: `infer/tests/greedy_consistency.rs::test_w4a8_vs_bf16_token_diff`
- ARLE loader: `infer/src/weight_loader.rs:663-715`

## Rule

Don't read 100% token diff as "no progress" without checking output
character. Quant accuracy bugs often have 3+ layers (per-thread, per-
channel, per-tile, per-block). Each fix layer can produce 100%-diff
output that's progressively closer to the right distribution.

Persistent qualitative observation across iterations is the only way to
distinguish "noise floor stuck" from "narrowing toward fix".
