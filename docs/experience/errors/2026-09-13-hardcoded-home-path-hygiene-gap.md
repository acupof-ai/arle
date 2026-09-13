# A hardcoded /home machine path sat in committed Rust; the abs-path check omits /home

Date: 2026-09-13. Found while investigating why
`crates/train/tests/test_infer_teacher.rs` never ran.

## Context

`crates/train/tests/test_infer_teacher.rs:16` hardcodes a developer-machine
absolute path in a test-fixture default:

```rust
// Redacted: "<home>/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base"
const DEFAULT_QWEN35_08B_DIR: &str = "<home>/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
```

`<home>` is the Linux user-home root segment, the machine-specific part this
entry is about.

The path exists on one person's box and nowhere else. `resolve_qwen35_08b_dir`
(:32-35) uses it as the fallback when its override
`ARLE_PARITY_QWEN35_08B_DIR` is unset, so on any other machine the "default"
is a nonexistent directory.

The pre-push hygiene check is meant to keep exactly these literals out of
tracked text.

## Root Cause

`scripts/lane_pr_precheck.py:53` builds the banned-segment set explicitly:

```python
_ABS_SEGMENTS = "|".join(["/" + s for s in
    ("Users/", "root/", "data0", "mnt/", "host/")])
```

`/home/` is not in the list. The check is an allow-by-omission denylist:
each machine-root prefix was added as it was found, and `/home/` (the Linux
home-root equivalent of the macOS `Users/` home root) was never enumerated, so the literal
passed hygiene on every push. The same defect class as the rest of this
week — the mechanism runs, but its input set is incomplete.

A second, related gap sits next to it: the file also never compiled in any
CI lane (file-level cuda cfg against no-cuda invocations, tracked
separately), so even a content scan that only runs on changed files had no
reason to look at it often. The path finding and the compile finding are
independent.

## Fix

Entry only; the marker list is owned by e1. The minimal change is adding
`"home/"` to the tuple at lane_pr_precheck.py:53, then resolving the one
existing hit (the test default above) so the repo is clean when the rule
lands — the rule and its first violation should not merge separately or the
hygiene check goes red on an unrelated file. Machine paths in tests should
be env/override-only with no committed default, or a repo-relative
`models/` path.

## Rule

A denylist of machine-root prefixes must be assembled from the full set of
home roots the platforms use, not the ones hit so far: the macOS
`Users/` root, Linux `home/`, and `root/` together. When adding a banned-prefix rule, grep the tree for the
new prefix and clear or convert its existing hits in the same change, or
the new rule starts life red.
