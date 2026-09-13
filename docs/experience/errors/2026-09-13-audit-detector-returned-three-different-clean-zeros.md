# An audit detector returned a clean zero three times, for three different reasons

Date: 2026-09-13.

## Context

A parity gate was found computing a worst-element statistic, printing it, and
never using it in its pass predicate — `max_over_rms` in
`crates/infer-cuda/examples/marlin_fp8_parity.rs`, produced at :294, printed at
:560, :563 and :622, absent from the predicate at :544-547. One instance of a
class is a reason to audit the class, so the question became whether any other
gate computes a statistic that no boolean context reads.

The audit was a text sweep over the 21 parity examples: extract the numeric
fields of each stats-shaped struct, find every use site, and report any field
whose uses never appear beside a comparison, an assertion or a boolean.

## Root Cause

The sweep reported "total flagged: 0" and exited 0. It was wrong, and then
wrong twice more. Each version failed for an unrelated reason and each failure
presented identically — no output, exit status 0, which reads as "the class is
clean".

1. `grep -oE '^\s+[a-z_]+: f64'`. BSD grep's ERE has no `\s`, so the field
   extractor matched nothing on every file and the per-file guard skipped all
   21. The tool this runs on is macOS, where `grep` is BSD.
2. `[a-z_]+` after that was fixed. Field names containing a digit do not match,
   so `rel_l2` was dropped from every struct. This one would not have produced
   a false "never gated" report, because the dropped field is the one that is
   gated — but any digit-bearing field elsewhere would have been skipped in
   silence.
3. `for fl in $flds` over a newline-separated list. **zsh does not word-split an
   unquoted parameter expansion**, so the loop ran once with the whole string
   bound to `$fl`, every per-field grep matched zero lines, and the zero-match
   guard skipped every file. In bash the same line iterates per field. This is
   the same family as `${PIPESTATUS[0]}` expanding to empty in zsh: a bash
   idiom that fails silently rather than erroring.

What caught all three was a positive control: assert that a known instance —
`marlin_fp8_parity.rs` / `max_over_rms` — appears in the output, and treat its
absence as a broken detector rather than a clean result. Versions 1 and 2 were
caught by controlling the intermediate (the field list must contain the three
known field names); version 3 by controlling the final output.

## Fix

The sweep was rewritten in Python, which removes the shell-dialect surface
entirely, and it now prints the positive control's verdict on every run beside
the count. The audit's actual result: two discarded statistics, `max_over_rms`
and `mean_ratio`, both in `marlin_fp8_parity.rs`, and nothing in any other
gate. The class is bounded to the gate that prompted the question.

The detector's own limit is stated with the result rather than left implicit:
it can only see statistics that live in a named struct, so a quantity computed
into a local and printed would not appear.

## Rule

A detector that reports a clean zero has not been shown to work. Before
quoting the zero, make it report a known instance — and when the search is
staged, control the intermediate as well as the output, because a stage that
silently produces an empty input set gives the same clean zero as a genuinely
clean tree. On this machine the shell is zsh and `grep` is BSD, so a bash-and-
GNU idiom is not a neutral choice of syntax: `\s` in an ERE, `${PIPESTATUS[0]}`,
and word-splitting an unquoted list all fail without an error.
