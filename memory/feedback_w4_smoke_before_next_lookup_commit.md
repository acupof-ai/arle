# Feedback: W4 Smoke Before Next Lookup Commit

When working on A-line W4 lookup/admission/session-slot changes, cargo tests
are not the success signal. After every commit that touches lookup,
admission, or slot lifecycle, immediately run a minimal W4 smoke with one
session warmup -> resume before taking the next implementation step.

Required smoke signal:

- `matched_prefix > 32`
- `avoided_prefill > 5%`

If the smoke still reports `matched_prefix == 32`, treat it as an implementation
bug and root-cause before stacking another patch. The final acceptance run still
requires canonical 128-session W4 and a real avoided-prefill number.
