# A refusal test was green for 55 days while the hostile state it refused never existed — platform-dependent fixture

Date: 2026-09-12. Found during the purposeful audit of shell checks that
cannot fail; fixed in PR #402.

## Context

`scripts/tests/test_pod_flow.sh` contains a block asserting that
`pick-gpu.sh check-free-set` refuses a GPU set containing a card held by a
live foreign claim. It planted a claim on GPU 2 and expected the check to
fail:

```bash
printf '...op=foreign\npid=%s\nstart=%s\n' "$$" "$(awk '{print $22}' /proc/$$/stat)" > "$TMP/tp4-busy/2"
set +e
… check-free-set 0,1,2,3 …; rc=$?; set -e
[ "$rc" -ne 0 ] && [ ! -e "$TMP/tp4-busy/0" ] && …
```

The block was green in the pre-push shell suite, which also runs on a macOS
host.

## Root cause

Two independent defects, each masking the other.

1. **The fixture could not construct a live claim on this host.**
   `pick-gpu.sh` decides a claim is live only when
   `$PROC_ROOT/<pid>/stat` exists and its field-22 start token matches.
   On macOS `/proc` does not exist, so the `awk … /proc/$$/stat` produced an
   empty start token and the liveness read classified every claim as stale.
   `check-free-set` reaped the planted claim and **succeeded** — the "busy"
   GPU was granted, the opposite of what the test claimed.
2. **The `&&` form converted that into silence.**
   `[ "$rc" -ne 0 ] && <state assertion>` is red only when the guarded
   command unexpectedly returns 0; under `set -e` a false `&&` list is
   non-fatal. The command did return 0 (the regression the test exists for),
   the line short-circuited, and the suite passed with the state assertion
   never evaluated.

The mechanism is **platform dependence**: the fixture encoded a Linux-only
state (a `/proc` start token) while running on a host without `/proc`. The
assertion itself was correct; it never once executed against the state it
was written for.

## How long

The block was introduced 2026-07-19 in `e5d9bc1b4` ("support receipt-bound
TP runs"), already with `/proc/$$/stat` and the `&&` chain. It was green and
meaningless on macOS for ~55 days until 2026-09-12.

## Fix (PR #402)

- Build the hostile state through the same indirection production uses:
  `PROC_ROOT` is overridable (`pick-gpu.sh:5` defaults to `/proc`; every
  read goes through it), so the test points `PROC_ROOT` at a synthetic root
  containing a fake live pid with a field-22 start token the claim matches.
  This constructs the state identically on macOS and Linux — the fake pids
  exist only in the synthetic root on either platform.
- Add the reverse control one semantic field away: the same claim on a pid
  with no stat file is reaped and the check passes, proving the gate keys on
  liveness rather than file presence.
- Split the `&&` line into an explicit rc assertion and an independent
  state assertion, so a returned 0 is itself red.

## Rule

A test that asserts a refusal must prove the refusal path actually executed.
A fixture that silently fails to construct the hostile state — here a
platform-dependent one (Linux `/proc` read on a macOS host) — pairs with a
weak assertion to produce a green that means nothing: the system under test
did the wrong thing and the test reported the right thing. Assert the
precondition (the hostile state is genuinely live) and the refusal
separately, and make the two differ by one field in a reverse control.
