# codex review --uncommitted WEDGED on OpenAI API call via local proxy — timeout(1) failed to enforce 900s limit

## Context

PF8.3 codex Strategy A' substrate fully validated (cargo check + clippy +
greedy_consistency + e2e all PASS, codex review caught 3 real bugs all
FIXED, post-fix re-PASS). Commit blocked by codex's FINAL `timeout 900s
codex review --uncommitted` pass which has been Working **47+ min**
cumulative — past the 900s (15 min) timeout.

## Root Cause

Direct ps verify per skill v1.11.0+ #32 (4b30c15 33min wedge precedent)
THIS tick:

```bash
$ ps aux | grep codex
ckl  1867385  06:47  timeout 900s codex review --uncommitted
ckl  1867386  06:47  node /home/ckl/.bun/bin/codex review --uncommitted
ckl  1867396  06:47  codex review --uncommitted   # 0:07 CPU in 47+ min = 0.2%

$ cat /proc/1867396/status | grep State
State: S (sleeping)

$ cat /proc/1867396/wchan
futex_wait

$ lsof -p 1867396 | grep TCP
codex 1867396 ckl  33u  IPv4  6738753  TCP localhost:60904->localhost:7897 (ESTABLISHED)
```

**Diagnosis**:
1. `timeout 900s` should have killed the process at 07:02 (06:47 + 15min)
2. Process still alive at ~07:14+ (12+ min past timeout)
3. Process state **S (sleeping)**, wchan **futex_wait** — blocked on a mutex
4. 27 threads (multi-threaded), all blocked
5. **Active TCP connection to localhost:7897** (typical Clash/V2Ray HTTP proxy port)
6. CPU time only 0:07 in 47+ min = ~0.2% utilization = network-blocked

**Root cause**: codex review subprocess is blocked on an HTTP request to
OpenAI's API (via the local proxy at port 7897). The remote API call is
hung — codex is in `futex_wait` waiting for the response future to
resolve. `timeout(1)` sends SIGTERM at 15min mark, but the codex CLI
either:
- Catches SIGTERM and tries graceful shutdown (which itself hangs on
  the same blocked future)
- Or the SIGTERM doesn't propagate to the blocked thread

Either way: timeout(1) fails to enforce the limit.

**Diagnostic refinement (51m+ tick)**: proxy + upstream are HEALTHY:

```bash
$ curl -s -o /dev/null -w "HTTP %{http_code} time_total %{time_total}s\n" \
    --max-time 5 http://localhost:7897
HTTP 400 time_total 0.002346s     # proxy alive, returns 400 for missing target

$ HTTPS_PROXY=http://localhost:7897 curl -s -o /dev/null \
    -w "HTTP %{http_code} time_total %{time_total}s\n" --max-time 8 \
    https://api.openai.com/v1/models
HTTP 401 time_total 0.393701s     # proxy → OpenAI works, API responds 401 (needs auth)
```

**This rules out transport-layer/proxy issues**. The wedge is
specifically PID 1867396's stuck session-state — not a network
infrastructure problem. The earlier-established TCP connection
(localhost:60904 → localhost:7897) is dead at the application layer
but alive at the OS layer (no FIN/RST received).

`kill -TERM 1867396` (or `-9`) is the ONLY recovery; there is nothing
to fix upstream.

Other codex CLI sessions (PIDs 85424, 642435) running fine concurrently
— rules out codex CLI binary corruption or system-wide codex issue.

## Fix

### Recovery for current wedged process

```bash
# Try graceful first (codex CLI can flush state):
kill -TERM 1867396

# If still alive after 30s:
kill -9 1867396

# Verify cleanup:
ps aux | grep "codex review"  # should be empty
```

After kill: codex CLI (in tmux pane 0:0) should return to prompt with
"Conversation interrupted" or similar. User can then:
- Inspect the diff one more time manually
- `git add` + `git commit` the PF8.3 substrate (skipping the review)
- The review CAUGHT 3 bugs already and they've been FIXED + re-verified
  via cargo check + clippy + tests; the additional review pass was a
  nice-to-have safety net, not a hard requirement

### Long-term prevention

For `codex review --uncommitted` invocations:
- DON'T rely on `timeout(N)` alone — the CLI can ignore signals during
  blocked network calls
- WRAP with explicit child PID kill: `(timeout 900s codex review &) ;
  PID=$!; sleep 950; kill -9 $PID 2>/dev/null`
- OR use `setsid + timeout --kill-after=30s` for forced kill after
  graceful timeout

Per CLAUDE.md §Delegation: codex review at Bash is the canonical
non-hanging path BUT this wedge proves it CAN hang on network issues
(not on subprocess hangs that the original CLAUDE.md note covered).

## Rule

When `codex review --uncommitted` exceeds its `timeout` limit:
1. Direct ps verify per skill #32 (state S + wchan futex_wait + low CPU
   + active network FD = wedged on network call, NOT making progress)
2. PushNotification user with concrete recovery (kill PID + manual commit)
3. Don't auto-kill from session — codex CLI session has unsaved state;
   user authorization preferred
4. Update skill v1.12.0+ candidate: add anti-pattern #34 = "timeout(1)
   may fail on subprocesses that catch SIGTERM during network blocks;
   use --kill-after for hard enforcement"

## Cross-references

- 4b30c15 (prior 33min wedge — different cause but similar signature)
- ace3cbe (codex review pattern empirical validation — this session
  caught 3 real bugs BEFORE this final-pass wedge)
- 9ccd36b (next-session pickup state — PF8.3 substrate validation
  trace; commit was waiting on this final review)
- skill v1.11.0+ #32 (peer Working >5min direct ps/log/curl verify)
- CLAUDE.md §Delegation (codex review CLI path canonical for non-hang
  delegation — but THIS wedge proves network-hang is still possible)
