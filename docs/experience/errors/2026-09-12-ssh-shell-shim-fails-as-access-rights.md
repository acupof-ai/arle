# A broken `ssh` shell shim fails as an access-rights error; credentials were never the problem — cause observed

Date: 2026-09-12. No code changed; this entry exists to keep the next
diagnosis from going down the credentials path.

## Context

Pushing git lanes (`git push` over SSH) failed in two ways in one session:

```
fatal: Could not read from remote repository.
Please make sure you have the correct access rights
and the repository exists.
```

and, on other attempts, an SSH keepalive `Broken pipe` / timeout. The
"access rights" wording is git's generic message for **any** failure of the
SSH command, not a verdict on keys. The natural first diagnosis — expired
key, dropped agent, wrong account — is wrong here.

## Phenomenon and isolation

```
$ ssh -T git@github.com
zsh:1: command not found: _kaku_wrapped_ssh      # function body absent

$ /usr/bin/ssh -T git@github.com
Hi cklxx! You've successfully authenticated ...   # credentials fine
```

The bare `ssh` resolved to a shell FUNCTION, not the binary:

```
$ type ssh
ssh is a shell function from
  ~/.claude/shell-snapshots/snapshot-zsh-*.sh
ssh () { _kaku_wrapped_ssh "$@"; }
```

The snapshot file defined the `ssh` wrapper (its line 1548) but did not
contain the `_kaku_wrapped_ssh` body it calls. The body lives in the Kaku
shell integration (`~/.config/kaku/zsh/kaku.zsh:609`); the Claude shell
snapshot captured the wrapper without sourcing the implementation. Git
invokes `$GIT_SSH_COMMAND` / `ssh`, hit the dangling function, the function
failed `command-not-found`, and git reported its standard
cannot-read-from-remote / access-rights text.

The user's `~/.zshrc` already contains a guard for exactly this:

```sh
# Fall back to the real ssh binary when the wrapper body isn't present.
if typeset -f ssh >/dev/null 2>&1 && ! typeset -f _kaku_wrapped_ssh >/dev/null 2>&1; then
    unset -f ssh
fi
```

It runs at interactive-shell start, but the agent's non-interactive command
shell sources the Claude snapshot, where the dangling function is defined
without that guard taking effect in the non-interactive context — so the
failure appeared in git/push invocations but not necessarily in a fresh
interactive terminal.

## Fix / workaround

Force the real binary for the failing invocation; nothing about keys or the
agent needed changing:

```sh
GIT_SSH_COMMAND="/usr/bin/ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
  git push ...
```

The keepalive options also address the separate, genuine long-upload idle
drop (the 5-minute `tn`/ssh cap documented elsewhere). No repo code is
implicated; no credentials were rotated.

## Rule

When a git-over-SSH push prints "correct access rights", do not start with
keys. Check what `ssh` actually is in THAT shell: `type ssh` — a shell
function or alias that shadows `/usr/bin/ssh` fails identically to an auth
rejection at git's error-reporting layer. Confirm credentials with the
binary explicitly: `/usr/bin/ssh -T git@github.com`. A wrapper captured
without its implementation is a shell-environment defect, and the generic
git wording must not be read as a verdict on the key.
