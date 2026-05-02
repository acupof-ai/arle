---
name: Do not commit user-specific absolute paths in docs
description: Replace /Users/<name>/... style paths with repo ids, $HOME-based paths, or generic placeholders before commit
type: feedback
---

User-facing or committed docs must not contain workstation-specific absolute
paths like `/Users/<name>/...`.

**Why:** Those paths leak machine-local details, do not reproduce on another
host, and the user explicitly called this out during the Metal DFlash work.

**How to apply:** Before committing docs or wins entries, grep for
`/Users/` and replace:

1. local model references with the Hugging Face repo id when that is enough
2. required local-cache examples with `$HOME/.cache/...`
3. any other machine-local location with a neutral placeholder
