# Keep Rule Docs Full Text

## Context

I interpreted the existing "AGENTS is canonical" rule too aggressively and
collapsed `CLAUDE.md` files into thin pointers while updating rule docs.

## Rule

When the user asks to update rule documents, preserve full-text rule content in
every rule doc unless they explicitly ask to collapse or deduplicate them.
`AGENTS.md` may remain the canonical source, but sibling `CLAUDE.md` files
should stay as complete synced documents rather than being converted to pointer
stubs by default.
