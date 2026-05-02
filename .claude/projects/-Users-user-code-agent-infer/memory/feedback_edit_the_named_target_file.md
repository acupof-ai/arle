# Rule

When the user describes a relationship between files, edit the file they named
as the target instruction file, not the neighboring file that seems easier to
change.

# Why

The user asked for `AGENTS.md` to encode that `CLAUDE.md` should point at it.
Editing `CLAUDE.md` directly satisfied the spirit but changed the wrong file.

# Preventive action

- Parse requests into two parts before editing: `file to change` and
  `file/path to reference`.
- For phrasing like "`CLAUDE.md` 内容链接到 `AGENTS.md`", default the edit target
  to the file the user later clarifies, not the referenced destination.
- State the exact file to be edited in the first status update before patching.
