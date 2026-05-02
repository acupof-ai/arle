# Rule

The landing-page statusline is navigation, not decoration.

# Why

The user explicitly pointed out that the bottom bar had become non-clickable
and had no interaction effect. Rendering it as `aria-hidden` spans broke a
useful part of the site's original design.

# Preventive action

- Keep the bottom statusline as real links where the labels imply navigation.
- Do not mark the whole statusline `aria-hidden` when it contains actions.
- Preserve the original terminal-statusline look, but keep hover and focus
  states visible.
