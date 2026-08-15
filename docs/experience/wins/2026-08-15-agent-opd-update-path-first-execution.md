# Agent-OPD parameter-update path executed for the first time

**Context.** Every prior agent-OPD run (0.8B student) produced all-fail rollouts
on the ansible staged task — reward variance zero, group discarded, update path
never entered. Switching the student to ThinkingCap-Qwen3.6-27B-FP8 (single
GPU, `--lora-target-set attention-qv`, `--cc-timeout 900`, `--max-update-seq 0`)
was the unblock.

**What worked.**
- All 4 claude rollouts passed (reward=1.0, 5–6 turns, 84–88 s wall each): the
  cc-harness lane (local serve on :8000, `ANTHROPIC_BASE_URL`, dummy key) is
  fully functional; earlier "cc timeout" failures were the 0.8B student's
  capability, not the harness.
- The all-pass group entered supervised writeback: 4 masked-writeback update
  steps completed at seq 21k–23k (forward 100–111 s, backward 218–298 s each),
  losses finite and small (last completed step 0.0535).
- The run was externally SIGKILLed (exit 137) during the 5th writeback step;
  node-side trace shows no OOM record — cause unknown. The round report did not
  print, but the quantity under test (update path executes on real rollouts)
  was observed directly.

**Rule.** For a binary pass reward, group variance needs a student strong enough
to pass sometimes — scale the student to the task before touching timeouts or
task difficulty.
