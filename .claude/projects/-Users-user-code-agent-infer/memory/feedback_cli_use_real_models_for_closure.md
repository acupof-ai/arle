# Rule

For CLI closure work in this repo, prefer real local models and the existing
train/eval flows over mocks when the user says the models are available.

# Why

The user explicitly clarified that mock-based testing was unnecessary here:
the machine already has usable models, and CLI work can be validated with
real end-to-end runs.

# Preventive action

- Before designing CLI tests, check whether local models or a tiny trainable
  model path already exist.
- Treat mock engines as a fallback only when real-model closure is impossible
  or prohibitively expensive.
- When the user says "models are available", bias toward live CLI smoke tests
  and real command execution in the verification phase.
