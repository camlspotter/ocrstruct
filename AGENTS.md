# Repository Notes

## Git

- Prefer running git commands from the target repository via `workdir` instead of `git -C ...` when possible.
- Avoid `git -C` unless there is a clear need for it, because it can trigger extra permission prompts in this environment.
