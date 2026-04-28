# AGENTS

## Python Tooling

- Force well-typed Python.  Avoid `isinstance` whenever possible.
- Run static type checks with `uv run pyright`.
- Run tests with `uv run --with pytest python -m pytest ...` to use uv's ephemeral dependency mode.

## Git Commit Workflow

- If the user asks you to commit, first run `uv run pyright` for the whole project, and commit only when there are no errors.
- When running git commands, switch `workdir` to the target repository instead of using `git -C`.
- When making a bug-fix commit, if there is already a recent commit that attempted to fix the same bug, propose squashing the commits. If the user approves, perform the squash.
