# Repository Instructions

This file and `CLAUDE.md` must stay aligned. `AGENTS.md` is the short operational contract for coding agents; `CLAUDE.md` may include additional project context, but must not contradict this file.

## Git Workflow

- Use GitHub Flow for every non-trivial task.
- Do not work directly on `main` unless the user explicitly requests it.
- Before making code changes, create or switch to a task branch named `codex/<short-task-name>` from the current default integration branch.
- At task completion, if files changed, review the diff, create a commit, and push the current branch to `origin`.
- Use a concise commit message in the form `<scope>: <summary>` when possible.
- Do not rewrite history unless the user explicitly asks for it.
- If push fails because credentials, remote policy, or network access are unavailable, report that clearly in the final response.

## Experiment Workflow

- Use a separate phase for each materially new experiment hypothesis or evaluation setup.
- Keep repeated parameter variations within the same phase as `PN-v01`, `PN-v02`, etc.
- Prefer `tmux` for long-running training, sweeps, or GPU-waiting jobs.
- Do not stop or overwrite another running experiment unless the user explicitly asks.
- Keep generated datasets and model artifacts out of commits unless the user explicitly asks to version them.

## Phase Documentation

When a phase is complete, documentation is mandatory before closing the task:

- Update or create `results/<phase-dir>/strategy.md` with the hypothesis, changes, experiment table, best result, and conclusion.
- Create or update `docs/tasks/phaseNN-summary.md` for each completed phase.
- Include the final metric basis used for the phase, for example `test_video`, `test_event_video`, threshold selection level, and confusion matrix.
- If a phase is still running, do not write it as completed; document only the current status or wait for completion.
- Keep each completed phase in a separate docs file instead of merging many completed phases into one large summary.

## Completion Checklist

- Run the most relevant verification you can for the change.
- Summarize the verification result in the final response.
- If the branch was newly created for the task, mention the branch name in the final response.
