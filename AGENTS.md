# Repository Instructions

## Git Workflow

- Use GitHub Flow for every non-trivial task.
- Do not work directly on `main` unless the user explicitly requests it.
- Before making code changes, create or switch to a task branch named `codex/<short-task-name>` from the current default integration branch.
- At task completion, if files changed, review the diff, create a commit, and push the current branch to `origin`.
- Use a concise commit message in the form `<scope>: <summary>` when possible.
- Do not rewrite history unless the user explicitly asks for it.
- If push fails because credentials, remote policy, or network access are unavailable, report that clearly in the final response.

## Completion Checklist

- Run the most relevant verification you can for the change.
- Summarize the verification result in the final response.
- If the branch was newly created for the task, mention the branch name in the final response.
