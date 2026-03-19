#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <branch-name> <commit-message>" >&2
  exit 1
fi

branch_name="$1"
commit_message="$2"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "$branch_name" ]]; then
  git switch -c "$branch_name" 2>/dev/null || git switch "$branch_name"
fi

if [[ -z "$(git status --short)" ]]; then
  echo "no changes to commit"
  exit 0
fi

git add -A
git commit -m "$commit_message"
git push -u origin "$(git rev-parse --abbrev-ref HEAD)"
