#!/usr/bin/env bash
# Synchronize the current GitHub commit to ModelScope without hiding failures.
set -euo pipefail

: "${MODELSCOPE_REMOTE:?MODELSCOPE_REMOTE is required}"
branch="${MODELSCOPE_BRANCH:-master}"
remote_ref="refs/remotes/modelscope/${branch}"

if git remote get-url modelscope >/dev/null 2>&1; then
  git remote set-url modelscope "$MODELSCOPE_REMOTE"
else
  git remote add modelscope "$MODELSCOPE_REMOTE"
fi

fetch_ok=false
for attempt in 1 2 3; do
  if git fetch --no-tags modelscope "+refs/heads/${branch}:${remote_ref}"; then
    fetch_ok=true
    break
  fi
  echo "ModelScope fetch failed (attempt ${attempt}/3); retrying..." >&2
  sleep "$((attempt * 2))"
done

if [ "$fetch_ok" != true ]; then
  echo "Unable to fetch ModelScope ${branch} after three attempts." >&2
  exit 1
fi

git show-ref --verify --quiet "$remote_ref"
git merge --no-edit -X ours "modelscope/${branch}"

push_ok=false
for attempt in 1 2 3; do
  if git push modelscope "HEAD:refs/heads/${branch}"; then
    push_ok=true
    break
  fi
  echo "ModelScope push failed (attempt ${attempt}/3); retrying..." >&2
  sleep "$((attempt * 2))"
  git fetch --no-tags modelscope "+refs/heads/${branch}:${remote_ref}"
  git merge --no-edit -X ours "modelscope/${branch}"
done

if [ "$push_ok" != true ]; then
  echo "Unable to push to ModelScope ${branch} after three attempts." >&2
  exit 1
fi
