#!/usr/bin/env bash
# Pre-publish gate: fail if the working tree leaks the board address, an author home
# path, the remote-root login, or the private Git hostname. Run from anywhere; checks
# the whole repo. Exit 0 = clean, exit 1 = leakage listed on stdout.
set -euo pipefail

cd "$(dirname "$0")/.."

# The one place these strings are allowed to exist: this script.
# Only git-tracked files matter — publishing goes through git, so gitignored local
# state (e.g. .claude/settings.local.json) is out of scope by construction.
pattern='192\.168\.|/Users/[A-Za-z0-9]|root@|gitea-kopcek'

hits=$(git ls-files -z \
  | grep -zv 'check-no-leakage\.sh' \
  | xargs -0 grep -InE "$pattern" -- \
  || true)

if [[ -n "$hits" ]]; then
  printf '%s\n' "$hits"
  echo "FAIL: leakage found ($(printf '%s\n' "$hits" | wc -l | tr -d ' ') hits)" >&2
  exit 1
fi

echo "OK: no leakage"
