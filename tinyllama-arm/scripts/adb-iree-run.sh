#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/adb-iree-run.sh <module.vmfb> [iree-run-module flags...]" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
serial="${ADB_SERIAL:?set ADB_SERIAL=<board-ip>:5555}"
remote_dir="${REMOTE_DIR:-/tmp/skainet-tinyllama}"
local_vmfb="$1"
shift

remote_vmfb="$remote_dir/$(basename "$local_vmfb")"

# Connect to the board, self-healing a stale local adb server (see adb-board-run.sh).
adb_connect() {
  local out
  out="$(adb connect "$serial" 2>&1)" || true
  if printf '%s' "$out" | grep -Eq 'connected to|already connected'; then
    return 0
  fi
  echo "adb connect failed ($out); restarting adb server and retrying" >&2
  adb kill-server >/dev/null 2>&1 || true
  adb start-server >/dev/null 2>&1 || true
  adb connect "$serial" >/dev/null 2>&1 || true
}

adb_connect
adb -s "$serial" shell mkdir -p "$remote_dir"
adb -s "$serial" push "$local_vmfb" "$remote_vmfb" >/dev/null

if [[ $# -eq 0 ]]; then
  set -- --device=local-task:// --function=tinyllama_step --print_statistics=true
fi

adb -s "$serial" shell iree-run-module --module="$remote_vmfb" "$@"

