#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/adb-iree-run.sh <module.vmfb> [iree-run-module flags...]" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
serial="${ADB_SERIAL:-192.168.3.26:5555}"
remote_dir="${REMOTE_DIR:-/tmp/skainet-tinyllama}"
local_vmfb="$1"
shift

remote_vmfb="$remote_dir/$(basename "$local_vmfb")"

adb connect "${serial%:5555}" >/dev/null || true
adb -s "$serial" shell mkdir -p "$remote_dir"
adb -s "$serial" push "$local_vmfb" "$remote_vmfb" >/dev/null

if [[ $# -eq 0 ]]; then
  set -- --device=local-task:// --function=tinyllama_step --print_statistics=true
fi

adb -s "$serial" shell iree-run-module --module="$remote_vmfb" "$@"

