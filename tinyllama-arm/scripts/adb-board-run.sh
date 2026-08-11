#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
serial="${ADB_SERIAL:?set ADB_SERIAL=<board-ip>:5555}"
remote_dir="${REMOTE_DIR:-/tmp/skainet-tinyllama}"
remote_bin="$remote_dir/tinyllama-skainet"
local_bin="$root_dir/eager/build/bin/linuxArm64/releaseExecutable/tinyllama-skainet.kexe"

cd "$root_dir"

if [[ ! -x "$local_bin" || "${FORCE_BUILD:-0}" == "1" ]]; then
  if [[ -z "${GRADLE_CMD:-}" ]]; then
    if command -v gradle >/dev/null 2>&1; then
      GRADLE_CMD=gradle
    else
      GRADLE_CMD=./gradlew
      export GRADLE_USER_HOME="${GRADLE_USER_HOME:-$root_dir/.gradle-home}"
    fi
  fi
  "$GRADLE_CMD" :eager:linkReleaseExecutableLinuxArm64
fi

# Connect to the board, self-healing a stale local adb server. A board that
# rebooted (e.g. after an OOM) leaves the adb daemon caching the old dead
# connection, so `adb connect` keeps returning "No route to host" even though
# the port is reachable. `adb kill-server` clears that state; retry once.
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
adb -s "$serial" push "$local_bin" "$remote_bin" >/dev/null
adb -s "$serial" shell chmod +x "$remote_bin"

# Yocto/Poky on the SL2619 image ships libcrypt.so.2, while the Kotlin/Native
# linuxArm64 runtime declares libcrypt.so.1. Keep the compatibility shim local
# to /tmp instead of modifying the root filesystem.
adb -s "$serial" shell ln -sf /usr/lib/libcrypt.so.2 "$remote_dir/libcrypt.so.1"

if [[ $# -eq 0 ]]; then
  set -- help
fi

remote_quote() {
  printf "'"
  printf "%s" "$1" | sed "s/'/'\\\\''/g"
  printf "'"
}

remote_env="LD_LIBRARY_PATH=$(remote_quote "$remote_dir")"
# Forward the opt-in profiling gate to the board process when set locally.
if [[ -n "${SKAINET_PROFILE:-}" ]]; then
  remote_env+=" SKAINET_PROFILE=$(remote_quote "$SKAINET_PROFILE")"
fi
remote_cmd="env $remote_env $(remote_quote "$remote_bin")"
for arg in "$@"; do
  remote_cmd+=" $(remote_quote "$arg")"
done

adb -s "$serial" shell "$remote_cmd"
