#!/usr/bin/env bash
# On-board greedy decode of TinyLlama on the Synaptics SL2619 via IREE.
#
# The board runs each prefill step with its own `iree-run-module` (local-task:// CPU,
# or torq:// once a Torq-compiled vmfb exists); this host orchestrates — builds the
# padded input, reads back the logits, argmaxes the last real position, appends, repeats.
# The fixed-seq prefill graph has no KV cache, so each step re-runs the full L-position
# forward and the board reloads the .irpa — correct but slow on the 2-core A55. Generate
# a handful of tokens to prove on-board generation; KV-cache decode is the perf follow-up.
#
# Requires (host): adb, python3 + numpy. Requires (board): iree-run-module (IREE 3.11 —
# the int8 vmfb is built with 3.11; the board runtime MUST be 3.11, not the old 3.10).
#
#   ADB_SERIAL=<BOARD_IP>:5555 scripts/decode-board.sh \
#     build/iree/int8_aarch64.vmfb build/iree/int8.irpa 8 1,5462,303,291 4
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
serial="${ADB_SERIAL:?set ADB_SERIAL=<board-ip>:5555}"
board_dir="${BOARD_DIR:-/home/skainet-tinyllama/iree}"   # ext4, NOT /tmp (tmpfs eats RAM)
vmfb="$1"; irpa="$2"; L="$3"; prompt="$4"; gen="${5:-4}"

adb_connect() { # self-heal a stale local adb server (see adb-iree-run.sh)
  local out; out="$(adb connect "$serial" 2>&1)" || true
  printf '%s' "$out" | grep -Eq 'connected to|already connected' && return 0
  adb kill-server >/dev/null 2>&1 || true; adb start-server >/dev/null 2>&1 || true
  adb connect "$serial" >/dev/null 2>&1 || true
}
adb() { command adb -s "$serial" "$@"; }

# The board's /usr/bin/iree-run-module is the old 3.10 runtime (rejects 3.11
# bytecode with "requires [Ch]"). The 3.11 runtime is pip-installed into the
# board's fcvenv — default to it; override with BOARD_IREE_RUN.
board_iree="${BOARD_IREE_RUN:-/home/root/fcvenv/bin/iree-run-module}"

adb_connect
if [[ "${SKIP_PUSH:-0}" != "1" ]]; then
  echo "### pushing artifacts to $board_dir (vmfb + 1.1 GB irpa — one time)"
  command adb -s "$serial" shell mkdir -p "$board_dir"
  command adb -s "$serial" push "$vmfb" "$board_dir/m.vmfb" >/dev/null
  command adb -s "$serial" push "$irpa" "$board_dir/p.irpa" >/dev/null
fi

IFS=',' read -ra tokens <<< "$prompt"
gen_ids=()
for ((s=0; s<gen; s++)); do
  k=${#tokens[@]}
  (( k >= L )) && { echo "### reached graph seqlen $L; stop"; break; }
  # pad current window to L with 0
  pad=("${tokens[@]}"); for ((i=k; i<L; i++)); do pad+=(0); done
  csv=$(IFS=,; echo "${pad[*]}")
  t0=$(date +%s.%N)
  command adb -s "$serial" shell "$board_iree --device=local-task:// \
    --module=$board_dir/m.vmfb --parameters=model=$board_dir/p.irpa \
    --function=tinyllama --input=1x${L}xi32=$csv --output=@$board_dir/out.npy" >/dev/null
  t1=$(date +%s.%N)
  echo "### step $s: board iree-run-module wall $(python3 -c "print(f'{$t1-$t0:.1f}')") s (incl. irpa load + adb)"
  command adb -s "$serial" pull "$board_dir/out.npy" /tmp/board_out.npy >/dev/null
  nxt=$(python3 - "$L" "$k" <<'PY'
import sys, numpy as np
L, k = int(sys.argv[1]), int(sys.argv[2])
v = np.load("/tmp/board_out.npy").reshape(1, L, -1)[0, k-1]
print(int(v.argmax()))
PY
)
  tokens+=("$nxt"); gen_ids+=("$nxt")
  echo "### step $s: next id=$nxt"
done
echo "PROMPT_IDS $prompt"
echo "GEN_IDS $(IFS=,; echo "${gen_ids[*]}")"
echo "ALL_IDS $(IFS=,; echo "${tokens[*]}")"
