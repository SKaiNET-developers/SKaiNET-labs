#!/usr/bin/env bash
# B1 bisect: find the seqLen at which iree-compile --compile-to=flow starts crashing
# on the REAL TinyLlama graph. Target-independent (flow stage), so host-target is irrelevant.
set -u
cd "$(dirname "$0")/.."
unset GRADLE_USER_HOME
BIS=build/iree/bisect
mkdir -p "$BIS"
SEQS="${SEQS:-1 2 3 4}"
IMG=skainet-iree:3.11.0

echo "### seqLen bisect: seqs = $SEQS"
for s in $SEQS; do
  echo
  echo "================ seqLen=$s ================"
  echo "--- export (gradle) ---"
  ./gradlew -PuseLocalSkainet=true :export-hlo:exportLlamaIree -Pseq="$s" --console=plain 2>&1 \
    | grep -E "Outputs:|EXTPARAMS|BUILD (SUCCESSFUL|FAILED)|error:" || true
  if [ ! -f build/iree/tinyllama_iree.mlir ]; then
    echo "seq=$s EXPORT_FAILED (no mlir)"; continue
  fi
  cp build/iree/tinyllama_iree.mlir "$BIS/seq$s.mlir"
  echo "--- iree-compile --compile-to=flow ---"
  docker run --rm -m18g -v "$PWD":/work -w /work --entrypoint sh "$IMG" -c \
    "iree-compile $BIS/seq$s.mlir --compile-to=flow -o /tmp/flow_$s.mlir --mlir-disable-threading >/tmp/err_$s 2>&1; echo EXIT=\$?; tail -3 /tmp/err_$s"
  rc=$?
  echo "seq=$s docker_rc=$rc"
done
echo
echo "### bisect done. MLIRs kept under $BIS/"
