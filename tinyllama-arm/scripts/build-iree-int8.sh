#!/usr/bin/env bash
# Build the full int8 TinyLlama IREE artifact set, end to end:
#   export (SKaiNET) -> quantize (int8 weight-only) -> .irpa -> vmfb (host + aarch64board).
# Produces, under build/iree/:
#   tinyllama_iree_int8.mlir, tinyllama_weights_int8.safetensors,
#   int8.irpa (~1.1 GB), int8_host.vmfb (local decode), int8_aarch64.vmfb (board).
#
#   SEQ=8 scripts/build-iree-int8.sh        # graph fixed-seq length (decode window)
#
# Requires: gradle composite (-PuseLocalSkainet), the GGUF model, and the IREE 3.11
# docker image (skainet-iree:3.11.0 = iree-compile+convert; override with IREE_IMAGE).
set -euo pipefail
cd "$(dirname "$0")/.."
unset GRADLE_USER_HOME || true
SEQ="${SEQ:-8}"
IMG="${IREE_IMAGE:-skainet-iree:3.11.0}"
B=build/iree

echo "### 1/4 export FP32 graph + weights (seq=$SEQ)"
./gradlew -PuseLocalSkainet=true :export-hlo:exportLlamaIree -Pseq="$SEQ" --console=plain \
  | grep -E "Outputs:|EXTPARAMS|BUILD"

echo "### 2/4 weight-only int8 quantize"
python3 scripts/quantize-irpa.py \
  "$B/tinyllama_iree.mlir" "$B/tinyllama_weights.safetensors" \
  "$B/tinyllama_iree_int8.mlir" "$B/tinyllama_weights_int8.safetensors"

echo "### 3/4 .irpa (int8)  + 4/4 compile vmfb (host + aarch64)"
docker run --rm -m18g -v "$PWD":/work -w /work --entrypoint sh "$IMG" -c "
  set -e
  iree-convert-parameters --parameters=$B/tinyllama_weights_int8.safetensors --output=$B/int8.irpa
  iree-compile $B/tinyllama_iree_int8.mlir --iree-hal-target-backends=llvm-cpu -o $B/int8_host.vmfb
  iree-compile $B/tinyllama_iree_int8.mlir --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu --iree-llvmcpu-target-cpu=generic \
    -o $B/int8_aarch64.vmfb
"
echo "### done. artifacts:"
ls -lh "$B"/int8.irpa "$B"/int8_host.vmfb "$B"/int8_aarch64.vmfb | awk '{print "  "$5"  "$9}'
echo "### host decode:  python3 scripts/decode-iree.py --vmfb $B/int8_host.vmfb --irpa $B/int8.irpa --seqlen $SEQ --prompt 1,5462,303,291 --gen 4"
echo "### board decode: ADB_SERIAL=<BOARD_IP>:5555 scripts/decode-board.sh $B/int8_aarch64.vmfb $B/int8.irpa $SEQ 1,5462,303,291 4"
