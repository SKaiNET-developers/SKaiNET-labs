#!/usr/bin/env bash
# B1 logit parity: compiled IREE TinyLlama vmfb vs eager SKaiNET (matches llama.cpp).
# Feeds the SAME fixed token prefill to both, compares top-10 next-token logits at the
# last position. Assumes the export ran (build/iree/tinyllama_iree.mlir + _weights.safetensors).
set -euo pipefail
cd "$(dirname "$0")/.."
unset GRADLE_USER_HOME || true
TOKENS="${PARITY_TOKENS:-1,5462,303,291,29901,1724,338,4323}"   # len must == export -Pseq
IMG="${IREE_IMAGE:-skainet-iree:3.11.0}"   # IREE 3.11: compile+run+convert+runtime
MLIR=build/iree/tinyllama_iree.mlir
SAFE=build/iree/tinyllama_weights.safetensors

echo "### tokens: $TOKENS"
N=$(echo "$TOKENS" | tr ',' '\n' | grep -c .)
echo "### IREE side ($IMG)"
docker run --rm -m18g -v "$PWD":/work -w /work --entrypoint sh "$IMG" -c "
  set -e
  iree-convert-parameters --parameters=$SAFE --output=/tmp/model.irpa >/dev/null 2>&1
  iree-compile $MLIR --iree-hal-target-backends=llvm-cpu -o /tmp/host.vmfb 2>/dev/null
  iree-run-module --module=/tmp/host.vmfb --parameters=model=/tmp/model.irpa \
    --function=tinyllama --input=1x${N}xi32=$TOKENS --output=@/tmp/logits.npy >/dev/null 2>&1
  python3 -c '
import numpy as np
v=np.load(\"/tmp/logits.npy\")[0,-1]
for i in np.argsort(v)[::-1][:10]: print(f\"[iree] id={int(i)} logit={float(v[i]):.3f}\")'
"

echo "### eager side (canonical fromGguf DEQUANTIZE_TO_FP32 + OptimizedLLMRuntime, matches llama.cpp)"
PARITY_CANON=1 PARITY_TOKENS="$TOKENS" \
  ./gradlew -PuseLocalSkainet=true -Pxmx=12g :bench:runJvm \
  --args='bench --variants eager-jvm --tokens 1 --ctx 64 --prompt "x"' --console=plain 2>/dev/null \
  | grep -E "parity:.*id=" | sed -E 's/\[parity:[^]]*\]/[eager]/'

echo "### match the two [iree]/[eager] id lists above (same ids, same order => parity)."
