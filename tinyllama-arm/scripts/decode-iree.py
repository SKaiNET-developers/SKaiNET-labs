#!/usr/bin/env python3
"""Greedy decode loop over the compiled TinyLlama IREE vmfb + external .irpa params.

The exported graph is fixed-seq prefill (func @tinyllama: [1,L]i32 -> [1,L,32000]f32,
no KV cache). We generate autoregressively by re-running the prefill on the growing
token window padded to L and reading argmax at the last real position (causal mask
makes pos k depend only on tokens 0..k, so right-padding is safe). Loads the module
and params ONCE, then loops — a genuine in-process decode loop.

  python3 scripts/decode-iree.py --vmfb build/iree/int8_host.vmfb \
      --irpa build/iree/int8.irpa --seqlen 8 --prompt 1,5462,303,291 --gen 4
"""
import argparse, sys
import numpy as np
import iree.runtime as rt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmfb", required=True)
    ap.add_argument("--irpa", required=True)
    ap.add_argument("--scope", default="model")
    ap.add_argument("--device", default="local-task")
    ap.add_argument("--seqlen", type=int, required=True, help="graph fixed seq L")
    ap.add_argument("--prompt", required=True, help="comma-separated token ids")
    ap.add_argument("--gen", type=int, default=8, help="tokens to generate")
    ap.add_argument("--func", default="tinyllama")
    a = ap.parse_args()
    L = a.seqlen
    tokens = [int(x) for x in a.prompt.split(",")]
    assert len(tokens) <= L, f"prompt ({len(tokens)}) > seqlen ({L})"

    # Load module + external params once (SystemContext adds the HAL module from config).
    config = rt.Config(a.device)
    inst = config.vm_instance
    idx = rt.ParameterIndex()
    idx.load(a.irpa)
    provider = idx.create_provider(scope=a.scope)
    param_mod = rt.create_io_parameters_module(inst, provider)
    main_mod = rt.VmModule.mmap(inst, a.vmfb)
    sysctx = rt.SystemContext(vm_modules=[param_mod, main_mod], config=config)
    func = sysctx.modules.module[a.func]

    generated = []
    for step in range(a.gen):
        if len(tokens) >= L:
            print(f"[decode] reached graph seqlen {L}; stop", file=sys.stderr)
            break
        inp = np.array([tokens + [0] * (L - len(tokens))], dtype=np.int32)  # [1, L]
        out = func(inp)
        logits = np.asarray(out).reshape(1, L, -1)[0, len(tokens) - 1]
        nxt = int(np.argmax(logits))
        tokens.append(nxt)
        generated.append(nxt)
        print(f"[decode] step {step}: next id={nxt} logit={float(logits[nxt]):.3f}")
    print("PROMPT_IDS", a.prompt)
    print("GEN_IDS", ",".join(map(str, generated)))
    print("ALL_IDS", ",".join(map(str, tokens)))

if __name__ == "__main__":
    main()
