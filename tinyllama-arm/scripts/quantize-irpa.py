#!/usr/bin/env python3
"""B2: weight-only int8 quantization of the exported TinyLlama IREE artifact.

Takes the FP32 export pair (StableHLO MLIR + flat F32 safetensors) and emits an
int8 pair that fits the 1.9 GB board: every large rank-2 matmul weight becomes an
i8 global + a per-row F32 scale, with a dequant (convert + broadcast-multiply)
spliced in at its load site. Small tensors (RoPE tables, RMSNorm gains) stay F32.

  python3 scripts/quantize-irpa.py \
      build/iree/tinyllama_iree.mlir build/iree/tinyllama_weights.safetensors \
      build/iree/tinyllama_iree_int8.mlir build/iree/tinyllama_weights_int8.safetensors

Then: iree-convert-parameters the int8 safetensors -> .irpa, iree-compile the int8 MLIR.
"""
import json, re, struct, sys
import numpy as np

MIN_NUMEL = 4096  # only quantize matrices this large; norms/RoPE stay F32

def read_safetensors(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        blob = f.read()
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        out[name] = (meta["dtype"], meta["shape"], blob[s:e])
    return out

def write_safetensors(path, tensors):
    # tensors: name -> (dtype_str, shape_list, raw_bytes)
    hdr, blob, off = {}, bytearray(), 0
    for name, (dt, shape, raw) in tensors.items():
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blob += raw
        off += len(raw)
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(blob)

def main():
    mlir_in, st_in, mlir_out, st_out = sys.argv[1:5]
    # weight global shapes from the MLIR
    decl = re.compile(r"^(\s*)util\.global private @(t\d+) = (#flow\.parameter\.named<[^>]*>) : tensor<([0-9x]+)xf32>\s*$")
    shapes = {}
    for line in open(mlir_in):
        m = decl.match(line)
        if m:
            shapes[m.group(2)] = [int(x) for x in m.group(4).split("x")]

    st = read_safetensors(st_in)
    quant = {}   # tN -> (rows, cols) for those we quantized
    out_t = {}
    for name, (dt, shape, raw) in st.items():
        dims = shapes.get(name)
        numel = int(np.prod(dims)) if dims else len(raw) // 4
        f32 = np.frombuffer(raw, dtype="<f4")
        if dims and len(dims) == 2 and numel >= MIN_NUMEL:
            rows, cols = dims
            w = f32.reshape(rows, cols)
            scale = np.maximum(np.abs(w).max(axis=1) / 127.0, 1e-12).astype("<f4")  # per row
            q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
            out_t[name] = ("I8", [numel], q.tobytes())
            out_t[name + "_scale"] = ("F32", [rows], scale.tobytes())
            quant[name] = (rows, cols)
        else:
            out_t[name] = ("F32", [numel], f32.astype("<f4").tobytes())
    write_safetensors(st_out, out_t)

    # rewrite the MLIR: decls -> i8 + scale global; load sites -> dequant
    load = re.compile(r"^(\s*)%(\w+) = util\.global\.load @(t\d+) : tensor<([0-9x]+)xf32>\s*$")
    seq = 0
    lines_out = []
    for line in open(mlir_in):
        m = decl.match(line)
        if m and m.group(2) in quant:
            ind, n, parm, sh = m.group(1), m.group(2), m.group(3), m.group(4)
            rows = quant[n][0]
            # scale param: same scope "model", name "<tN>_scale"
            scope = re.search(r'named<"([^"]+)"::"[^"]+">', parm).group(1)
            lines_out.append(f"{ind}util.global private @{n} = {parm} : tensor<{sh}xi8>\n")
            lines_out.append(f'{ind}util.global private @{n}_scale = #flow.parameter.named<"{scope}"::"{n}_scale"> : tensor<{rows}xf32>\n')
            continue
        m = load.match(line)
        if m and m.group(3) in quant:
            ind, res, n, sh = m.group(1), m.group(2), m.group(3), m.group(4)
            rows = quant[n][0]
            seq += 1
            qv, sv, fv, bv = f"%qz{seq}", f"%sz{seq}", f"%fz{seq}", f"%bz{seq}"
            lines_out.append(f"{ind}{qv} = util.global.load @{n} : tensor<{sh}xi8>\n")
            lines_out.append(f"{ind}{sv} = util.global.load @{n}_scale : tensor<{rows}xf32>\n")
            lines_out.append(f"{ind}{fv} = stablehlo.convert {qv} : (tensor<{sh}xi8>) -> tensor<{sh}xf32>\n")
            lines_out.append(f"{ind}{bv} = \"stablehlo.broadcast_in_dim\"({sv}) <{{broadcast_dimensions = array<i64: 0>}}> : (tensor<{rows}xf32>) -> tensor<{sh}xf32>\n")
            lines_out.append(f"{ind}%{res} = stablehlo.multiply {fv}, {bv} : tensor<{sh}xf32>\n")
            continue
        lines_out.append(line)
    open(mlir_out, "w").writelines(lines_out)

    nb = sum(len(v[2]) for v in out_t.values())
    print(f"quantized {len(quant)} matrices to int8; kept {len(out_t)-2*len(quant)} F32")
    print(f"new safetensors: {nb/1e9:.2f} GB  ({st_out})")
    print(f"new MLIR: {mlir_out}")

if __name__ == "__main__":
    main()
