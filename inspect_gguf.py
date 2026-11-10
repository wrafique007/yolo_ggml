#!/usr/bin/env python3
"""
Print the layer list stored in a yolov7 GGUF file and check that the shapes fit together,
without running the C++ engine. Use it right after converting.

  python inspect_gguf.py yolov7-f32.gguf            # table of layers + shapes, errors marked
  python inspect_gguf.py yolov7-f32.gguf --size 320

Shapes are printed PyTorch-style as [C, H, W]. The tool reports the first layer whose
inputs don't fit (wrong 'from' index, channel mismatch, missing tensor...) and exits with 1.
"""

import argparse
import sys

import gguf


def parse(desc):
    kind, *kvs = desc.split()
    d = {"kind": kind}
    for kv in kvs:
        k, v = kv.split("=", 1)
        d[k] = [int(x) for x in v.split(",")] if k == "from" else v
    for k in ("k", "s", "p", "scale"):
        if k in d:
            d[k] = int(d[k])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    r = gguf.GGUFReader(args.gguf)
    layers = r.fields["yolov7.layers"].contents()
    # GGUFReader shapes are in ggml order: conv weight = [KW, KH, IC, OC]
    tshape = {t.name: [int(x) for x in t.shape] for t in r.tensors}

    shapes = []                                   # [C, H, W] per layer, None for detect
    errors = 0

    def err(i, desc, msg, srcs):
        nonlocal errors
        errors += 1
        print(f"\n*** ERROR at layer {i}: {desc}\n    {msg}")
        for f, s in srcs:
            print(f"    from {f:>4}{' (network input)' if f == -1 else ''} -> {s}")

    for i, desc in enumerate(layers):
        L = parse(desc)
        srcs = []
        bad = False
        for f in L.get("from", []):
            if f == -1:
                srcs.append((f, [3, args.size, args.size]))
            elif f < -1 or f >= i:
                err(i, desc, f"'from' index {f} is invalid (must be -1 for the input, or 0..{i - 1}). "
                             "Relative indices like -1/-2 from the YAML must be converted to absolute ones.", srcs)
                bad = True
                break
            elif shapes[f] is None:
                err(i, desc, f"'from' {f} is the detect layer, which has no single output", srcs)
                bad = True
                break
            else:
                srcs.append((f, shapes[f]))
        if bad:
            break

        kind, out = L["kind"], None
        if kind == "conv":
            w = tshape.get(f"l{i}.weight")
            if w is None:
                err(i, desc, f"tensor l{i}.weight is missing (tensor names must use this layer's flat index)", srcs)
                break
            kw, kh, ic, oc = w
            c, h, wd = srcs[0][1]
            if ic != c:
                err(i, desc, f"conv weight l{i}.weight expects {ic} input channels, but its input has {c}", srcs)
                break
            k, s, p = L["k"], L["s"], L["p"]
            if (kw, kh) != (k, k):
                err(i, desc, f"metadata says k={k} but l{i}.weight is {kw}x{kh}", srcs)
                break
            out = [oc, (h + 2 * p - k) // s + 1, (wd + 2 * p - k) // s + 1]
        elif kind == "maxpool":
            c, h, wd = srcs[0][1]
            k, s, p = L["k"], L["s"], L["p"]
            out = [c, (h + 2 * p - k) // s + 1, (wd + 2 * p - k) // s + 1]
        elif kind == "upsample":
            c, h, wd = srcs[0][1]
            out = [c, h * L["scale"], wd * L["scale"]]
        elif kind in ("concat", "add"):
            hw = {tuple(s[1:]) for _, s in srcs}
            if len(hw) != 1:
                err(i, desc, f"{kind} inputs have different spatial sizes", srcs)
                break
            if kind == "add":
                if len({s[0] for _, s in srcs}) != 1:
                    err(i, desc, "add inputs have different channel counts", srcs)
                    break
                out = list(srcs[0][1])
            else:
                out = [sum(s[0] for _, s in srcs)] + list(srcs[0][1][1:])
        elif kind == "detect":
            for j, (f, s) in enumerate(srcs):
                ic = tshape[f"det{j}.weight"][2]
                if ic != s[0]:
                    err(i, desc, f"detect head {j} expects {ic} input channels but gets {s[0]}", srcs)
                    break
            print(f"{i:4}  {desc:<52} -> heads {[s for _, s in srcs]}")
            shapes.append(None)
            continue
        else:
            err(i, desc, f"unknown layer type '{kind}' (yolov7.cpp has no code for it)", srcs)
            break

        shapes.append(out)
        print(f"{i:4}  {desc:<52} -> {out}")

    if errors:
        sys.exit(1)
    print(f"\nOK: {len(layers)} layers, all shapes consistent at {args.size}x{args.size}")


if __name__ == "__main__":
    main()
