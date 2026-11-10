#!/usr/bin/env python3
"""
Check the ggml engine against an independent numpy re-implementation of the
original PyTorch forward pass.

The reference runs straight from the .pt checkpoint, with BatchNorm and the
IDetect implicit layers applied separately (unfused), so it also checks the
fusion done by the converter. No torch needed.

  ./build/yolov7 -m yolov7-tiny-f32.gguf -i image.jpg --dump dumpdir
  python verify.py yolov7-tiny.pt dumpdir
"""

import argparse
import sys

import numpy as np

from convert_yolov7_to_gguf import (as_int, children, children_items, load_checkpoint_numpy, resolve_from, to_np)


def conv2d(x, w, b, stride, pad):
    # x [C,H,W], w [O,C,KH,KW] -> [O,OH,OW]
    c, h, wd = x.shape
    o, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh, ow = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
    s = xp.strides
    cols = np.lib.stride_tricks.as_strided(
        xp, shape=(c, kh, kw, oh, ow), strides=(s[0], s[1], s[2], s[1] * stride, s[2] * stride))
    y = np.tensordot(w, cols, axes=([1, 2, 3], [0, 1, 2]))
    return y + (b[:, None, None] if b is not None else 0)


def maxpool(x, k, s, p):
    c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p)), constant_values=-np.inf)
    oh, ow = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
    st = xp.strides
    win = np.lib.stride_tricks.as_strided(
        xp, shape=(c, oh, ow, k, k), strides=(st[0], st[1] * s, st[2] * s, st[1], st[2]))
    return win.max(axis=(3, 4))


def act(x, m):
    name = type(m).__name__
    if name == "LeakyReLU":
        return np.where(x > 0, x, x * float(m.negative_slope))
    if name == "SiLU":
        return x / (1 + np.exp(-x))
    if name == "ReLU":
        return np.maximum(x, 0)
    return x


def forward(model, x):
    outs, mods = [], children(model.model)
    for i, m in enumerate(mods):
        src = [x if j < 0 else outs[j] for j in resolve_from(m.f, i)]
        kind = type(m).__name__
        if kind == "Conv":
            c = m.conv
            y = conv2d(src[0], to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "bn" in m._modules:
                bn = m.bn
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]
            y = act(y, m.act)
        elif kind == "Concat":
            y = np.concatenate(src, axis=0)
        elif kind in ("MP", "SP"):
            y = maxpool(src[0], as_int(m.m.kernel_size), as_int(m.m.stride), as_int(m.m.padding))
        elif kind == "Upsample":
            sf = int(m.scale_factor)
            y = src[0].repeat(sf, axis=1).repeat(sf, axis=2)
        elif kind in ("Detect", "IDetect"):
            y = []
            convs = children(m.m)
            for j, s in enumerate(src):
                if kind == "IDetect":
                    s = s + to_np(children(m.ia)[j].implicit)[0]
                h = conv2d(s, to_np(convs[j].weight), to_np(convs[j].bias), 1, 0)
                if kind == "IDetect":
                    h = h * to_np(children(m.im)[j].implicit)[0]
                y.append(h)
        elif kind == "SPPCSPC":
            layers = children_items(m)

            c = layers['cv1'].conv
            m = layers['cv1']
            y = conv2d(src[0], to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "bn" in m._modules:
                bn = m.bn
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]
            y = act(y, m.act)

            c = layers['cv3'].conv
            m = layers['cv3']
            y = conv2d(y, to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "bn" in m._modules:
                bn = m.bn
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]
            y = act(y, m.act)

            c = layers['cv4'].conv
            m = layers['cv4']
            y = conv2d(y, to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "bn" in m._modules:
                bn = m.bn
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]
            x1 = act(y, m.act)

            maxpool_outputs = []
            maxpool2D = layers['m']._modules

            m = maxpool2D['0']
            y = maxpool(x1, as_int(m.kernel_size), as_int(m.stride), as_int(m.padding))
            maxpool_outputs.append(y)

            m = maxpool2D['1']
            y = maxpool(x1, as_int(m.kernel_size), as_int(m.stride), as_int(m.padding))
            maxpool_outputs.append(y)

            m = maxpool2D['2']
            y = maxpool(x1, as_int(m.kernel_size), as_int(m.stride), as_int(m.padding))
            maxpool_outputs.append(y)

            concat_result = np.concatenate([x1] + maxpool_outputs, axis=0)

            c = layers['cv5'].conv
            m = layers['cv5']
            y = conv2d(concat_result, to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "bn" in m._modules:
                bn = m.bn
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]
            y = act(y, m.act)

            c = layers['cv6'].conv
            m = layers['cv6']
            y = conv2d(y, to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "bn" in m._modules:
                bn = m.bn
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]
            y1 = act(y, m.act)

            c = layers['cv2'].conv
            m = layers['cv2']
            y = conv2d(src[0], to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "bn" in m._modules:
                bn = m.bn
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]
            y2 = act(y, m.act)

            concat_result = np.concatenate((y1, y2), axis=0)

            c = layers['cv7'].conv
            m = layers['cv7']
            y = conv2d(concat_result, to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "bn" in m._modules:
                bn = m.bn
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]
            y = act(y, m.act)

        elif kind == "RepConv":
            layers = children_items(m)

            c = layers['rbr_dense']._modules['0']
            m = layers['rbr_dense']
            y = conv2d(src[0], to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "1" in m._modules:
                bn = m._modules['1']
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y1 = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]

            c = layers['rbr_1x1']._modules['0']
            m = layers['rbr_1x1']
            y = conv2d(src[0], to_np(c.weight), to_np(c.bias), as_int(c.stride), as_int(c.padding))
            if "1" in m._modules:
                bn = m._modules['1']
                y = (y - to_np(bn.running_mean)[:, None, None]) / np.sqrt(to_np(bn.running_var)[:, None, None] + float(bn.eps))
                y2 = y * to_np(bn.weight)[:, None, None] + to_np(bn.bias)[:, None, None]

            # Add y1 and y2 and apply the activation
            y = act(y1+y2, layers['act'])
        else:
            sys.exit(f"unsupported layer {kind}")
        outs.append(y)
    return outs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("dump_dir")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    ckpt = load_checkpoint_numpy(args.checkpoint)
    model = ckpt.get("ema") or ckpt["model"]
    x = np.fromfile(f"{args.dump_dir}/input.bin", dtype=np.float32).reshape(3, args.size, args.size)

    ref = forward(model, x.astype(np.float64))
    worst = 0.0
    for j, r in enumerate(ref):
        g = np.fromfile(f"{args.dump_dir}/head{j}.bin", dtype=np.float32).reshape(r.shape)
        err = np.abs(g - r)
        rel = err.max() / (np.abs(r).max() + 1e-9)
        worst = max(worst, rel)
        print(f"head{j} {tuple(r.shape)}: max abs err {err.max():.3e}, mean abs err {err.mean():.3e}, max rel err {rel:.3e}")
    print("PASS" if worst < 1e-2 else "FAIL", f"(worst relative error {worst:.2e})")


if __name__ == "__main__":
    main()
