#!/usr/bin/env python3
"""
Convert a YOLOv7 / YOLOv7-tiny PyTorch checkpoint (*.pt from WongKinYiu/yolov7)
into a GGUF file that the ggml inference program (yolov7.cpp) can run.

What the converter does
  * walks model.model (the nn.Sequential built from the YAML config)
  * folds BatchNorm into each Conv's weight and bias
  * fuses IDetect's ImplicitA / ImplicitM into the detection 1x1 convs
  * writes the network topology as GGUF metadata (one string per layer)
    and the weights as GGUF tensors

Torch is NOT required: the checkpoint is a zip of pickles + raw storages, and
it is read directly with numpy. The yolov7 repository is NOT required either.
Pass --torch to load with torch.load() instead (then the yolov7 repo must be
importable, e.g. PYTHONPATH=/path/to/yolov7).

Supported layers: Conv (+BN, LeakyReLU/SiLU/ReLU/none), Concat, MP, SP,
nn.MaxPool2d, nn.Upsample (nearest, integer scale), Detect, IDetect,
RepConv (fused into one 3x3 conv) and SPPCSPC (written as its primitive
conv / maxpool / concat steps). That covers yolov7-tiny and yolov7.

Usage
  pip install numpy gguf
  python convert_yolov7_to_gguf.py yolov7-tiny.pt yolov7-tiny-f16.gguf --type f16
"""

import argparse
import collections
import pickle
import sys
import zipfile

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("error: the 'gguf' package is missing. Install it with: pip install gguf\n"
             "       (or add llama.cpp/gguf-py to PYTHONPATH)")


# ----------------------------------------------------------------------------
# Torch-free checkpoint loader
# ----------------------------------------------------------------------------

_STORAGE_DTYPES = {
    "FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
    "BFloat16Storage": "bf16",
}


class _StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


class _Stub:
    """Stand-in for any non-builtin class in the pickle (yolov7 modules, torch.nn modules...).
    Attribute lookup falls through to _parameters / _buffers / _modules like nn.Module does."""

    def __init__(self, *args, **kwargs):
        self._args = args

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:  # (dict, slots)
            state = {**(state[0] or {}), **(state[1] or {})}
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __getattr__(self, name):
        d = self.__dict__
        for k in ("_parameters", "_buffers", "_modules"):
            if k in d and name in d[k]:
                return d[k][name]
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")


_stub_classes = {}


def _stub_class(module, name):
    key = (module, name)
    if key not in _stub_classes:
        _stub_classes[key] = type(name, (_Stub,), {"__module__": module})
    return _stub_classes[key]


def _rebuild_tensor(storage, offset, size, stride, *_args, **_kwargs):
    if len(size) == 0:
        return storage[offset:offset + 1].reshape(()).copy()
    itemsize = storage.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(s * itemsize for s in stride))
    return view.copy()


def _rebuild_parameter(data, *_args, **_kwargs):
    return data


class _CheckpointUnpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self._zf, self._prefix, self._cache = zf, prefix, {}

    def find_class(self, module, name):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name.startswith("_rebuild_parameter"):
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(_STORAGE_DTYPES[name])
        if module in ("collections", "builtins", "copyreg", "_codecs") or module.startswith("numpy"):
            return super().find_class(module, name)
        if module == "torch._utils" and name == "_rebuild_device_tensor_from_numpy":
            return lambda data, dtype, device, requires_grad: data
        return _stub_class(module, name)

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, storage_type, key, _loc, _numel = pid
        if key not in self._cache:
            raw = self._zf.read(f"{self._prefix}data/{key}")
            dt = storage_type.dtype if isinstance(storage_type, _StorageType) else np.float32
            if dt == "bf16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
            else:
                arr = np.frombuffer(raw, dtype=dt)
            self._cache[key] = arr
        return self._cache[key]


def load_checkpoint_numpy(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")]
    if not pkl:
        sys.exit(f"error: {path} is not a zip-format torch checkpoint (legacy format); "
                 "re-save it with a recent torch or use --torch")
    prefix = pkl[0][: -len("data.pkl")]
    with zf.open(pkl[0]) as f:
        return _CheckpointUnpickler(f, zf, prefix).load()


def load_checkpoint_torch(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")


# ----------------------------------------------------------------------------
# Model walking (works on real torch modules and on the stubs above)
# ----------------------------------------------------------------------------

def to_np(t):
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t.astype(np.float32)
    if isinstance(t, (int, float)):
        return np.array(t, dtype=np.float32)
    return t.detach().float().cpu().numpy()


def children(m):
    return list(m._modules.values())

def children_items(m):
    return m._modules


def as_int(v):
    if isinstance(v, (tuple, list)):
        assert all(x == v[0] for x in v), f"non-square value {v} is not supported"
        v = v[0]
    return int(v)


def activation(act):
    if act is None or act is True or act is False:
        return "none"
    name = type(act).__name__
    if name == "LeakyReLU":
        return f"leaky:{float(act.negative_slope)}"
    if name == "SiLU":
        return "silu"
    if name == "ReLU":
        return "relu"
    if name == "Identity":
        return "none"
    raise ValueError(f"unsupported activation {name}")


def fold_conv_bn(conv, bn):
    w = to_np(conv.weight)
    b = to_np(conv.bias) if conv.bias is not None else np.zeros(w.shape[0], np.float32)
    if bn is not None:
        gamma, beta = to_np(bn.weight), to_np(bn.bias)
        mean, var = to_np(bn.running_mean), to_np(bn.running_var)
        scale = gamma / np.sqrt(var + float(bn.eps))
        w = w * scale[:, None, None, None]
        b = (b - mean) * scale + beta
    return w.astype(np.float32), b.astype(np.float32)


def conv_geometry(conv):
    if as_int(conv.groups) != 1:
        raise ValueError("grouped convolution is not supported")
    if as_int(conv.dilation) != 1:
        raise ValueError("dilated convolution is not supported")
    if isinstance(conv.padding, str):
        raise ValueError("string padding is not supported")
    return as_int(conv.kernel_size), as_int(conv.stride), as_int(conv.padding)


def resolve_from(f, i):
    fs = f if isinstance(f, (list, tuple)) else [f]
    # negative = relative to this layer; -1 for layer 0 means the network input (kept as -1)
    return [(i + x if x < 0 else x) for x in fs]


def fold_repconv(m):
    """RepConv (training form: 3x3+BN, 1x1+BN, optional identity BN) -> one 3x3 conv.
    Same math as RepConv.get_equivalent_kernel_bias() in yolov7/models/common.py."""
    if "rbr_reparam" in m._modules:                       # already reparameterized (deploy=True)
        c = m.rbr_reparam
        k, s, p = conv_geometry(c)
        w, b = fold_conv_bn(c, None)
        return w, b, k, s, p
    dense_conv, dense_bn = children(m.rbr_dense)
    one_conv, one_bn = children(m.rbr_1x1)
    k, s, p = conv_geometry(dense_conv)
    if k != 3 or as_int(one_conv.kernel_size) != 1:
        raise ValueError("RepConv: expected a 3x3 dense branch and a 1x1 branch")
    w3, b3 = fold_conv_bn(dense_conv, dense_bn)
    w1, b1 = fold_conv_bn(one_conv, one_bn)
    w = w3 + np.pad(w1, ((0, 0), (0, 0), (1, 1), (1, 1)))  # _pad_1x1_to_3x3_tensor
    b = b3 + b1
    idn = m._modules.get("rbr_identity")
    if idn is not None:                                   # identity = 3x3 kernel with 1 at the centre
        c = w.shape[0]
        wid = np.zeros_like(w)
        wid[np.arange(c), np.arange(c), 1, 1] = 1.0
        scale = to_np(idn.weight) / np.sqrt(to_np(idn.running_var) + float(idn.eps))
        w = w + wid * scale[:, None, None, None]
        b = b + to_np(idn.bias) - to_np(idn.running_mean) * scale
    return w.astype(np.float32), b.astype(np.float32), k, s, p


def pool_geometry(pool):
    if getattr(pool, "ceil_mode", False):
        raise ValueError("ceil_mode pooling is not supported")
    return as_int(pool.kernel_size), as_int(pool.stride), as_int(pool.padding)


class GraphWriter:
    """Builds the flat layer list stored in GGUF metadata.

    Every entry is one primitive the engine knows (conv / maxpool / concat / upsample / detect).
    A composite PyTorch module such as SPPCSPC is written as several entries, so its internal
    torch.cat calls become ordinary 'concat' entries. Because of that, the flat index of a layer
    can differ from its index in model.model; out_idx maps model.model index -> flat index of
    that module's output, and every 'from' is translated through it."""

    def __init__(self):
        self.layers, self.tensors, self.out_idx = [], {}, []

    def _add(self, desc):
        self.layers.append(desc)
        return len(self.layers) - 1

    @staticmethod
    def _fr(srcs):
        return ",".join(str(s) for s in srcs)

    def conv(self, src, w, b, k, s, p, act):
        i = len(self.layers)
        self.tensors[f"l{i}.weight"], self.tensors[f"l{i}.bias"] = w, b
        return self._add(f"conv from={src} k={k} s={s} p={p} act={act}")

    def conv_module(self, src, m):                        # yolov7 Conv = Conv2d + BN + act
        k, s, p = conv_geometry(m.conv)
        w, b = fold_conv_bn(m.conv, m.bn if "bn" in m._modules else None)
        return self.conv(src, w, b, k, s, p, activation(m.act))

    def maxpool(self, src, pool):
        k, s, p = pool_geometry(pool)
        return self._add(f"maxpool from={src} k={k} s={s} p={p}")

    def concat(self, srcs):                               # torch.cat(srcs, dim=1)
        return self._add(f"concat from={self._fr(srcs)}")

    def add(self, srcs, act="none"):                      # act(srcs[0] + srcs[1] + ...)
        if len(srcs) < 2:
            raise ValueError("add needs at least two inputs")
        return self._add(f"add from={self._fr(srcs)} act={act}")

    def sppcspc(self, x, m):
        # x1 = self.cv4(self.cv3(self.cv1(x)))
        x1 = self.conv_module(self.conv_module(self.conv_module(x, m.cv1), m.cv3), m.cv4)
        # torch.cat([x1] + [m(x1) for m in self.m], 1)
        pools = [self.maxpool(x1, pool) for pool in children(m.m)]
        cat1 = self.concat([x1] + pools)
        # y1 = self.cv6(self.cv5(cat1))
        y1 = self.conv_module(self.conv_module(cat1, m.cv5), m.cv6)
        # y2 = self.cv2(x)
        y2 = self.conv_module(x, m.cv2)
        # self.cv7(torch.cat((y1, y2), dim=1))
        return self.conv_module(self.concat([y1, y2]), m.cv7)


def convert(model, ftype, graph_decode=False):
    gw = GraphWriter()
    tensors = gw.tensors
    mods = children(model.model)
    detect = None

    for i, m in enumerate(mods):
        kind = type(m).__name__
        # model.model indices -> flat indices (-1 = network input)
        src = [gw.out_idx[j] if j >= 0 else -1 for j in resolve_from(m.f, i)]
        fr = gw._fr(src)

        if kind == "Conv":
            gw.out_idx.append(gw.conv_module(src[0], m))
            continue

        if kind == "RepConv":
            w, b, k, s, p = fold_repconv(m)
            gw.out_idx.append(gw.conv(src[0], w, b, k, s, p, activation(m.act)))
            continue

        if kind == "SPPCSPC":
            gw.out_idx.append(gw.sppcspc(src[0], m))
            continue

        if kind == "Concat":
            if int(m.d) != 1:
                raise ValueError(f"layer {i}: Concat on dim {m.d} is not supported")
            gw.out_idx.append(gw.concat(src))
            continue

        if kind in ("MP", "SP", "MaxPool2d"):
            gw.out_idx.append(gw.maxpool(src[0], m if kind == "MaxPool2d" else m.m))
            continue

        gw.out_idx.append(len(gw.layers))                 # the single entry appended below
        layers = gw.layers

        if kind == "Upsample":
            if m.mode != "nearest" or m.scale_factor is None:
                raise ValueError(f"layer {i}: only nearest Upsample with scale_factor is supported")
            sf = m.scale_factor
            sf = sf[0] if isinstance(sf, (tuple, list)) else sf
            if float(sf) != int(sf):
                raise ValueError(f"layer {i}: non-integer upsample scale {sf}")
            layers.append(f"upsample from={fr} scale={int(sf)}")

        elif kind in ("Detect", "IDetect"):
            if i != len(mods) - 1:
                raise ValueError("Detect must be the last layer")
            detect = m
            convs = children(m.m)
            ias = children(m.ia) if kind == "IDetect" else [None] * len(convs)
            ims = children(m.im) if kind == "IDetect" else [None] * len(convs)
            for j, (c, ia, im) in enumerate(zip(convs, ias, ims)):
                w = to_np(c.weight)
                b = to_np(c.bias) if c.bias is not None else np.zeros(w.shape[0], np.float32)
                if ia is not None:   # conv(x + a) = conv(x) + W @ a
                    b = b + w[:, :, 0, 0] @ to_np(ia.implicit).reshape(-1)
                if im is not None:   # m * conv(x)
                    mul = to_np(im.implicit).reshape(-1)
                    w = w * mul[:, None, None, None]
                    b = b * mul
                tensors[f"det{j}.weight"], tensors[f"det{j}.bias"] = w.astype(np.float32), b.astype(np.float32)
            # decode=1 asks the engine to build Detect's inference path inside the ggml graph:
            # view -> permute -> sigmoid -> xy/wh decode -> view(bs,-1,no) -> torch.cat(z, 1)
            layers.append(f"detect from={fr}" + (" decode=1" if graph_decode else ""))

        else:
            raise ValueError(f"layer {i}: unsupported module type {kind}")

    layers = gw.layers
    if detect is None:
        raise ValueError("no Detect/IDetect layer found")

    nl, na = int(detect.nl), int(detect.na)
    nc = int(detect.nc)
    strides = to_np(detect.stride).reshape(-1).astype(int).tolist()
    if "anchor_grid" in detect._buffers or hasattr(detect, "anchor_grid"):
        anchors = to_np(detect.anchor_grid).reshape(nl, na, 2)          # pixels
    else:
        anchors = to_np(detect.anchors).reshape(nl, na, 2) * np.array(strides)[:, None, None]

    if graph_decode:
        # anchor_grid in PyTorch is (1, na, 1, 1, 2) per level; stored as numpy (na, 1, 1, 2)
        # = ggml [2, 1, 1, na], which broadcasts over the [no, nx, ny, na] head tensor.
        for j in range(nl):
            tensors[f"det{j}.anchor_grid"] = anchors[j].reshape(na, 1, 1, 2).astype(np.float32)

    names = getattr(model, "names", None) or [f"class{c}" for c in range(nc)]
    meta = dict(nl=nl, na=na, nc=nc, strides=strides, anchors=anchors, names=list(names))
    return layers, tensors, meta


def write_gguf(path, layers, tensors, meta, ftype, name):
    w = gguf.GGUFWriter(path, arch="yolov7")
    w.add_name(name)
    w.add_uint32("yolov7.num_classes", meta["nc"])
    w.add_uint32("yolov7.num_anchors", meta["na"])
    w.add_uint32("yolov7.input_size", 640)
    w.add_array("yolov7.strides", [int(s) for s in meta["strides"]])
    w.add_array("yolov7.anchors", [float(x) for x in meta["anchors"].reshape(-1)])
    w.add_array("yolov7.class_names", [str(n) for n in meta["names"]])
    w.add_array("yolov7.layers", layers)

    for tname, arr in tensors.items():
        if tname.endswith(".weight") and arr.ndim == 4 and ftype == "f16":
            arr = arr.astype(np.float16)
        else:
            arr = arr.astype(np.float32)        # biases and anchors always f32
        w.add_tensor(tname, np.ascontiguousarray(arr))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="yolov7-tiny.pt")
    ap.add_argument("output", help="output .gguf")
    ap.add_argument("--type", choices=["f32", "f16"], default="f16", help="conv weight type (default f16)")
    ap.add_argument("--torch", action="store_true", help="load with torch.load (needs yolov7 repo on PYTHONPATH)")
    ap.add_argument("--graph-decode", action="store_true",
                    help="do Detect's sigmoid/decode/torch.cat inside the ggml graph "
                         "(output [1, N, 5+nc] like the PyTorch model) instead of in C++ after it")
    args = ap.parse_args()

    ckpt = load_checkpoint_torch(args.checkpoint) if args.torch else load_checkpoint_numpy(args.checkpoint)
    model = ckpt.get("ema") or ckpt["model"] if isinstance(ckpt, dict) else ckpt

    layers, tensors, meta = convert(model, args.type, args.graph_decode)
    name = args.checkpoint.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    write_gguf(args.output, layers, tensors, meta, args.type, name)

    n_params = sum(t.size for t in tensors.values())
    print(f"wrote {args.output}: {len(layers)} layers, {len(tensors)} tensors, "
          f"{n_params / 1e6:.2f}M params, {meta['nc']} classes, strides {meta['strides']}")


if __name__ == "__main__":
    main()
