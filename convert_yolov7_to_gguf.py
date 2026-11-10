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
nn.MaxPool2d, nn.Upsample (nearest, integer scale), Detect, IDetect.
That covers yolov7-tiny. Full YOLOv7 additionally needs RepConv and SPPCSPC
and is rejected with a clear error.

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

from enum import Enum

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


class MODEL_NAMES(str, Enum):
    V7 ='v7'
    V7X ='v7x'
    V7_TINY = 'v7-tiny'

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

def resolve_from_concat(f, i, count, start):
    fs = f if isinstance(f, (list, tuple)) else [f]
    # negative = relative to this layer; -1 for layer 0 means the network input (kept as -1)
    return [(i + x if x < 0 else x if x < start else x+count) for x in fs]

def convert(model, ftype, model_name):
    layers, tensors = [], {}
    mods = children(model.model)

    detect = None

    count = 0 # This variable keeps track of layers which are added by SPPCSPC and RepConv operator
    start = 0 # This keeps track of the index where SPPCSPC layers was added
    repconv_count = 0 # This keeps track of how many RepConv layers have been added
    for i, m in enumerate(mods):
        kind = type(m).__name__

        if kind == "Concat":
            src = resolve_from_concat(m.f, i+count, count, start)

        elif kind == "Detect" or kind == "IDetect":
            if MODEL_NAMES.V7.value == model_name:
                src = [115, 118, 121]
            elif MODEL_NAMES.V7X.value == model_name:
                src = [129, 130, 131]
            elif MODEL_NAMES.V7_TINY.value == model_name:
                raise ValueError("Add detection head for tiny model")

        else:
            src = resolve_from_concat(m.f, i+count, count, start)

        fr = ",".join(str(x) for x in src)


        if kind == "Conv":
            k, s, p = conv_geometry(m.conv)
            w, b = fold_conv_bn(m.conv, getattr(m, "bn", None) if "bn" in m._modules else None)
            if (int(fr) < start):
                tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
                layers.append(f"conv from={fr} k={k} s={s} p={p} act={activation(m.act)}")
            else:
                tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
                layers.append(f"conv from={fr} k={k} s={s} p={p} act={activation(m.act)}")

        elif kind == "Concat":
            if int(m.d) != 1:
                raise ValueError(f"layer {i}: Concat on dim {m.d} is not supported")
            layers.append(f"concat from={fr}")

        elif kind in ("MP", "SP", "MaxPool2d"):
            pool = m if kind == "MaxPool2d" else m.m
            k, s, p = as_int(pool.kernel_size), as_int(pool.stride), as_int(pool.padding)
            if getattr(pool, "ceil_mode", False):
                raise ValueError(f"layer {i}: ceil_mode pooling is not supported")
            layers.append(f"maxpool from={fr} k={k} s={s} p={p}")

        elif kind == "Upsample":
            if m.mode != "nearest" or m.scale_factor is None:
                raise ValueError(f"layer {i}: only nearest Upsample with scale_factor is supported")
            sf = m.scale_factor
            sf = sf[0] if isinstance(sf, (tuple, list)) else sf
            if float(sf) != int(sf):
                raise ValueError(f"layer {i}: non-integer upsample scale {sf}")
            layers.append(f"upsample from={fr} scale={int(sf)}")

        elif kind in ("Detect", "IDetect"):
            graph_decode = False
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

        elif kind in ("SPPCSPC"):

            # Here is the forward function of SPPCSPC node:
            # All self.cvn (n represents layer number e.g 1,2,3..) are convolution layers
            # x1 = self.cv4(self.cv3(self.cv1(x)))
            # y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
            # y2 = self.cv2(x)
            # return self.cv7(torch.cat((y1, y2), dim=1))

            # cv1, 51
            start = i
            cv1 = m._modules["cv1"]
            k, s, p = conv_geometry(cv1.conv)
            w, b = fold_conv_bn(cv1.conv, getattr(cv1, "bn", None) if "bn" in cv1._modules else None)
            tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
            layers.append(f"conv from={str(int(fr)+count)} k={k} s={s} p={p} act={activation(cv1.act)}")
            count += 1

            # cv3
            cv3 = m._modules["cv3"]
            k, s, p = conv_geometry(cv3.conv)
            w, b = fold_conv_bn(cv3.conv, getattr(cv3, "bn", None) if "bn" in cv3._modules else None)
            tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
            layers.append(f"conv from={str(int(fr)+count)} k={k} s={s} p={p} act={activation(cv3.act)}")
            count += 1

            # cv4, 53
            cv4 = m._modules["cv4"]
            k, s, p = conv_geometry(cv4.conv)
            w, b = fold_conv_bn(cv4.conv, getattr(cv4, "bn", None) if "bn" in cv4._modules else None)
            tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
            layers.append(f"conv from={str(int(fr)+count)} k={k} s={s} p={p} act={activation(cv4.act)}")
            count += 1
            cv4_count = count

            # MaxPool2d Module List, name of the ModuleList is m
            # ModuleList[0]
            pool = m._modules["m"]._modules["0"]
            k, s, p = as_int(pool.kernel_size), as_int(pool.stride), as_int(pool.padding)
            if getattr(pool, "ceil_mode", False):
                raise ValueError(f"layer {i}: ceil_mode pooling is not supported")
            layers.append(f"maxpool from={str(int(fr)+cv4_count)} k={k} s={s} p={p}")
            count += 1

            # MaxPool2d Module List, name of the ModuleList is m
            # ModuleList[1]
            pool = m._modules["m"]._modules["1"]
            k, s, p = as_int(pool.kernel_size), as_int(pool.stride), as_int(pool.padding)
            if getattr(pool, "ceil_mode", False):
                raise ValueError(f"layer {i}: ceil_mode pooling is not supported")
            layers.append(f"maxpool from={str(int(fr)+cv4_count)} k={k} s={s} p={p}")
            count += 1

            # MaxPool2d Module List, name of the ModuleList is m
            # ModuleList[2]
            pool = m._modules["m"]._modules["2"]
            k, s, p = as_int(pool.kernel_size), as_int(pool.stride), as_int(pool.padding)
            if getattr(pool, "ceil_mode", False):
                raise ValueError(f"layer {i}: ceil_mode pooling is not supported")
            layers.append(f"maxpool from={str(int(fr)+cv4_count)} k={k} s={s} p={p}")
            count += 1

            # Concat
            # Below code is building concat operation from following part of SPPCSPC: y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
            fr_of_concat = str(int(fr)+count-3) + "," + str(int(fr)+count-2) + "," + str(int(fr)+count-1) + "," + str(int(fr)+count) # fr_of_concat = "53,54,55,56" for yolov7
            layers.append(f"concat from={fr_of_concat}")
            count += 1

            # cv5, 58
            cv5 = m._modules["cv5"]
            k, s, p = conv_geometry(cv5.conv)
            w, b = fold_conv_bn(cv5.conv, getattr(cv5, "bn", None) if "bn" in cv5._modules else None)
            tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
            layers.append(f"conv from={str(int(fr)+count)} k={k} s={s} p={p} act={activation(cv5.act)}")
            count += 1

            # cv6, 59
            cv6 = m._modules["cv6"]
            k, s, p = conv_geometry(cv6.conv)
            w, b = fold_conv_bn(cv6.conv, getattr(cv6, "bn", None) if "bn" in cv6._modules else None)
            tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
            layers.append(f"conv from={str(int(fr)+count)} k={k} s={s} p={p} act={activation(cv6.act)}")
            count += 1

            # cv2
            cv2 = m._modules["cv2"]
            k, s, p = conv_geometry(cv2.conv)
            w, b = fold_conv_bn(cv2.conv, getattr(cv2, "bn", None) if "bn" in cv2._modules else None)
            tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
            layers.append(f"conv from={fr} k={k} s={s} p={p} act={activation(cv2.act)}")
            count += 1

            # Concat, 61
            fr_of_concat = str(int(fr)+count-1) + "," + str(int(fr)+count) # fr_of_concat = "59,60" for yolov7
            layers.append(f"concat from={fr_of_concat}")
            count += 1

            # cv7, 62
            cv7 = m._modules["cv7"]
            k, s, p = conv_geometry(cv7.conv)
            w, b = fold_conv_bn(cv7.conv, getattr(cv7, "bn", None) if "bn" in cv7._modules else None)
            tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
            layers.append(f"conv from={str(int(fr)+count)} k={k} s={s} p={p} act={activation(cv7.act)}")
            # Don't need to increment "count" for this layer because "i" will be incremented by the for loop and we are adding "count" to "i"


        elif kind in ("RepConv"):

            # This is RepConv for yolov7 self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs))
            num_of_layers = len(layers)

            if repconv_count == 0:
                deduct_count = 0
            elif repconv_count == 1:
                deduct_count = 2
            elif repconv_count == 2:
                deduct_count = 4
            else:
                raise ValueError("RepConv cannot be more than 3 as yolov7 only have " \
                "3 RepConv layers")

            # rbr_dense is a Sequential List
            rbr_dense = m._modules["rbr_dense"]._modules
            k, s, p = conv_geometry(rbr_dense["0"])
            w, b = fold_conv_bn(rbr_dense['0'], rbr_dense['1'])
            tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
            layers.append(f"conv from={str(int(fr)-deduct_count)} k={k} s={s} p={p}")
            count += 1

            # rbr_1x1 is a Sequential List
            rbr_1x1 = m._modules["rbr_1x1"]._modules
            k, s, p = conv_geometry(rbr_1x1["0"])
            w, b = fold_conv_bn(rbr_1x1['0'], rbr_1x1['1'])
            # In case of rbr_1x1, its input is also the input of rbr_dense, hence we need to reduce the "from" by 1 to point
            # to the input of rbr_dense
            tensors[f"l{i+count}.weight"], tensors[f"l{i+count}.bias"] = w, b
            layers.append(f"conv from={str(int(fr)-deduct_count)} k={k} s={s} p={p}")
            count += 1

            # Add
            fr_of_add = str(num_of_layers) + "," + str(num_of_layers+1)
            layers.append(f"add from={fr_of_add} act={activation(m._modules['act'])}")
            # Don't need to increment "count" for this layer because "i" will be incremented by the for loop and we are adding "count" to "i"

            repconv_count += 1


        elif kind in ("ReOrg", "DownC", "SPPF", "Bottleneck", "C3"):
            raise ValueError(f"layer {i}: {kind} is used by the larger YOLOv7 variants and is not "
                             "implemented in this converter/engine yet (yolov7-tiny does not use it)")
        else:
            string = f"layer {i}: unsupported module type {kind}"
            raise ValueError(string)

    if detect is None:
        raise ValueError("no Detect/IDetect layer found")

    nl, na = int(detect.nl), int(detect.na)
    nc = int(detect.nc)
    strides = to_np(detect.stride).reshape(-1).astype(int).tolist()
    if "anchor_grid" in detect._buffers or hasattr(detect, "anchor_grid"):
        anchors = to_np(detect.anchor_grid).reshape(nl, na, 2)          # pixels
    else:
        anchors = to_np(detect.anchors).reshape(nl, na, 2) * np.array(strides)[:, None, None]

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
        if arr.ndim == 4 and ftype == "f16":
            arr = arr.astype(np.float16)
        else:
            arr = arr.astype(np.float32)        # biases always f32
        w.add_tensor(tname, np.ascontiguousarray(arr))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

def check_model_name(name):

    if "." in name:
        parts = name.split(".")
        lower_case_name = parts[0].lower()

        if MODEL_NAMES.V7X in lower_case_name:
            return MODEL_NAMES.V7X.value
        elif MODEL_NAMES.V7 in lower_case_name:
            return MODEL_NAMES.V7.value
        elif MODEL_NAMES.V7_TINY in lower_case_name:
            MODEL_NAMES.V7_TINY.value
        else:
            output = f"Could not determine model name from {name}"
            raise ValueError(output)
    else:
        output = f"Could not determine model name from {name}"
        raise ValueError(output)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="yolov7.pt")
    ap.add_argument("output", help="output .gguf")
    ap.add_argument("--type", choices=["f32", "f16"], default="f16", help="conv weight type (default f16)")
    ap.add_argument("--torch", action="store_true", help="load with torch.load (needs yolov7 repo on PYTHONPATH)")
    args = ap.parse_args()

    ckpt = load_checkpoint_torch(args.checkpoint) if args.torch else load_checkpoint_numpy(args.checkpoint)
    model = ckpt.get("ema") or ckpt["model"] if isinstance(ckpt, dict) else ckpt

    model_name = check_model_name(args.checkpoint)
    layers, tensors, meta = convert(model, args.type, model_name)
    name = args.checkpoint.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    write_gguf(args.output, layers, tensors, meta, args.type, name)

    n_params = sum(t.size for t in tensors.values())
    print(f"wrote {args.output}: {len(layers)} layers, {len(tensors)} tensors, "
          f"{n_params / 1e6:.2f}M params, {meta['nc']} classes, strides {meta['strides']}")


if __name__ == "__main__":
    main()
