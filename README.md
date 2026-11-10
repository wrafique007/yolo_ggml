# yolov7 on ggml

Object detection with YOLOv7 using [ggml](https://github.com/ggml-org/ggml) as the inference engine. It runs on CPU, or on CUDA, Metal or Vulkan with the usual ggml build flags.

| File | Purpose |
|---|---|
| `convert_yolov7_to_gguf.py` | `.pt` checkpoint → `.gguf` (folds BN, fuses IDetect, stores topology as metadata) |
| `yolov7.cpp` | ggml inference: load GGUF, letterbox, graph, decode, NMS, draw boxes |
| `verify.py` | Compares ggml outputs to a numpy re-implementation of the PyTorch forward pass |
| `inspect.py` | This checks input and output shpaes of layers and lets you know if there is any discrepancy |
| `CMakeLists.txt` | Builds against a local ggml (`-DGGML_DIR=...`) or fetches it automatically |

## 1. Convert the weights

```bash
pip install numpy gguf
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python convert_yolov7_to_gguf.py yolov7-tiny.pt yolov7-tiny-f32.gguf --type f32
python convert_yolov7_to_gguf.py yolov7-tiny.pt yolov7-tiny-f16.gguf --type f16

python convert_yolov7_to_gguf.py yolov7.pt yolov7-f32.gguf --type f32

python inspect_gguf.py yolov7-f32.gguf
```

You don't need torch or the yolov7 repo; the checkpoint is read directly. Your own fine-tuned yolov7-tiny checkpoints work the same way, and class names come from the checkpoint.

## 2. Build

```bash
cmake -B build                       # fetches ggml automatically
cmake -B build -DGGML_BACKTRACE_LLDB=ON
# or: cmake -B build -DGGML_DIR=/path/to/ggml
# GPU: add -DGGML_CUDA=ON / -DGGML_METAL=ON / -DGGML_VULKAN=ON
cmake --build build -j
```

## 3. Run

```bash
./build/yolov7 -m yolov7-tiny-f32.gguf -i horses.jpg -o out.jpg
./build/yolov7 -m yolov7-tiny-f16.gguf -i horses.jpg --gpu        # GPU backend
./build/yolov7 -m yolov7-tiny-f32.gguf -i horses.jpg -s 416 -t 8   # smaller input, 8 threads

./build/yolov7 -m yolov7-f32.gguf -i horses.jpg -o out.jpg
```

Options: `--conf 0.25`, `--iou 0.45`, `--bench N`, `--dump DIR`.

## 4. Verify (optional)

```bash
./build/yolov7 -m yolov7-tiny-f32.gguf -i horses.jpg --dump dump
python verify.py yolov7-tiny.pt dump

./build/yolov7 -m yolov7-f32.gguf -i horses.jpg --dump dump
python verify.py yolov7.pt dump
```

## How it maps onto ggml

| YOLOv7 | ggml |
|---|---|
| Conv + BN (folded) | `ggml_im2col` + `ggml_mul_mat` + `ggml_add` |
| LeakyReLU / SiLU | `ggml_leaky_relu` / `ggml_silu` |
| MP, SP | `ggml_pool_2d(GGML_OP_POOL_MAX)` |
| Upsample | `ggml_upscale(..., GGML_SCALE_MODE_NEAREST)` |
| Concat | `ggml_concat(..., dim=2)` |
| Detect | 1×1 conv on the backend; sigmoid decode and NMS on the CPU |

Tensors use ggml's `[W, H, C, N]` layout, which is the same memory layout as PyTorch NCHW with batch 1. That's why the letterboxed planar RGB buffer is copied straight into the input tensor.

## Limits

* Batch size 1, square input (multiple of 32).
* Labels are printed to stdout; boxes are drawn without text.
