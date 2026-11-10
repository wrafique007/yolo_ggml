// YOLOv7(-tiny) object detection with ggml as the inference engine.
//
//   ./yolov7 -m yolov7-tiny-f16.gguf -i image.jpg -o out.jpg
//
// The network topology is read from the GGUF metadata written by
// convert_yolov7_to_gguf.py, so this file contains no hard-coded layer list.
// Pipeline: load GGUF -> letterbox image -> build ggml graph -> run on the
// chosen backend (CPU, or the best GPU backend ggml was built with) ->
// decode the three detection heads -> NMS -> draw boxes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

enum class layer_type { conv, concat, maxpool, upsample, add, detect };
enum class act_type   { none, leaky, silu, relu };

struct yolo_layer {
    layer_type       type;
    std::vector<int> from;          // absolute layer indices, -1 = network input
    int k = 1, s = 1, p = 0;        // conv / pool geometry
    act_type act   = act_type::none;
    float    slope = 0.1f;          // leaky relu slope
    int      scale = 2;             // upsample factor
    ggml_tensor * w = nullptr;      // conv weight [KW, KH, IC, OC]
    ggml_tensor * b = nullptr;      // conv bias   [OC]
    std::vector<ggml_tensor *> det_w, det_b;   // detect head 1x1 convs
    bool decode = false;                       // build Detect's decode + torch.cat in the graph
    std::vector<ggml_tensor *> det_anchor;     // anchor_grid per level [2, 1, 1, na] (decode only)
};

struct yolo_model {
    std::vector<yolo_layer>  layers;
    int                      nc = 80, na = 3, input_size = 640;
    std::vector<int>         strides;
    std::vector<float>       anchors;      // [nl][na][2] in pixels
    std::vector<std::string> names;

    ggml_backend_t        backend     = nullptr;   // weights live here
    ggml_backend_t        backend_cpu = nullptr;   // fallback for unsupported ops
    ggml_context        * ctx_w       = nullptr;
    ggml_backend_buffer_t buf_w       = nullptr;
    ggml_backend_sched_t  sched       = nullptr;

    ~yolo_model() {
        if (sched)       ggml_backend_sched_free(sched);
        if (buf_w)       ggml_backend_buffer_free(buf_w);
        if (ctx_w)       ggml_free(ctx_w);
        if (backend_cpu && backend_cpu != backend) ggml_backend_free(backend_cpu);
        if (backend)     ggml_backend_free(backend);
    }
};

static std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::stoi(tok));
    return v;
}

// "conv from=12 k=3 s=1 p=1 act=leaky:0.1"
static yolo_layer parse_layer(const std::string & desc) {
    std::stringstream ss(desc);
    std::string kind, kv;
    ss >> kind;
    yolo_layer L;
    if      (kind == "conv")     L.type = layer_type::conv;
    else if (kind == "concat")   L.type = layer_type::concat;
    else if (kind == "maxpool")  L.type = layer_type::maxpool;
    else if (kind == "upsample") L.type = layer_type::upsample;
    else if (kind == "add")      L.type = layer_type::add;
    else if (kind == "detect")   L.type = layer_type::detect;
    else throw std::runtime_error("unknown layer type: " + kind);

    while (ss >> kv) {
        const auto eq = kv.find('=');
        const std::string key = kv.substr(0, eq), val = kv.substr(eq + 1);
        if      (key == "from")  L.from  = parse_ints(val);
        else if (key == "k")     L.k     = std::stoi(val);
        else if (key == "s")     L.s     = std::stoi(val);
        else if (key == "p")     L.p     = std::stoi(val);
        else if (key == "scale") L.scale = std::stoi(val);
        else if (key == "decode") L.decode = std::stoi(val) != 0;
        else if (key == "act") {
            if      (val == "none") L.act = act_type::none;
            else if (val == "silu") L.act = act_type::silu;
            else if (val == "relu") L.act = act_type::relu;
            else if (val.rfind("leaky", 0) == 0) {
                L.act = act_type::leaky;
                const auto c = val.find(':');
                if (c != std::string::npos) L.slope = std::stof(val.substr(c + 1));
            } else throw std::runtime_error("unknown activation: " + val);
        }
    }
    return L;
}

static int64_t gguf_key(const gguf_context * g, const char * key) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return id;
}

template <typename T>
static std::vector<T> gguf_numeric_array(const gguf_context * g, const char * key) {
    const int64_t id = gguf_key(g, key);
    const size_t  n  = gguf_get_arr_n(g, id);
    const void  * d  = gguf_get_arr_data(g, id);
    std::vector<T> out(n);
    switch (gguf_get_arr_type(g, id)) {
        case GGUF_TYPE_INT32:   for (size_t i = 0; i < n; i++) out[i] = (T) ((const int32_t  *) d)[i]; break;
        case GGUF_TYPE_UINT32:  for (size_t i = 0; i < n; i++) out[i] = (T) ((const uint32_t *) d)[i]; break;
        case GGUF_TYPE_FLOAT32: for (size_t i = 0; i < n; i++) out[i] = (T) ((const float    *) d)[i]; break;
        default: throw std::runtime_error(std::string("unexpected array type for ") + key);
    }
    return out;
}

static ggml_backend_t init_backend(bool use_gpu, int n_threads) {
    ggml_backend_load_all();   // no-op for static builds, loads plugins for GGML_BACKEND_DL builds

    ggml_backend_t backend = nullptr;
    if (use_gpu) {
        backend = ggml_backend_init_best();          // GPU if available, else CPU
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!backend) throw std::runtime_error("failed to initialize a ggml backend");

    // set thread count through the registry so this works for any CPU backend build
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_threads) set_threads(backend, n_threads);
    }
    return backend;
}

static void load_model(yolo_model & model, const char * path, bool use_gpu, int n_threads) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) throw std::runtime_error(std::string("failed to open GGUF file: ") + path);

    // --- hyper-parameters & topology ---
    model.nc         = (int) gguf_get_val_u32(g, gguf_key(g, "yolov7.num_classes"));
    model.na         = (int) gguf_get_val_u32(g, gguf_key(g, "yolov7.num_anchors"));
    model.input_size = (int) gguf_get_val_u32(g, gguf_key(g, "yolov7.input_size"));
    model.strides    = gguf_numeric_array<int>  (g, "yolov7.strides");
    model.anchors    = gguf_numeric_array<float>(g, "yolov7.anchors");

    const int64_t names_id = gguf_key(g, "yolov7.class_names");
    for (size_t i = 0; i < gguf_get_arr_n(g, names_id); i++) model.names.emplace_back(gguf_get_arr_str(g, names_id, i));

    const int64_t layers_id = gguf_key(g, "yolov7.layers");
    for (size_t i = 0; i < gguf_get_arr_n(g, layers_id); i++) model.layers.push_back(parse_layer(gguf_get_arr_str(g, layers_id, i)));

    // --- backend & weights ---
    model.backend = init_backend(use_gpu, n_threads);

    const int64_t n_tensors = gguf_get_n_tensors(g);
    ggml_init_params ip = { (size_t) n_tensors * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
    model.ctx_w = ggml_init(ip);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char  * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src  = ggml_get_tensor(meta, name);
        ggml_tensor * dst  = ggml_dup_tensor(model.ctx_w, src);
        ggml_set_name(dst, name);
    }
    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buf_w) throw std::runtime_error("failed to allocate weight buffer");

    FILE * f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot reopen ") + path);
    std::vector<uint8_t> tmp;
    for (int64_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = ggml_get_tensor(model.ctx_w, gguf_get_tensor_name(g, i));
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
        const size_t n   = ggml_nbytes(t);
        tmp.resize(n);
        if (std::fseek(f, (long) off, SEEK_SET) != 0 || std::fread(tmp.data(), 1, n, f) != n) {
            std::fclose(f);
            throw std::runtime_error(std::string("failed to read tensor ") + ggml_get_name(t));
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, n);
    }
    std::fclose(f);

    // --- bind tensors to layers ---
    auto get = [&](const std::string & name) {
        ggml_tensor * t = ggml_get_tensor(model.ctx_w, name.c_str());
        if (!t)
        {
            throw std::runtime_error("missing tensor " + name);
        }
        return t;
    };
    for (size_t i = 0; i < model.layers.size(); i++) {
        yolo_layer & L = model.layers[i];
        if (L.type == layer_type::conv) {
            L.w = get("l" + std::to_string(i) + ".weight");
            L.b = get("l" + std::to_string(i) + ".bias");
        } else if (L.type == layer_type::detect) {
            for (size_t j = 0; j < L.from.size(); j++) {
                L.det_w.push_back(get("det" + std::to_string(j) + ".weight"));
                L.det_b.push_back(get("det" + std::to_string(j) + ".bias"));
                if (L.decode) L.det_anchor.push_back(get("det" + std::to_string(j) + ".anchor_grid"));
            }
        }
    }

    gguf_free(g);
    ggml_free(meta);

    // --- scheduler: primary backend + CPU fallback for any op the GPU lacks ---
    std::vector<ggml_backend_t> backends = { model.backend };
    ggml_backend_dev_t dev = ggml_backend_get_device(model.backend);
    if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        model.backend_cpu = init_backend(false, n_threads);
        backends.push_back(model.backend_cpu);
    }
    model.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                         GGML_DEFAULT_GRAPH_SIZE, false, true);

    std::printf("loaded %s: %zu layers, %d classes, backend: %s\n",
                path, model.layers.size(), model.nc, ggml_backend_name(model.backend));
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

// conv2d via im2col + mul_mat. Activations are kept in the weight type
// (f32 weights -> exact f32 path, f16 weights -> f16 im2col, faster).
static ggml_tensor * conv2d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int s, int p) {
    const ggml_type col_type = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * col = ggml_im2col(ctx, w, x, s, s, p, p, 1, 1, true, col_type);   // [K, OW, OH, N]
    ggml_tensor * y = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2] * col->ne[3]), // [K, OW*OH]
        ggml_reshape_2d(ctx, w,   w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));        // [K, OC]
    y = ggml_reshape_4d(ctx, y, col->ne[1], col->ne[2], w->ne[3], 1);                // [OW, OH, OC, 1]
    return ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
}

static ggml_tensor * apply_act(ggml_context * ctx, ggml_tensor * x, const yolo_layer & L) {
    switch (L.act) {
        case act_type::leaky: return ggml_leaky_relu(ctx, x, L.slope, true);
        case act_type::silu:  return ggml_silu(ctx, x);
        case act_type::relu:  return ggml_relu(ctx, x);
        case act_type::none:  return x;
    }
    return x;
}

struct yolo_graph {
    ggml_context              * ctx   = nullptr;
    ggml_cgraph               * gf    = nullptr;
    ggml_tensor               * input = nullptr;
    std::vector<ggml_tensor *>  heads;           // [nx, ny, na*no] per level (raw conv output)
    // only when the model was converted with --graph-decode:
    ggml_tensor               * pred  = nullptr; // [no, N] == PyTorch torch.cat(z, 1) of shape [1, N, no]
    std::vector<ggml_tensor *>  grids;           // per level [2, nx, ny, 1], filled before compute
    std::vector<std::vector<float>> grid_data;
    ~yolo_graph() { if (ctx) ggml_free(ctx); }
};

// Inference branch of yolov7 Detect.forward for one level, inside the graph:
//   x = x.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
//   y = x.sigmoid()
//   y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + grid) * stride
//   y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid
//   z.append(y.view(bs, -1, no))
// ggml dims are listed fastest-first, so PyTorch (bs, na, ny, nx, no) is ggml [no, nx, ny, na].
static ggml_tensor * detect_decode(ggml_context * ctx, ggml_tensor * h, ggml_tensor * grid,
                                   ggml_tensor * anchor, int na, float stride) {
    const int64_t nx = h->ne[0], ny = h->ne[1], no = h->ne[2] / na;

    ggml_tensor * t = ggml_reshape_4d(ctx, h, nx, ny, no, na);          // .view(bs, na, no, ny, nx)
    t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3));               // .permute(0,1,3,4,2).contiguous()
    t = ggml_sigmoid(ctx, t);                                           // .sigmoid()

    // y[..., 0:2], y[..., 2:4], y[..., 4:] as views along the fastest dim
    auto slice = [&](int64_t start, int64_t len) {
        return ggml_cont(ctx, ggml_view_4d(ctx, t, len, nx, ny, na, t->nb[1], t->nb[2], t->nb[3],
                                           start * ggml_element_size(t)));
    };
    ggml_tensor * xy   = slice(0, 2);
    ggml_tensor * wh   = slice(2, 2);
    ggml_tensor * conf = slice(4, no - 4);

    // (s*2 - 0.5 + grid) * stride  ==  s*(2*stride) + (grid - 0.5)*stride   (grid input is pre-shifted)
    xy = ggml_add(ctx, ggml_scale(ctx, xy, 2.0f * stride), grid);       // grid [2, nx, ny, 1] broadcasts over na
    // (s*2)^2 * anchor  ==  4 * s^2 * anchor
    wh = ggml_scale(ctx, ggml_mul(ctx, ggml_sqr(ctx, wh), anchor), 4.0f); // anchor [2, 1, 1, na] broadcasts

    ggml_tensor * y = ggml_concat(ctx, ggml_concat(ctx, xy, wh, 0), conf, 0);  // back to [no, nx, ny, na]
    return ggml_reshape_2d(ctx, y, no, nx * ny * na);                   // .view(bs, -1, no)
}

static void build_graph(yolo_graph & G, const yolo_model & model, int size) {
    ggml_init_params ip = {
        ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), nullptr, /*no_alloc*/ true };
    G.ctx = ggml_init(ip);
    G.gf  = ggml_new_graph(G.ctx);
    ggml_context * ctx = G.ctx;

    G.input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, size, size, 3, 1);   // [W, H, C, N], planar RGB
    ggml_set_name(G.input, "input");
    ggml_set_input(G.input);

    std::vector<ggml_tensor *> out(model.layers.size(), nullptr);
    auto in = [&](int idx) { return idx < 0 ? G.input : out[idx]; };

    for (size_t i = 0; i < model.layers.size(); i++) {
        const yolo_layer & L = model.layers[i];
        ggml_tensor * x = in(L.from[0]);
        switch (L.type) {
            case layer_type::conv:
                // if (x->ne[2] != L.w->ne[2]) // waleed
                // {
                //     std:: printf("x->ne[2]=%d L.w->ne[2]=%d layer=%lu k=%d s=%d p=%d from=[", x->ne[2], L.w->ne[2], i, L.k, L.s, L.p);

                //     for (size_t j=0; j < L.from.size(); j++)
                //     {
                //         std::printf("%s%d", (j ? "," : ""), L.from[j]);
                //     }
                //     std::printf("]\n");
                // }
                x = apply_act(ctx, conv2d(ctx, x, L.w, L.b, L.s, L.p), L);
                break;
            case layer_type::add:   // act(from[0] + from[1] + ...), all inputs the same shape
                for (size_t j = 1; j < L.from.size(); j++) {
                    ggml_tensor * y = in(L.from[j]);
                    if (!ggml_are_same_shape(x, y))
                        throw std::runtime_error("add layer " + std::to_string(i) + ": inputs have different shapes");
                    x = ggml_add(ctx, x, y);
                }
                x = apply_act(ctx, x, L);
                break;
            case layer_type::concat:
                for (size_t j = 1; j < L.from.size(); j++) x = ggml_concat(ctx, x, in(L.from[j]), 2);
                break;
            case layer_type::maxpool:
                x = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, L.k, L.k, L.s, L.s, (float) L.p, (float) L.p);
                break;
            case layer_type::upsample:
                x = ggml_upscale(ctx, x, L.scale, GGML_SCALE_MODE_NEAREST);
                break;
            case layer_type::detect: {
                std::vector<ggml_tensor *> z;
                for (size_t j = 0; j < L.from.size(); j++) {
                    // {// waleed
                    //     std:: printf("layer=%lu k=%d s=%d p=%d from=%d det", j, L.k, L.s, L.p, L.from[j]);
                    // }
                    ggml_tensor * h = conv2d(ctx, in(L.from[j]), L.det_w[j], L.det_b[j], 1, 0);   // self.m[i](x[i])
                    ggml_set_name(h, ("head" + std::to_string(j)).c_str());
                    ggml_set_output(h);
                    ggml_build_forward_expand(G.gf, h);
                    G.heads.push_back(h);

                    if (L.decode) {
                        const int nx = (int) h->ne[0], ny = (int) h->ne[1];
                        const float stride = (float) model.strides[j];
                        ggml_tensor * grid = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, nx, ny, 1);
                        ggml_set_name(grid, ("grid" + std::to_string(j)).c_str());
                        ggml_set_input(grid);
                        G.grids.push_back(grid);
                        // _make_grid(nx, ny), pre-shifted: (grid - 0.5) * stride
                        std::vector<float> g((size_t) 2 * nx * ny);
                        for (int gy = 0; gy < ny; gy++)
                            for (int gx = 0; gx < nx; gx++) {
                                g[((size_t) gy * nx + gx) * 2 + 0] = (gx - 0.5f) * stride;
                                g[((size_t) gy * nx + gx) * 2 + 1] = (gy - 0.5f) * stride;
                            }
                        G.grid_data.push_back(std::move(g));
                        z.push_back(detect_decode(ctx, h, grid, L.det_anchor[j], model.na, stride));
                    }
                }
                if (L.decode) {
                    ggml_tensor * pred = z[0];
                    for (size_t j = 1; j < z.size(); j++) pred = ggml_concat(ctx, pred, z[j], 1);   // torch.cat(z, 1)
                    ggml_set_name(pred, "pred");
                    ggml_set_output(pred);
                    ggml_build_forward_expand(G.gf, pred);
                    G.pred = pred;
                }
                x = nullptr;
                break;
            }
        }
        out[i] = x;
    }
}

// ---------------------------------------------------------------------------
// pre / post processing
// ---------------------------------------------------------------------------

struct letterbox_info { float ratio; int pad_x, pad_y; };

// Same geometry as yolov7's letterbox(auto=False): keep aspect ratio, pad with 114.
// Output is planar RGB float in [0,1], i.e. exactly the [W, H, 3] ggml layout.
static letterbox_info letterbox(const uint8_t * img, int w, int h, int size, std::vector<float> & out) {
    const float r  = std::min((float) size / w, (float) size / h);
    const int   nw = (int) std::lround(w * r), nh = (int) std::lround(h * r);
    const float dw = (size - nw) / 2.0f, dh = (size - nh) / 2.0f;
    const int   left = (int) std::lround(dw - 0.1f), top = (int) std::lround(dh - 0.1f);

    out.assign((size_t) 3 * size * size, 114.0f / 255.0f);
    const float sx = (float) w / nw, sy = (float) h / nh;
    for (int y = 0; y < nh; y++) {
        // bilinear, half-pixel centers (matches cv2.INTER_LINEAR for upscaling)
        const float fy = std::max(0.0f, (y + 0.5f) * sy - 0.5f);
        const int   y0 = std::min((int) fy, h - 1), y1 = std::min(y0 + 1, h - 1);
        const float ay = fy - y0;
        for (int x = 0; x < nw; x++) {
            const float fx = std::max(0.0f, (x + 0.5f) * sx - 0.5f);
            const int   x0 = std::min((int) fx, w - 1), x1 = std::min(x0 + 1, w - 1);
            const float ax = fx - x0;
            for (int c = 0; c < 3; c++) {
                const float v00 = img[(y0 * w + x0) * 3 + c], v01 = img[(y0 * w + x1) * 3 + c];
                const float v10 = img[(y1 * w + x0) * 3 + c], v11 = img[(y1 * w + x1) * 3 + c];
                const float v = (v00 * (1 - ax) + v01 * ax) * (1 - ay) + (v10 * (1 - ax) + v11 * ax) * ay;
                out[(size_t) c * size * size + (size_t) (y + top) * size + (x + left)] = v / 255.0f;
            }
        }
    }
    return { r, left, top };
}

struct detection { float x0, y0, x1, y1, score; int cls; };

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static float iou(const detection & a, const detection & b) {
    const float ix = std::max(0.0f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    const float iy = std::max(0.0f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    const float inter = ix * iy;
    const float uni = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

// heads[l] holds [nx, ny, na*(5+nc)] raw logits; decoding follows yolov7 Detect.forward
static std::vector<detection> decode(const yolo_model & model, const std::vector<std::vector<float>> & heads,
                                     const std::vector<ggml_tensor *> & shapes, float conf_thres) {
    const int no = model.nc + 5;
    std::vector<detection> dets;
    for (size_t l = 0; l < heads.size(); l++) {
        const int nx = (int) shapes[l]->ne[0], ny = (int) shapes[l]->ne[1];
        const float stride = (float) model.strides[l];
        const float * d = heads[l].data();
        const size_t plane = (size_t) nx * ny;
        for (int a = 0; a < model.na; a++) {
            const float aw = model.anchors[(l * model.na + a) * 2 + 0];
            const float ah = model.anchors[(l * model.na + a) * 2 + 1];
            const float * p = d + (size_t) a * no * plane;
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    const size_t o = (size_t) y * nx + x;
                    const float obj = sigmoid(p[4 * plane + o]);
                    if (obj < conf_thres) continue;
                    int best = 0; float best_logit = p[5 * plane + o];
                    for (int c = 1; c < model.nc; c++) {
                        const float v = p[(5 + c) * plane + o];
                        if (v > best_logit) { best_logit = v; best = c; }
                    }
                    const float score = obj * sigmoid(best_logit);
                    if (score < conf_thres) continue;
                    const float cx = (sigmoid(p[0 * plane + o]) * 2.0f - 0.5f + x) * stride;
                    const float cy = (sigmoid(p[1 * plane + o]) * 2.0f - 0.5f + y) * stride;
                    const float bw = std::pow(sigmoid(p[2 * plane + o]) * 2.0f, 2.0f) * aw;
                    const float bh = std::pow(sigmoid(p[3 * plane + o]) * 2.0f, 2.0f) * ah;
                    dets.push_back({ cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, score, best });
                }
            }
        }
    }

    return dets;
}

// Same filtering for the in-graph output: pred is [no, N] with rows already decoded
// (cx, cy, w, h in pixels, sigmoid(obj), sigmoid(cls)...) -- the PyTorch model's output[0].
static std::vector<detection> filter_pred(const std::vector<float> & pred, int no, float conf_thres) {
    std::vector<detection> dets;
    const size_t n = pred.size() / no;
    for (size_t i = 0; i < n; i++) {
        const float * r = pred.data() + i * no;
        if (r[4] < conf_thres) continue;
        const int best = (int) (std::max_element(r + 5, r + no) - (r + 5));
        const float score = r[4] * r[5 + best];
        if (score < conf_thres) continue;
        dets.push_back({ r[0] - r[2] / 2, r[1] - r[3] / 2, r[0] + r[2] / 2, r[1] + r[3] / 2, score, best });
    }
    return dets;
}

// class-aware NMS
static std::vector<detection> nms(std::vector<detection> dets, float iou_thres) {
    std::sort(dets.begin(), dets.end(), [](const detection & a, const detection & b) { return a.score > b.score; });
    std::vector<detection> keep;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (!removed[j] && dets[j].cls == dets[i].cls && iou(dets[i], dets[j]) > iou_thres) removed[j] = true;
        }
    }
    return keep;
}

static void draw_box(uint8_t * img, int w, int h, const detection & d, const uint8_t color[3], int thick) {
    const int x0 = std::clamp((int) d.x0, 0, w - 1), x1 = std::clamp((int) d.x1, 0, w - 1);
    const int y0 = std::clamp((int) d.y0, 0, h - 1), y1 = std::clamp((int) d.y1, 0, h - 1);
    auto put = [&](int x, int y) { if (x >= 0 && x < w && y >= 0 && y < h) std::memcpy(img + (y * w + x) * 3, color, 3); };
    for (int t = 0; t < thick; t++) {
        for (int x = x0; x <= x1; x++) { put(x, y0 + t); put(x, y1 - t); }
        for (int y = y0; y <= y1; y++) { put(x0 + t, y); put(x1 - t, y); }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

struct params {
    std::string model, image, output = "predictions.jpg", dump_dir;
    int   n_threads  = (int) std::max(1u, std::thread::hardware_concurrency());
    int   size       = 0;      // 0 = model default (640)
    float conf_thres = 0.25f, iou_thres = 0.45f;
    bool  use_gpu    = false;
    int   bench      = 0;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m model.gguf -i image.jpg [options]\n"
        "  -o, --output FILE   output image with boxes (default predictions.jpg)\n"
        "  -t, --threads N     CPU threads (default: all cores)\n"
        "  -s, --size N        network input size, multiple of 32 (default 640)\n"
        "  --conf F            confidence threshold (default 0.25)\n"
        "  --iou F             NMS IoU threshold (default 0.45)\n"
        "  --gpu               use the best available GPU backend (CUDA/Metal/Vulkan...)\n"
        "  --bench N           run N extra timed inferences\n"
        "  --dump DIR          write input and raw head outputs as .bin (for verify.py)\n", argv0);
}

static bool parse_args(int argc, char ** argv, params & p) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if      (a == "-m" || a == "--model")   p.model      = next();
        else if (a == "-i" || a == "--image")   p.image      = next();
        else if (a == "-o" || a == "--output")  p.output     = next();
        else if (a == "-t" || a == "--threads") p.n_threads  = std::atoi(next());
        else if (a == "-s" || a == "--size")    p.size       = std::atoi(next());
        else if (a == "--conf")                 p.conf_thres = (float) std::atof(next());
        else if (a == "--iou")                  p.iou_thres  = (float) std::atof(next());
        else if (a == "--gpu")                  p.use_gpu    = true;
        else if (a == "--bench")                p.bench      = std::atoi(next());
        else if (a == "--dump")                 p.dump_dir   = next();
        else if (a == "-h" || a == "--help")    return false;
        else throw std::runtime_error("unknown argument " + a);
    }
    return !p.model.empty() && !p.image.empty();
}

static void write_bin(const std::string & path, const float * data, size_t n) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
}

int main(int argc, char ** argv) {
    params p;
    try {
        if (!parse_args(argc, argv, p)) { usage(argv[0]); return 1; }

        yolo_model model;
        load_model(model, p.model.c_str(), p.use_gpu, p.n_threads);
        const int size = p.size > 0 ? p.size : model.input_size;
        const int max_stride = *std::max_element(model.strides.begin(), model.strides.end());
        if (size % max_stride != 0) throw std::runtime_error("input size must be a multiple of " + std::to_string(max_stride));

        int w, h, c;
        uint8_t * img = stbi_load(p.image.c_str(), &w, &h, &c, 3);
        if (!img) throw std::runtime_error("failed to load image " + p.image);

        std::vector<float> input;
        const letterbox_info lb = letterbox(img, w, h, size, input);

        yolo_graph G;
        build_graph(G, model, size);
        if (!ggml_backend_sched_alloc_graph(model.sched, G.gf)) throw std::runtime_error("failed to allocate compute graph");

        auto run = [&]() {
            ggml_backend_tensor_set(G.input, input.data(), 0, ggml_nbytes(G.input));
            for (size_t j = 0; j < G.grids.size(); j++)
                ggml_backend_tensor_set(G.grids[j], G.grid_data[j].data(), 0, ggml_nbytes(G.grids[j]));
            if (ggml_backend_sched_graph_compute(model.sched, G.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed");
        };

        auto t0 = std::chrono::high_resolution_clock::now();
        run();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::printf("inference (%dx%d, %d threads): %.1f ms\n", size, size, p.n_threads,
                    std::chrono::duration<double, std::milli>(t1 - t0).count());

        if (p.bench > 0) {
            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < p.bench; i++) run();
            t1 = std::chrono::high_resolution_clock::now();
            std::printf("bench: %.1f ms / image over %d runs\n",
                        std::chrono::duration<double, std::milli>(t1 - t0).count() / p.bench, p.bench);
        }

        std::vector<std::vector<float>> heads(G.heads.size());
        for (size_t l = 0; l < G.heads.size(); l++) {
            heads[l].resize(ggml_nelements(G.heads[l]));
            ggml_backend_tensor_get(G.heads[l], heads[l].data(), 0, ggml_nbytes(G.heads[l]));
        }

        std::vector<float> pred;
        if (G.pred) {
            pred.resize(ggml_nelements(G.pred));
            ggml_backend_tensor_get(G.pred, pred.data(), 0, ggml_nbytes(G.pred));
            std::printf("graph output 'pred': [1, %lld, %lld] (decoded in-graph, like torch.cat(z, 1))\n",
                        (long long) G.pred->ne[1], (long long) G.pred->ne[0]);
        }

        if (!p.dump_dir.empty()) {
            write_bin(p.dump_dir + "/input.bin", input.data(), input.size());
            for (size_t l = 0; l < heads.size(); l++)
                write_bin(p.dump_dir + "/head" + std::to_string(l) + ".bin", heads[l].data(), heads[l].size());
            if (G.pred) write_bin(p.dump_dir + "/pred.bin", pred.data(), pred.size());
        }

        // waleed
        std::vector<detection> dets = G.pred
            ? nms(filter_pred(pred, model.nc + 5, p.conf_thres), p.iou_thres)
            : nms(decode(model, heads, G.heads, p.conf_thres), p.iou_thres);

        // undo letterbox and draw
        for (auto & d : dets) {
            d.x0 = std::clamp((d.x0 - lb.pad_x) / lb.ratio, 0.0f, (float) w);
            d.x1 = std::clamp((d.x1 - lb.pad_x) / lb.ratio, 0.0f, (float) w);
            d.y0 = std::clamp((d.y0 - lb.pad_y) / lb.ratio, 0.0f, (float) h);
            d.y1 = std::clamp((d.y1 - lb.pad_y) / lb.ratio, 0.0f, (float) h);
            uint8_t color[3] = { (uint8_t) (37 * d.cls % 256), (uint8_t) (17 * d.cls * 3 % 256 + 60), (uint8_t) (29 * d.cls * 5 % 256) };
            draw_box(img, w, h, d, color, std::max(2, (w + h) / 600));
            const char * label = d.cls < (int) model.names.size() ? model.names[d.cls].c_str() : "?";
            std::printf("%-14s %.2f  [%4.0f, %4.0f, %4.0f, %4.0f]\n", label, d.score, d.x0, d.y0, d.x1, d.y1);
        }
        std::printf("%zu detections\n", dets.size());

        if (!stbi_write_jpg(p.output.c_str(), w, h, 3, img, 90)) throw std::runtime_error("failed to write " + p.output);
        std::printf("saved %s\n", p.output.c_str());
        stbi_image_free(img);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
