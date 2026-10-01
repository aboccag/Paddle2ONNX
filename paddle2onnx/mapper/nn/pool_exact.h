// Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// Pooling that ONNX's MaxPool / AveragePool attributes cannot say, emitted
// exactly. Shared by pool2d and pool3d.
//
// Three Paddle behaviours have no ONNX attribute, each measured against
// Paddle 3.3 on a 1056-config grid (ceil_mode x exclusive x kernel x stride x
// padding x size, 2-D and 3-D):
//
//  * Under ceil_mode Paddle keeps every window the ceil formula counts, even a
//    last one that starts in the right padding. onnxruntime drops that window
//    -- PyTorch's rule, `if ((out - 1) * stride >= in + pad_begin) --out` --
//    and returns one row or column fewer (66 grid configs: wrong shape).
//  * A non-exclusive average divides by the window clipped to the *explicit*
//    padding, not to the ceil_mode overflow beyond it. onnxruntime's
//    count_include_pad=1 counts the overflow too (41 configs: silently wrong
//    values, the converted model runs).
//  * Padding as large as the kernel is legal in Paddle; ONNX requires
//    pads < kernel_shape. The mapper's pre-pad workaround padded with 0, which
//    onnxruntime then fuses back into the pool's pads and refuses at load --
//    and 0 is the wrong value for a max pool anyway (144 configs).
//
// What Paddle returns for a window holding no input was measured too. Max:
// -FLT_MAX, its initial value. Average: Paddle divides the (zero) sum by the
// product of per-axis counts it never clamps -- exclusive
// min(start+k, in) - max(start, 0), non-exclusive min(start+k, in+pad) -
// start -- so a window in the leading padding is 0/0 = NaN, and one that the
// ceil_mode overflow places past the input is 0/(negative) = -0. The exact
// emission below reproduces each of these.
//
// The exact form needs every spatial extent to be known at conversion time;
// callers keep their previous emission for dynamic shapes.

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {
namespace pool_exact {

struct Axis {
  int64_t in;         // input extent (> 0: static)
  int64_t k;          // kernel
  int64_t s;          // stride
  int64_t pad_begin;  // explicit padding, Paddle's
  int64_t pad_end;
};

inline int64_t FloorDiv(int64_t a, int64_t b) {
  return a >= 0 ? a / b : -((-a + b - 1) / b);
}

// Output extent Paddle computes: the plain floor / ceil formula.
inline int64_t PaddleOutSize(const Axis& a, bool ceil_mode) {
  int64_t span = a.in + a.pad_begin + a.pad_end - a.k;
  return (ceil_mode ? -FloorDiv(-span, a.s) : FloorDiv(span, a.s)) + 1;
}

// Output extent onnxruntime computes from the same attributes: under
// ceil_mode it drops a last window starting in the right padding.
inline int64_t OrtOutSize(const Axis& a, bool ceil_mode) {
  int64_t out = PaddleOutSize(a, ceil_mode);
  if (ceil_mode && (out - 1) * a.s >= a.in + a.pad_begin) --out;
  return out;
}

// Does the plain MaxPool/AveragePool emission disagree with Paddle here?
// Only true for configurations the grid showed the plain emission gets wrong,
// so every model that converts correctly today keeps its graph byte for byte.
inline bool NeedsExact(const std::vector<Axis>& axes, bool ceil_mode,
                       bool is_avg, bool exclusive) {
  for (const auto& a : axes) {
    if (a.in <= 0 || a.s <= 0 || a.k <= 0) return false;  // dynamic: unknowable
  }
  for (const auto& a : axes) {
    if (a.pad_begin >= a.k || a.pad_end >= a.k) return true;
    if (ceil_mode && PaddleOutSize(a, true) != OrtOutSize(a, true)) {
      return true;
    }
    if (ceil_mode && is_avg && !exclusive) {
      int64_t out = PaddleOutSize(a, true);
      int64_t last_end = (out - 1) * a.s - a.pad_begin + a.k;
      // onnxruntime counts the ceil_mode overflow into the divisor.
      if (last_end > a.in + a.pad_end) return true;
    }
  }
  return false;
}

// Paddle's per-axis divisor for window i, unclamped exactly as Paddle leaves
// it (it may be zero or negative for a window holding no input).
inline int64_t WindowCount(const Axis& a, int64_t i, bool exclusive) {
  int64_t start = i * a.s - a.pad_begin;
  int64_t end = start + a.k;
  if (exclusive) return std::min(end, a.in) - std::max(start, int64_t(0));
  return std::min(end, a.in + a.pad_end) - start;
}

// Input elements window i actually covers on this axis.
inline int64_t RealOverlap(const Axis& a, int64_t i) {
  int64_t start = i * a.s - a.pad_begin;
  int64_t n = std::min(start + a.k, a.in) - std::max(start, int64_t(0));
  return n > 0 ? n : 0;
}

// Emits Paddle's pool over float tensor `x` (rank 2 + axes.size()) and returns
// the result's name, or "" when the plan is not exact (the caller then keeps
// its previous emission). `exclusive` is Paddle's attribute, not ONNX's.
inline std::string EmitExactPool(OnnxHelper* helper, const std::string& x,
                                 const std::vector<Axis>& axes, bool ceil_mode,
                                 bool is_max, bool exclusive) {
  const size_t nd = axes.size();
  std::vector<int64_t> outs, pad_end_x, kernel, strides;
  for (const auto& a : axes) {
    int64_t out = PaddleOutSize(a, ceil_mode);
    if (out <= 0) return "";
    // End padding that makes the floor formula produce exactly `out` windows,
    // all lying inside the padded tensor: the ceil_mode overflow becomes
    // explicit padding, so no window is left for onnxruntime to drop.
    int64_t pe = (out - 1) * a.s + a.k - a.in - a.pad_begin;
    if (pe < 0) pe = 0;
    if (FloorDiv(a.in + a.pad_begin + pe - a.k, a.s) + 1 != out) return "";
    outs.push_back(out);
    pad_end_x.push_back(pe);
    kernel.push_back(a.k);
    strides.push_back(a.s);
  }

  // Max pads with the lowest float, which is Paddle's own initial value
  // (-FLT_MAX) and so also what an all-padding window returns. Average pads
  // with 1, not 0, and the padding's contribution is removed below: a Pad of
  // 0 is what onnxruntime fuses back into the pool's pads, re-creating
  // pads >= kernel_shape and refusing the model at load. A non-zero value is
  // never fused.
  std::vector<int64_t> pads(2 * (nd + 2), 0);
  for (size_t d = 0; d < nd; ++d) {
    pads[2 + d] = axes[d].pad_begin;
    pads[nd + 2 + 2 + d] = pad_end_x[d];
  }
  float pad_value = is_max ? std::numeric_limits<float>::lowest() : 1.0f;
  std::string padded;
  if (helper->GetOpsetVersion() >= 11) {
    std::string pads_node =
        helper->Constant(ONNX_NAMESPACE::TensorProto::INT64, pads);
    std::string value_node = helper->Constant(
        ONNX_NAMESPACE::TensorProto::FLOAT, std::vector<float>{pad_value});
    padded = helper->MakeNode("Pad", {x, pads_node, value_node})->output(0);
  } else {
    auto pad = helper->MakeNode("Pad", {x});
    AddAttribute(pad, "mode", std::string("constant"));
    AddAttribute(pad, "pads", pads);
    AddAttribute(pad, "value", pad_value);
    padded = pad->output(0);
  }

  auto pool = helper->MakeNode(is_max ? "MaxPool" : "AveragePool", {padded});
  AddAttribute(pool, "kernel_shape", kernel);
  AddAttribute(pool, "strides", strides);
  AddAttribute(pool, "pads", std::vector<int64_t>(2 * nd, 0));
  if (is_max) return pool->output(0);

  // Every window now lies inside the padded tensor, so AveragePool returns
  // (sum_input + 1 * n_pad) / K over the full kernel volume K, where n_pad is
  // the window's padded elements. Paddle returns sum_input / count, with count
  // its own divisor (the product of WindowCount). So
  //   out = mean * (K / count) - n_pad / count
  // with both coefficients constant. A window holding no input has
  // sum_input == 0, so Paddle returns 0 / count: NaN when count == 0
  // (emitted as inf - inf), and a zero otherwise (emitted as 0 * mean - 0).
  int64_t volume = 1;
  for (auto k : kernel) volume *= k;
  int64_t total = 1;
  for (auto o : outs) total *= o;
  std::vector<float> scale(total, 1.0f), offset(total, 0.0f);
  for (int64_t flat = 0; flat < total; ++flat) {
    int64_t rem = flat, count = 1, real = 1;
    for (int64_t d = static_cast<int64_t>(nd) - 1; d >= 0; --d) {
      int64_t i = rem % outs[d];
      rem /= outs[d];
      count *= WindowCount(axes[d], i, exclusive);
      real *= RealOverlap(axes[d], i);
    }
    const float inf = std::numeric_limits<float>::infinity();
    if (real > 0) {
      const int64_t n_pad = volume - real;
      scale[flat] = static_cast<float>(volume) / count;
      offset[flat] = static_cast<float>(n_pad) / count;
    } else if (count == 0) {
      scale[flat] = inf;  // mean is 1 here: inf - inf = NaN, as Paddle's 0/0
      offset[flat] = inf;
    } else {
      scale[flat] = 0.0f;  // 0 / count, count != 0
      offset[flat] = 0.0f;
    }
  }
  std::vector<int64_t> coef_shape = {1, 1};
  coef_shape.insert(coef_shape.end(), outs.begin(), outs.end());
  std::string scale_node = helper->Constant(
      coef_shape, ONNX_NAMESPACE::TensorProto::FLOAT, scale);
  std::string offset_node = helper->Constant(
      coef_shape, ONNX_NAMESPACE::TensorProto::FLOAT, offset);
  std::string scaled =
      helper->MakeNode("Mul", {pool->output(0), scale_node})->output(0);
  return helper->MakeNode("Sub", {scaled, offset_node})->output(0);
}

// Adaptive pooling with Paddle's windows [floor(i*in/out), ceil((i+1)*in/out))
// on every spatial axis, exact for any in/out. Max and mean are both separable
// over a box (max of maxes; mean of equal-length means), so each axis is
// reduced in turn: one Slice + Reduce per output index, then a Concat. Axes
// where in == out are the identity and are skipped.
inline std::string EmitAdaptiveBySlices(OnnxHelper* helper,
                                        const std::string& x,
                                        const std::vector<int64_t>& in,
                                        const std::vector<int64_t>& out,
                                        bool is_max) {
  std::string cur = x;
  const char* reduce = is_max ? "ReduceMax" : "ReduceMean";
  for (size_t d = 0; d < in.size(); ++d) {
    if (in[d] == out[d]) continue;
    int64_t axis = static_cast<int64_t>(d) + 2;
    std::vector<std::string> pieces;
    for (int64_t i = 0; i < out[d]; ++i) {
      int64_t start = (i * in[d]) / out[d];
      int64_t end = ((i + 1) * in[d] + out[d] - 1) / out[d];
      std::string sliced = helper->Slice(cur, {axis}, {start}, {end});
      std::shared_ptr<ONNX_NAMESPACE::NodeProto> node;
      if (helper->GetOpsetVersion() >= 18) {
        std::string axes_node = helper->Constant(
            ONNX_NAMESPACE::TensorProto::INT64, std::vector<int64_t>{axis});
        node = helper->MakeNode(reduce, {sliced, axes_node});
      } else {
        node = helper->MakeNode(reduce, {sliced});
        AddAttribute(node, "axes", std::vector<int64_t>{axis});
      }
      AddAttribute(node, "keepdims", static_cast<int64_t>(1));
      pieces.push_back(node->output(0));
    }
    cur = pieces.size() == 1 ? pieces[0] : helper->Concat(pieces, axis);
  }
  return cur;
}

}  // namespace pool_exact
}  // namespace paddle2onnx
