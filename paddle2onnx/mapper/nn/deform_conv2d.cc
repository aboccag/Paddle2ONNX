// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/mapper/nn/deform_conv2d.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(deformable_conv, DeformConv2dMapper)
REGISTER_PIR_MAPPER(deformable_conv, DeformConv2dMapper)

int32_t DeformConv2dMapper::GetMinOpsetVersion(bool verbose) {
  auto kernel_info = GetInput("Filter");
  if (kernel_info[0].shape.size() != 4) {
    Error() << "deformable_conv only supports a 4-D filter, but its rank is "
            << kernel_info[0].shape.size() << "." << std::endl;
    return -1;
  }
  if (groups_ != 1) {
    Error() << "deformable_conv with groups != 1 is not supported yet, but got "
            << groups_ << "." << std::endl;
    return -1;
  }
  if (!HasInput("Mask")) {
    Error() << "deformable_conv without a modulation mask is not supported yet."
            << std::endl;
    return -1;
  }
  // GridSample arrived in opset 16.
  Logger(verbose, 16) << RequireOpset(16) << std::endl;
  return 16;
}

void DeformConv2dMapper::Opset16() {
  auto input_info = GetInput("Input");
  auto kernel_info = GetInput("Filter");
  auto offset_info = GetInput("Offset");
  auto mask_info = GetInput("Mask");
  auto output_info = GetOutput("Output");

  const int64_t out_channels = kernel_info[0].shape[0];
  const int64_t in_channels = kernel_info[0].shape[1] * groups_;
  const int64_t kh = kernel_info[0].shape[2];
  const int64_t kw = kernel_info[0].shape[3];
  const int64_t taps = kh * kw;
  const int64_t dg = deformable_groups_;
  const int64_t c_per_dg = in_channels / dg;

  const int64_t stride_h = strides_[0], stride_w = strides_[1];
  const int64_t dil_h = dilations_[0], dil_w = dilations_[1];
  const int64_t pad_h = paddings_[0];
  const int64_t pad_w = paddings_.size() == 2 ? paddings_[1] : paddings_[2];

  const int32_t dtype = input_info[0].dtype;
  auto onnx_dtype = GetOnnxDtype(dtype);
  auto i64 = [&](const std::vector<int64_t>& v) {
    return helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, v);
  };
  auto f1 = [&](float v) { return helper_->Constant({1}, onnx_dtype, v); };
  auto mul = [&](const std::string& a, const std::string& b) {
    return helper_->MakeNode("Mul", {a, b})->output(0);
  };
  auto add = [&](const std::string& a, const std::string& b) {
    return helper_->MakeNode("Add", {a, b})->output(0);
  };
  auto sub = [&](const std::string& a, const std::string& b) {
    return helper_->MakeNode("Sub", {a, b})->output(0);
  };

  // ── dynamic sizes ────────────────────────────────────────────────────────
  auto x_shape = helper_->MakeNode("Shape", {input_info[0].name})->output(0);
  auto off_shape = helper_->MakeNode("Shape", {offset_info[0].name})->output(0);
  auto batch = helper_->Slice(off_shape, {0}, {0}, {1});
  auto out_h = helper_->Slice(off_shape, {0}, {2}, {3});
  auto out_w = helper_->Slice(off_shape, {0}, {3}, {4});
  auto in_h_f = helper_->AutoCast(helper_->Slice(x_shape, {0}, {2}, {3}),
                                  P2ODataType::INT64, dtype);
  auto in_w_f = helper_->AutoCast(helper_->Slice(x_shape, {0}, {3}, {4}),
                                  P2ODataType::INT64, dtype);

  // ── sampling grid ────────────────────────────────────────────────────────
  // phi's kernel samples at
  //   y = h_out * stride_h - pad_h + p * dilation_h + offset_y
  //   x = w_out * stride_w - pad_w + q * dilation_w + offset_x
  // with bilinear interpolation and zeros outside the image.
  auto arange = [&](const std::string& count) {
    auto zero = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                  std::vector<int64_t>(1, 0));
    auto one = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                 std::vector<int64_t>(1, 1));
    auto rng = helper_->MakeNode("Range",
                                 {helper_->Squeeze(zero, {0}),
                                  helper_->Squeeze(count, {0}),
                                  helper_->Squeeze(one, {0})})
                   ->output(0);
    return helper_->AutoCast(rng, P2ODataType::INT64, dtype);
  };
  auto rows = sub(mul(arange(out_h), f1(static_cast<float>(stride_h))),
                  f1(static_cast<float>(pad_h)));
  auto cols = sub(mul(arange(out_w), f1(static_cast<float>(stride_w))),
                  f1(static_cast<float>(pad_w)));

  std::vector<float> tap_y(taps), tap_x(taps);
  for (int64_t p = 0; p < kh; ++p) {
    for (int64_t q = 0; q < kw; ++q) {
      tap_y[p * kw + q] = static_cast<float>(p * dil_h);
      tap_x[p * kw + q] = static_cast<float>(q * dil_w);
    }
  }
  std::vector<int64_t> tap_shape = {taps, 1, 1};
  auto tap_y_c = helper_->Constant(tap_shape, onnx_dtype, tap_y);
  auto tap_x_c = helper_->Constant(tap_shape, onnx_dtype, tap_x);
  // [taps, out_h, 1] and [taps, 1, out_w]
  auto base_y = add(tap_y_c, helper_->Reshape(rows, {1, -1, 1}));
  auto base_x = add(tap_x_c, helper_->Reshape(cols, {1, 1, -1}));

  // offset channels are laid out as [dg][taps][2], y first.
  auto off = helper_->MakeNode(
      "Reshape",
      {offset_info[0].name,
       helper_->Concat({batch, i64({dg, taps, 2}), out_h, out_w}, 0)})
      ->output(0);  // [N, dg, taps, 2, out_h, out_w]
  auto off_parts = helper_->Split(off, {1, 1}, 3);
  auto off_y = helper_->Squeeze(off_parts[0], {3});  // [N, dg, taps, oh, ow]
  auto off_x = helper_->Squeeze(off_parts[1], {3});

  auto sample_y = add(off_y, base_y);
  auto sample_x = add(off_x, base_x);

  // Normalise to [-1, 1] the way GridSample reads it with align_corners=1.
  auto two = f1(2.0f), one_f = f1(1.0f);
  auto gy = sub(helper_->MakeNode(
                    "Div", {mul(two, sample_y), sub(in_h_f, one_f)})->output(0),
                one_f);
  auto gx = sub(helper_->MakeNode(
                    "Div", {mul(two, sample_x), sub(in_w_f, one_f)})->output(0),
                one_f);
  // gx / gy are [N, dg, taps, out_h, out_w]; the helper rejects negative axes.
  auto grid = helper_->Concat(
      {helper_->Unsqueeze(gx, {5}), helper_->Unsqueeze(gy, {5})}, 5);
  // [N * dg, taps, out_h * out_w, 2]
  auto n_dg = mul(batch, i64({dg}));
  auto hw = mul(out_h, out_w);
  auto grid_r = helper_->MakeNode(
      "Reshape", {grid, helper_->Concat({n_dg, i64({taps}), hw, i64({2})}, 0)})
      ->output(0);

  // ── sample ───────────────────────────────────────────────────────────────
  auto x_r = helper_->MakeNode(
      "Reshape",
      {input_info[0].name,
       helper_->Concat({n_dg,
                        i64({c_per_dg}),
                        helper_->Slice(x_shape, {0}, {2}, {3}),
                        helper_->Slice(x_shape, {0}, {3}, {4})},
                       0)})
      ->output(0);
  auto sampled = helper_->MakeNode("GridSample", {x_r, grid_r});
  AddAttribute(sampled, "mode", "bilinear");
  AddAttribute(sampled, "padding_mode", "zeros");
  AddAttribute(sampled, "align_corners", static_cast<int64_t>(1));
  // [N, in_channels, taps, out_h * out_w]
  auto cols_t = helper_->MakeNode(
      "Reshape",
      {sampled->output(0),
       helper_->Concat({batch, i64({in_channels, taps}), hw}, 0)})
      ->output(0);

  // ── modulation ───────────────────────────────────────────────────────────
  // mask is one value per (dg, tap); every channel of a group shares it.
  auto mask_r = helper_->MakeNode(
      "Reshape",
      {mask_info[0].name,
       helper_->Concat({batch, i64({dg, 1, taps}), hw}, 0)})
      ->output(0);
  auto mask_e = helper_->MakeNode(
      "Expand",
      {mask_r,
       helper_->Concat({batch, i64({dg, c_per_dg, taps}), hw}, 0)})
      ->output(0);
  auto mask_f = helper_->MakeNode(
      "Reshape",
      {mask_e, helper_->Concat({batch, i64({in_channels, taps}), hw}, 0)})
      ->output(0);
  auto modulated = mul(cols_t, mask_f);

  // ── contract with the filter ─────────────────────────────────────────────
  auto patches = helper_->MakeNode(
      "Reshape",
      {modulated, helper_->Concat({batch, i64({in_channels * taps}), hw}, 0)})
      ->output(0);
  auto weight = helper_->Reshape(kernel_info[0].name,
                                 {out_channels, in_channels * taps});
  auto out = helper_->MakeNode("MatMul", {weight, patches})->output(0);
  helper_->MakeNode("Reshape",
                    {out,
                     helper_->Concat({batch, i64({out_channels}), out_h, out_w},
                                     0)},
                    {output_info[0].name});
}
}  // namespace paddle2onnx
