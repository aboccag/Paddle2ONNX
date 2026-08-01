// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/mapper/detection/distribute_fpn_proposals.h"

#include <cmath>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(distribute_fpn_proposals, DistributeFpnProposalsMapper)

int32_t DistributeFpnProposalsMapper::GetMinOpsetVersion(bool verbose) {
  // NonZero (opset 9) and TopK with a tensor `k` (opset 10); 11 keeps this
  // mapper aligned with GenerateProposalsMapper, which the same graphs need.
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void DistributeFpnProposalsMapper::Opset11() {
  auto rois_info = GetInput(0);  // [R, 4]

  auto multi_rois_info = GetOutput(0);      // vector, one [-1, 4] per level
  auto multi_num_info = GetOutput(1);       // vector, one [-1] per level
  auto restore_index_info = GetOutput(2);   // [R, 1]

  int32_t dtype = rois_info[0].dtype;
  auto onnx_dtype = GetOnnxDtype(dtype);
  const int64_t num_levels = max_level_ - min_level_ + 1;

  auto f_const = [&](float v) {
    return helper_->Constant({1}, onnx_dtype, v);
  };

  // ── Per-RoI target level ─────────────────────────────────────────────────
  //   scale = sqrt(area)
  //   level = floor(log2(scale / refer_scale + 1e-8) + refer_level)
  //   level = clip(level, min_level, max_level)
  auto xy = helper_->Split(rois_info[0].name, std::vector<int64_t>(4, 1), 1);
  auto sub = [&](const std::string& x, const std::string& y) {
    return helper_->MakeNode("Sub", {x, y})->output(0);
  };
  auto add = [&](const std::string& x, const std::string& y) {
    return helper_->MakeNode("Add", {x, y})->output(0);
  };

  auto w = sub(xy[2], xy[0]);
  auto h = sub(xy[3], xy[1]);
  if (pixel_offset_) {
    auto one = f_const(1.0f);
    w = add(w, one);
    h = add(h, one);
  }
  auto area = helper_->MakeNode("Mul", {w, h})->output(0);
  auto scale = helper_->MakeNode("Sqrt", {area})->output(0);

  auto ratio = helper_->MakeNode(
                          "Div",
                          {scale, f_const(static_cast<float>(refer_scale_))})
                  ->output(0);
  ratio = add(ratio, f_const(1e-8f));
  // ONNX has only a natural logarithm; log2(x) = ln(x) / ln(2).
  auto log2 = helper_->MakeNode(
                          "Mul",
                          {helper_->MakeNode("Log", {ratio})->output(0),
                           f_const(1.4426950408889634f)})
                  ->output(0);
  auto target = helper_->MakeNode(
                            "Floor",
                            {add(log2, f_const(static_cast<float>(refer_level_)))})
                    ->output(0);
  target = helper_->MakeNode(
                       "Max",
                       {helper_->MakeNode(
                                    "Min",
                                    {target,
                                     f_const(static_cast<float>(max_level_))})
                            ->output(0),
                        f_const(static_cast<float>(min_level_))})
               ->output(0);
  auto target_1d = helper_->Reshape(target, {-1});

  // ── Split the RoIs across levels ─────────────────────────────────────────
  std::vector<std::string> per_level_indices;
  for (int64_t lvl = 0; lvl < num_levels; ++lvl) {
    auto lvl_c = f_const(static_cast<float>(min_level_ + lvl));
    auto mask = helper_->MakeNode("Equal", {target_1d, lvl_c})->output(0);
    auto idx = helper_->Reshape(
        helper_->MakeNode("NonZero", {mask})->output(0), {-1});
    per_level_indices.push_back(idx);

    auto lvl_rois = helper_->MakeNode("Gather", {rois_info[0].name, idx});
    AddAttribute(lvl_rois, "axis", static_cast<int64_t>(0));
    helper_->MakeNode(
        "Identity", {lvl_rois->output(0)}, {multi_rois_info[lvl].name});

    // Batch size is 1 here, so the per-level RoI count is a single element.
    auto count = helper_->Slice(
        helper_->MakeNode("Shape", {lvl_rois->output(0)})->output(0),
        {0}, {0}, {1});
    helper_->AutoCast(count,
                      multi_num_info[lvl].name,
                      P2ODataType::INT64,
                      multi_num_info[lvl].dtype);
  }

  // ── Restore index ────────────────────────────────────────────────────────
  // Concatenating the per-level index lists gives the permutation `perm` with
  // concat(level_rois) == rois[perm]. Paddle returns the *inverse*, so that
  // gather(concat(level_rois), restore_index) recovers the original order.
  // Since perm is a permutation of 0..R-1, sorting it ascending yields the
  // inverse permutation as TopK's index output.
  auto perm = helper_->Concat(per_level_indices, 0);
  auto num_rois = helper_->MakeNode("Shape", {perm})->output(0);
  auto sorted = helper_->MakeNode("TopK", {perm, num_rois}, 2);
  AddAttribute(sorted, "axis", static_cast<int64_t>(0));
  AddAttribute(sorted, "largest", static_cast<int64_t>(0));
  AddAttribute(sorted, "sorted", static_cast<int64_t>(1));

  auto restore = helper_->Reshape(sorted->output(1), {-1, 1});
  helper_->AutoCast(restore,
                    restore_index_info[0].name,
                    P2ODataType::INT64,
                    restore_index_info[0].dtype);
}

}  // namespace paddle2onnx
