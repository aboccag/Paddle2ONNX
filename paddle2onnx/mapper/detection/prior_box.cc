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

#include "paddle2onnx/mapper/detection/prior_box.h"

#include <cmath>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(prior_box, PriorBoxMapper)

namespace {
// Mirrors phi::funcs::ExpandAspectRatios: 1.0 always comes first, duplicates
// are dropped, and each kept ratio is followed by its reciprocal when flipping.
std::vector<float> ExpandAspectRatios(const std::vector<float>& input,
                                      bool flip) {
  constexpr float epsilon = 1e-6f;
  std::vector<float> output;
  output.push_back(1.0f);
  for (size_t i = 0; i < input.size(); ++i) {
    float ar = input[i];
    bool already_exist = false;
    for (size_t j = 0; j < output.size(); ++j) {
      if (std::fabs(ar - output[j]) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output.push_back(ar);
      if (flip) output.push_back(1.0f / ar);
    }
  }
  return output;
}
}  // namespace

int32_t PriorBoxMapper::GetMinOpsetVersion(bool verbose) {
  auto input_info = GetInput(0);
  auto image_info = GetInput(1);
  if (input_info[0].Rank() != 4 || image_info[0].Rank() != 4) {
    Error() << "prior_box expects 4-D input and image tensors, but their ranks "
               "are "
            << input_info[0].Rank() << " and " << image_info[0].Rank() << "."
            << std::endl;
    return -1;
  }
  // Only the batch dimension may be dynamic: the priors depend on the spatial
  // sizes, which are fixed for every SSD variant.
  if (input_info[0].shape[2] <= 0 || input_info[0].shape[3] <= 0 ||
      image_info[0].shape[2] <= 0 || image_info[0].shape[3] <= 0) {
    Error() << "prior_box requires static feature-map and image spatial sizes, "
               "but got feature "
            << input_info[0].shape[2] << "x" << input_info[0].shape[3]
            << " and image " << image_info[0].shape[2] << "x"
            << image_info[0].shape[3] << "." << std::endl;
    return -1;
  }
  return 7;
}

void PriorBoxMapper::Opset7() {
  auto input_info = GetInput(0);
  auto image_info = GetInput(1);
  auto boxes_info = GetOutput(0);
  auto vars_info = GetOutput(1);

  const int64_t feature_height = input_info[0].shape[2];
  const int64_t feature_width = input_info[0].shape[3];
  const float img_height = static_cast<float>(image_info[0].shape[2]);
  const float img_width = static_cast<float>(image_info[0].shape[3]);

  // A non-positive step means "derive it from the image and feature sizes".
  float step_width = step_w_ > 0 ? step_w_ : img_width / feature_width;
  float step_height = step_h_ > 0 ? step_h_ : img_height / feature_height;

  auto aspect_ratios = ExpandAspectRatios(aspect_ratios_, flip_);
  const int64_t num_priors =
      static_cast<int64_t>(aspect_ratios.size() * min_sizes_.size() +
                           max_sizes_.size());

  std::vector<float> boxes;
  boxes.reserve(feature_height * feature_width * num_priors * 4);

  // Kept deliberately close to phi's PriorBoxKernel, including the order in
  // which the priors are emitted — the box head's channels are laid out to
  // match it, so a different order silently mislabels every prediction.
  auto emit = [&](float center_x, float center_y, float box_width,
                  float box_height) {
    boxes.push_back((center_x - box_width) / img_width);
    boxes.push_back((center_y - box_height) / img_height);
    boxes.push_back((center_x + box_width) / img_width);
    boxes.push_back((center_y + box_height) / img_height);
  };

  for (int64_t h = 0; h < feature_height; ++h) {
    for (int64_t w = 0; w < feature_width; ++w) {
      float center_x = (w + offset_) * step_width;
      float center_y = (h + offset_) * step_height;
      for (size_t s = 0; s < min_sizes_.size(); ++s) {
        float min_size = min_sizes_[s];
        float box_width, box_height;
        if (min_max_aspect_ratios_order_) {
          box_width = box_height = static_cast<float>(min_size / 2.0);
          emit(center_x, center_y, box_width, box_height);
          if (!max_sizes_.empty()) {
            float max_size = max_sizes_[s];
            box_width = box_height =
                static_cast<float>(std::sqrt(min_size * max_size) / 2.0);
            emit(center_x, center_y, box_width, box_height);
          }
          for (size_t r = 0; r < aspect_ratios.size(); ++r) {
            float ar = aspect_ratios[r];
            if (std::fabs(ar - 1.0f) < 1e-6f) continue;
            box_width = static_cast<float>(min_size * std::sqrt(ar) / 2.0);
            box_height = static_cast<float>(min_size / std::sqrt(ar) / 2.0);
            emit(center_x, center_y, box_width, box_height);
          }
        } else {
          for (size_t r = 0; r < aspect_ratios.size(); ++r) {
            float ar = aspect_ratios[r];
            box_width = static_cast<float>(min_size * std::sqrt(ar) / 2.0);
            box_height = static_cast<float>(min_size / std::sqrt(ar) / 2.0);
            emit(center_x, center_y, box_width, box_height);
          }
          if (!max_sizes_.empty()) {
            float max_size = max_sizes_[s];
            box_width = box_height =
                static_cast<float>(std::sqrt(min_size * max_size) / 2.0);
            emit(center_x, center_y, box_width, box_height);
          }
        }
      }
    }
  }

  if (clip_) {
    for (size_t i = 0; i < boxes.size(); ++i) {
      boxes[i] = std::min(std::max(boxes[i], 0.0f), 1.0f);
    }
  }

  // The variances are the same four numbers repeated for every prior.
  std::vector<float> vars;
  vars.reserve(boxes.size());
  const int64_t num_boxes = feature_height * feature_width * num_priors;
  for (int64_t i = 0; i < num_boxes; ++i) {
    for (size_t j = 0; j < variances_.size(); ++j) vars.push_back(variances_[j]);
  }

  const std::vector<int64_t> shape = {
      feature_height, feature_width, num_priors, 4};
  auto onnx_dtype = GetOnnxDtype(boxes_info[0].dtype);
  helper_->MakeNode("Identity",
                    {helper_->Constant(shape, onnx_dtype, boxes)},
                    {boxes_info[0].name});
  helper_->MakeNode("Identity",
                    {helper_->Constant(shape, onnx_dtype, vars)},
                    {vars_info[0].name});
}

}  // namespace paddle2onnx
