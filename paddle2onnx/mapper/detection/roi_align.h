// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class RoiAlignMapper : public Mapper {
 public:
  RoiAlignMapper(const PaddleParser& p,
                 OnnxHelper* helper,
                 int64_t block_id,
                 int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    MarkAsExperimentalOp();
    GetAttr("pooled_height", &pooled_height_);
    GetAttr("pooled_width", &pooled_width_);
    GetAttr("spatial_scale", &spatial_scale_);
    GetAttr("sampling_ratio", &sampling_ratio_);
    GetAttr("aligned", &aligned_);
  }

  RoiAlignMapper(const PaddlePirParser& p,
                 OnnxHelper* helper,
                 int64_t op_id,
                 bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    MarkAsExperimentalOp();
    GetAttr("pooled_height", &pooled_height_);
    GetAttr("pooled_width", &pooled_width_);
    GetAttr("spatial_scale", &spatial_scale_);
    GetAttr("sampling_ratio", &sampling_ratio_);
    GetAttr("aligned", &aligned_);
  }

  int32_t GetMinOpsetVersion(bool verbose) override {
    // RoiAlign only grew coordinate_transformation_mode at opset 16, and every
    // earlier version behaves as output_half_pixel -- Paddle's aligned=false.
    // An aligned=true op therefore cannot be expressed below 16: the exported
    // graph would load, run, and be offset by half a pixel, which is a silent
    // wrongness rather than a missing feature. Ask for 16 instead.
    if (aligned_) {
      Logger(verbose, 16) << "aligned=true requires coordinate_transformation_"
                             "mode, which RoiAlign only has from opset 16. "
                          << RequireOpset(16) << std::endl;
      return 16;
    }
    Logger(verbose, 10) << RequireOpset(10) << std::endl;
    return 10;
  }
  void Opset10() override;
  // RoiAlign-16 introduced coordinate_transformation_mode and defaults it to
  // half_pixel, which is NOT what RoiAlign-10 did. Paddle's `aligned` flag
  // selects between the two, so from opset 16 on it must be emitted.
  void Opset16() override;

 private:
  void Export(bool emit_coordinate_transformation_mode);

 private:
  int64_t pooled_height_;
  int64_t pooled_width_;
  float spatial_scale_;
  int64_t sampling_ratio_;
  bool aligned_;
};

}  // namespace paddle2onnx
