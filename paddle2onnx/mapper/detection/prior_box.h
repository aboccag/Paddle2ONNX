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

#pragma once
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

// pd_op.prior_box — the SSD anchor generator. It is the only operator standing
// between the SSD family and ONNX.
//
// Every parameter is an attribute and the feature-map and image sizes are
// static for SSD (the architecture requires a fixed input resolution), so the
// whole prior tensor is evaluated at conversion time and emitted as a constant.
// That is exact and costs nothing at inference; a dynamic-shape variant would
// need the grid built with Range/Expand and is rejected explicitly instead of
// being emitted untested.
class PriorBoxMapper : public Mapper {
 public:
  PriorBoxMapper(const PaddlePirParser& p,
                 OnnxHelper* helper,
                 int64_t op_id,
                 bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    GetAttr("min_sizes", &min_sizes_);
    GetAttr("max_sizes", &max_sizes_);
    GetAttr("aspect_ratios", &aspect_ratios_);
    GetAttr("variances", &variances_);
    GetAttr("flip", &flip_);
    GetAttr("clip", &clip_);
    GetAttr("step_w", &step_w_);
    GetAttr("step_h", &step_h_);
    GetAttr("offset", &offset_);
    GetAttr("min_max_aspect_ratios_order", &min_max_aspect_ratios_order_);
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset7() override;

 private:
  std::vector<float> min_sizes_;
  std::vector<float> max_sizes_;
  std::vector<float> aspect_ratios_;
  std::vector<float> variances_;
  bool flip_;
  bool clip_;
  float step_w_;
  float step_h_;
  float offset_;
  bool min_max_aspect_ratios_order_;
};

}  // namespace paddle2onnx
