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
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

// Matrix NMS decays the score of every box by how much it overlaps the
// higher-scoring boxes of its own class, instead of removing it outright. ONNX
// has no equivalent operator and no loop over classes, so the decomposition
// flattens every (class, box) pair above the score threshold into a single
// score-sorted list and computes one overlap matrix over all of them, with the
// cross-class entries masked out. See scripts/matrix_nms_ref.py in the PoC for
// the numpy model this mirrors, which is checked against Paddle directly.
class MatrixNmsMapper : public Mapper {
 public:
  MatrixNmsMapper(const PaddleParser& p,
                  OnnxHelper* helper,
                  int64_t block_id,
                  int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttrs();
  }

  MatrixNmsMapper(const PaddlePirParser& p,
                  OnnxHelper* helper,
                  int64_t op_id,
                  bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    in_pir_mode = true;
    GetAttrs();
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;

 private:
  void GetAttrs() {
    GetAttr("score_threshold", &score_threshold_);
    GetAttr("post_threshold", &post_threshold_);
    GetAttr("nms_top_k", &nms_top_k_);
    GetAttr("keep_top_k", &keep_top_k_);
    GetAttr("background_label", &background_label_);
    GetAttr("normalized", &normalized_);
    GetAttr("use_gaussian", &use_gaussian_);
    GetAttr("gaussian_sigma", &gaussian_sigma_);
  }

  // ReduceMax/ReduceMin/ReduceSum took their axes as an attribute up to opset
  // 17 (13 for ReduceSum) and as an input after that.
  std::string Reduce(const std::string& op_type,
                     const std::string& input,
                     int64_t axis,
                     bool keepdims);

  float score_threshold_ = 0.f;
  float post_threshold_ = 0.f;
  float gaussian_sigma_ = 2.f;
  int64_t nms_top_k_ = -1;
  int64_t keep_top_k_ = -1;
  int64_t background_label_ = -1;
  bool normalized_ = true;
  bool use_gaussian_ = false;
};

}  // namespace paddle2onnx
