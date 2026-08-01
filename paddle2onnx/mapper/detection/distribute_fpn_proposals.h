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
#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

// pd_op.distribute_fpn_proposals — assigns each RoI to an FPN level according
// to its scale, and returns the permutation needed to undo the regrouping.
//
// Decomposed into arithmetic + NonZero/Gather. PIR only, for the same reason
// as GenerateProposalsMapper.
class DistributeFpnProposalsMapper : public Mapper {
 public:
  DistributeFpnProposalsMapper(const PaddlePirParser& p,
                               OnnxHelper* helper,
                               int64_t op_id,
                               bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    MarkAsExperimentalOp();
    GetAttr("min_level", &min_level_);
    GetAttr("max_level", &max_level_);
    GetAttr("refer_level", &refer_level_);
    GetAttr("refer_scale", &refer_scale_);
    GetAttr("pixel_offset", &pixel_offset_);
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;

 private:
  int64_t min_level_;
  int64_t max_level_;
  int64_t refer_level_;
  int64_t refer_scale_;
  bool pixel_offset_;
};

}  // namespace paddle2onnx
