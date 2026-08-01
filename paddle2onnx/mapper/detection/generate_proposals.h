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

// pd_op.generate_proposals — the RPN proposal stage of two-stage detectors
// (Faster / Mask / Cascade R-CNN).
//
// There is no single ONNX operator for this, so it is decomposed into
// TopK / box-decode arithmetic / Clip / NonMaxSuppression. Registered for the
// PIR parser only: the legacy IR spells the op `generate_proposals_v2` and is
// not reachable on Paddle >= 3.0, where PIR export is mandatory.
//
// Batch size is assumed to be 1, which is what PaddleDetection itself enforces
// when exporting an RCNN to ONNX ("Exporting RCNN model to ONNX only support
// batch_size = 1", ppdet/engine/trainer.py).
class GenerateProposalsMapper : public Mapper {
 public:
  GenerateProposalsMapper(const PaddlePirParser& p,
                          OnnxHelper* helper,
                          int64_t op_id,
                          bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    MarkAsExperimentalOp();
    GetAttr("pre_nms_top_n", &pre_nms_top_n_);
    GetAttr("post_nms_top_n", &post_nms_top_n_);
    GetAttr("nms_thresh", &nms_thresh_);
    GetAttr("min_size", &min_size_);
    GetAttr("eta", &eta_);
    GetAttr("pixel_offset", &pixel_offset_);
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;

 private:
  int64_t pre_nms_top_n_;
  int64_t post_nms_top_n_;
  float nms_thresh_;
  float min_size_;
  float eta_;
  bool pixel_offset_;
};

}  // namespace paddle2onnx
