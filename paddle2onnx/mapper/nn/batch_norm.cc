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

#include "paddle2onnx/mapper/nn/batch_norm.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(batch_norm, BatchNormMapper)
REGISTER_PIR_MAPPER(batch_norm, BatchNormMapper)

int32_t BatchNormMapper::GetMinOpsetVersion(bool verbose) {
  if (trainable_statistics_) {
    Logger(verbose, 14) << RequireOpset(14) << std::endl;
    return 14;
  }
  return 7;
}

namespace {
// Paddle's batch_norm collapses every channels-last layout onto one attribute
// value: F.batch_norm maps NLC, NHWC and NDHWC all to data_format "NHWC"
// (`'NCHW' if data_format[1] == 'C' else 'NHWC'`). So an "NHWC" batch_norm may
// be rank 3 (BatchNorm1D) or rank 5 (BatchNorm3D), and a fixed {0, 3, 1, 2}
// would be a malformed Transpose on both. The permutation moves the last axis
// to position 1 whatever the rank.
inline std::vector<int64_t> ChannelsLastToFirstPerm(size_t rank) {
  std::vector<int64_t> perm = {0, static_cast<int64_t>(rank) - 1};
  for (int64_t i = 1; i < static_cast<int64_t>(rank) - 1; ++i) {
    perm.push_back(i);
  }
  return perm;
}
inline std::vector<int64_t> ChannelsFirstToLastPerm(size_t rank) {
  std::vector<int64_t> perm = {0};
  for (int64_t i = 2; i < static_cast<int64_t>(rank); ++i) {
    perm.push_back(i);
  }
  perm.push_back(1);
  return perm;
}
}  // namespace

void BatchNormMapper::Opset7() {
  auto input_info = GetInput("X");
  auto mean_info = GetInput("Mean");
  auto variance_info = GetInput("Variance");
  auto output_info = GetOutput("Y");

  // ONNX's BatchNormalization reads channels at axis 1, and this mapper never
  // consulted data_format_ -- although the PIR constructor has always parsed
  // it. Under NHWC it therefore handed a channels-last tensor to a channels-
  // first operator, producing a graph that converts and is then refused at
  // load:
  //   Node (BatchNormalization.0) [ShapeInferenceError] Dimension mismatch in
  //   unification between 64 and 112
  // -- 64 channels against a 112 spatial extent. Unlike conv2d and pool2d this
  // one never refused, because it did not know there was anything to refuse.
  const bool nhwc =
      data_format_ == "NHWC" && input_info[0].shape.size() >= 3;
  std::string nhwc_final_output;
  const size_t rank = input_info[0].shape.size();
  if (nhwc) {
    input_info[0].name = helper_->Transpose(
        input_info[0].name, ChannelsLastToFirstPerm(rank));
    nhwc_final_output = output_info[0].name;
    output_info[0].name = MapperHelper::Get()->GenName("batch_norm.nchw");
  }

  std::string scale_name, bias_name;
  int64_t numel = 1;
  for (auto s : mean_info[0].shape) {
    numel *= s;
  }
  if (HasInput("Scale")) {
    scale_name = GetInput("Scale")[0].name;
  } else {
    std::vector<int64_t> values(numel, 1);
    scale_name = helper_->Constant(
        mean_info[0].shape, GetOnnxDtype(mean_info[0].dtype), values);
  }

  if (HasInput("Bias")) {
    bias_name = GetInput("Bias")[0].name;
  } else {
    std::vector<int64_t> values(numel, 0);
    bias_name = helper_->Constant(
        mean_info[0].shape, GetOnnxDtype(mean_info[0].dtype), values);
  }

  auto node = helper_->MakeNode("BatchNormalization",
                                {input_info[0].name,
                                 scale_name,
                                 bias_name,
                                 mean_info[0].name,
                                 variance_info[0].name},
                                {output_info[0].name});
  if (helper_->GetOpsetVersion() < 9) {
    int64_t spatial = 1;
    AddAttribute(node, "spatial", spatial);
  }

  AddAttribute(node, "epsilon", epsilon_);
  AddAttribute(node, "momentum", momentum_);

  if (nhwc) {
    helper_->Transpose(
        output_info[0].name, nhwc_final_output, ChannelsFirstToLastPerm(rank));
  }
}

void BatchNormMapper::Opset14() {
  auto input_info = GetInput("X");
  auto mean_info = GetInput("Mean");
  auto variance_info = GetInput("Variance");
  auto output_info = GetOutput("Y");

  // Same transpose-around as Opset7 above; see the comment there.
  const bool nhwc =
      data_format_ == "NHWC" && input_info[0].shape.size() >= 3;
  std::string nhwc_final_output;
  const size_t rank = input_info[0].shape.size();
  if (nhwc) {
    input_info[0].name = helper_->Transpose(
        input_info[0].name, ChannelsLastToFirstPerm(rank));
    nhwc_final_output = output_info[0].name;
    output_info[0].name = MapperHelper::Get()->GenName("batch_norm.nchw");
  }
  auto mean_out_info = GetOutput("MeanOut");
  auto variance_out_info = GetOutput("VarianceOut");

  std::string scale_name, bias_name;
  int64_t numel = 1;
  for (auto s : mean_info[0].shape) {
    numel *= s;
  }
  if (HasInput("Scale")) {
    scale_name = GetInput("Scale")[0].name;
  } else {
    std::vector<int64_t> values(numel, 1);
    scale_name = helper_->Constant(
        mean_info[0].shape, GetOnnxDtype(mean_info[0].dtype), values);
  }

  if (HasInput("Bias")) {
    bias_name = GetInput("Bias")[0].name;
  } else {
    std::vector<int64_t> values(numel, 0);
    bias_name = helper_->Constant(
        mean_info[0].shape, GetOnnxDtype(mean_info[0].dtype), values);
  }

  std::vector<std::string> output_names;
  output_names.push_back(output_info[0].name);
  if (trainable_statistics_) {
    output_names.push_back(mean_out_info[0].name);
    output_names.push_back(variance_out_info[0].name);
  }
  auto node = helper_->MakeNode("BatchNormalization",
                                {input_info[0].name,
                                 scale_name,
                                 bias_name,
                                 mean_info[0].name,
                                 variance_info[0].name},
                                output_names);
  if (helper_->GetOpsetVersion() < 9) {
    int64_t spatial = 1;
    AddAttribute(node, "spatial", spatial);
  }

  AddAttribute(node, "epsilon", epsilon_);
  AddAttribute(node, "momentum", momentum_);
  AddAttribute(
      node, "training_mode", static_cast<int64_t>(trainable_statistics_));

  if (nhwc) {
    helper_->Transpose(
        output_info[0].name, nhwc_final_output, ChannelsFirstToLastPerm(rank));
  }
}

}  // namespace paddle2onnx
