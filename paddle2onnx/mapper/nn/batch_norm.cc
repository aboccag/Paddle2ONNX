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
  const bool nhwc = data_format_ == "NHWC";
  std::string nhwc_final_output;
  if (nhwc) {
    input_info[0].name = helper_->Transpose(input_info[0].name, {0, 3, 1, 2});
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
        output_info[0].name, nhwc_final_output, {0, 2, 3, 1});
  }
}

void BatchNormMapper::Opset14() {
  auto input_info = GetInput("X");
  auto mean_info = GetInput("Mean");
  auto variance_info = GetInput("Variance");
  auto output_info = GetOutput("Y");

  // Same transpose-around as Opset7 above; see the comment there.
  const bool nhwc = data_format_ == "NHWC";
  std::string nhwc_final_output;
  if (nhwc) {
    input_info[0].name = helper_->Transpose(input_info[0].name, {0, 3, 1, 2});
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
        output_info[0].name, nhwc_final_output, {0, 2, 3, 1});
  }
}

}  // namespace paddle2onnx
