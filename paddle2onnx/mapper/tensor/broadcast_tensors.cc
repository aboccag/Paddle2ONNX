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

#include "paddle2onnx/mapper/tensor/broadcast_tensors.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(broadcast_tensors, BroadcastTensorsMapper)

int32_t BroadcastTensorsMapper::GetMinOpsetVersion(bool verbose) {
  // Expand and ConstantOfShape both arrived in opset 9.
  Logger(verbose, 9) << RequireOpset(9) << std::endl;
  return 9;
}

void BroadcastTensorsMapper::Opset9() {
  auto inputs = GetInput(0);    // vector of tensors
  auto outputs = GetOutput(0);  // vector of tensors, same arity

  // The common shape is derived by adding together zero tensors of each input's
  // shape: ONNX broadcasting gives the result the broadcast shape, and this
  // works for any ranks without having to reason about them here.
  std::string acc;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto shape = helper_->MakeNode("Shape", {inputs[i].name})->output(0);
    auto zeros = helper_->ConstOfShape(
        shape, ONNX_NAMESPACE::TensorProto::FLOAT, static_cast<float>(0.0));
    acc = acc.empty() ? zeros
                      : helper_->MakeNode("Add", {acc, zeros})->output(0);
  }
  auto target_shape = helper_->MakeNode("Shape", {acc})->output(0);

  for (size_t i = 0; i < inputs.size() && i < outputs.size(); ++i) {
    helper_->MakeNode(
        "Expand", {inputs[i].name, target_shape}, {outputs[i].name});
  }
}

}  // namespace paddle2onnx
