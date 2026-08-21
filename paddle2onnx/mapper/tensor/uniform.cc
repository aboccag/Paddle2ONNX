// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/mapper/tensor/uniform.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(uniform, UniformMapper)

int32_t UniformMapper::GetMinOpsetVersion(bool verbose) {
  // A shape that is only known at run time has to be materialised with
  // ConstantOfShape, which arrived in opset 9. A constant one is still an
  // attribute on RandomUniform, and still works at 7.
  if (!IsConstantInput("shape")) {
    Logger(verbose, 9) << "While shape is not a constant tensor, "
                       << RequireOpset(9) << std::endl;
    return 9;
  }
  return 7;
}

void UniformMapper::Opset7() {
  auto output_info = GetOutput("out");
  auto shape_info = GetInput("shape");
  auto min_info = GetInput("min");
  auto max_info = GetInput("max");

  if (min_info[0].Rank() != 0 || max_info[0].Rank() != 0) {
    Error() << "min/max must be scalar tensors for op uniform " << std::endl;
  }
  std::vector<float> min_val{0.0f}, max_val{1.0f};
  helper_->TryGetTensorValue<float>(min_info[0].name, &min_val);
  helper_->TryGetTensorValue<float>(max_info[0].name, &max_val);

  auto onnx_dtype = GetOnnxDtype(dtype_);

  // RandomUniform carries its shape as an attribute, so it can only express a
  // shape known at conversion time. `paddle.rand((B * S, D, R))` — SegNeXt's
  // Hamburger head builds its NMF bases that way on every forward — has the
  // batch in it, and under a dynamic export the batch is a run-time value.
  //
  // Both lookups then fail. The previous version of this mapper ignored that
  // and emitted `shape=[]` anyway: a degenerate RandomUniform, and a graph
  // onnxruntime refuses to load at all because the F.normalize that follows
  // reduces over axis 1 of what is no longer a rank-3 tensor
  // ("axis must be in [-rank, rank-1]. Input rank was 1"). Four segnext
  // configs converted to exactly that, and the conversion exited 0.
  std::vector<int64_t> shape_values;
  bool shape_is_known =
      helper_->TryGetTensorValue<int64_t>(shape_info[0].name, &shape_values) ||
      TryGetInputValue("shape", &shape_values);

  if (shape_is_known && !shape_values.empty()) {
    auto random_node =
        helper_->MakeNode("RandomUniform", {}, {output_info[0].name});
    AddAttribute(random_node, "shape", shape_values);
    AddAttribute(random_node, "low", min_val[0]);
    AddAttribute(random_node, "high", max_val[0]);
    AddAttribute(random_node, "dtype", static_cast<int64_t>(onnx_dtype));
    if (seed_ != 0) {
      AddAttribute(random_node, "seed", static_cast<float>(seed_));
    }
    return;
  }

  // Otherwise take the shape as a tensor: ConstantOfShape materialises a
  // buffer of that exact run-time shape and RandomUniformLike fills it. This
  // is what the sibling `gaussian` mapper already does for the same case, and
  // it also covers the scalar `paddle.rand(())` — an empty shape tensor gives
  // ConstantOfShape a rank-0 output, and the fill preserves it.
  auto shape_tensor = helper_->AutoCast(
      shape_info[0].name, shape_info[0].dtype, P2ODataType::INT64);
  auto like = helper_->ConstOfShape(shape_tensor, onnx_dtype, 0.0f);
  auto random_node =
      helper_->MakeNode("RandomUniformLike", {like}, {output_info[0].name});
  AddAttribute(random_node, "low", min_val[0]);
  AddAttribute(random_node, "high", max_val[0]);
  AddAttribute(random_node, "dtype", static_cast<int64_t>(onnx_dtype));
  if (seed_ != 0) {
    AddAttribute(random_node, "seed", static_cast<float>(seed_));
  }
}
}  // namespace paddle2onnx
