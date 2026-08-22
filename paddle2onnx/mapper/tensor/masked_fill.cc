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

#include "paddle2onnx/mapper/tensor/masked_fill.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(masked_fill, MaskedFillMapper)

void MaskedFillMapper::Opset9() {
  auto x_info = GetInput("x");
  auto mask_info = GetInput("mask");
  auto value_info = GetInput("value");
  auto out_info = GetOutput("out");

  // `paddle.masked_fill(x, mask, value)` writes `value` where `mask` is true and
  // keeps `x` elsewhere, broadcasting the mask against x. That is exactly
  // `Where(mask, value, x)` — the operand order is the trap: ONNX's Where takes
  // the condition first and the *true* branch second, so passing (mask, x,
  // value) inverts the fill and still produces a well-formed tensor of the right
  // shape.
  auto mask = mask_info[0].name;
  if (mask_info[0].dtype != P2ODataType::BOOL) {
    // The Python wrapper casts for us, but the operator itself does not promise
    // to have been called through it.
    mask = helper_->AutoCast(mask, mask_info[0].dtype, P2ODataType::BOOL);
  }

  // Where needs both branches at one type. `value` is a 0-d tensor of x's dtype
  // whenever it came from a scalar, which is the common case; a tensor `value`
  // of another type is cast rather than refused, because Paddle's own kernel
  // promotes it.
  auto value = value_info[0].name;
  if (value_info[0].dtype != x_info[0].dtype) {
    value = helper_->AutoCast(value, value_info[0].dtype, x_info[0].dtype);
  }

  helper_->MakeNode("Where", {mask, value, x_info[0].name}, {out_info[0].name});
}

}  // namespace paddle2onnx
