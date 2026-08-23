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

#include "paddle2onnx/mapper/nn/conv2d.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(conv2d, Conv2dMapper)
REGISTER_MAPPER(depthwise_conv2d, Conv2dMapper)
REGISTER_PIR_MAPPER(conv2d, Conv2dMapper)
REGISTER_PIR_MAPPER(depthwise_conv2d, Conv2dMapper)

int32_t Conv2dMapper::GetMinOpsetVersion(bool verbose) {
  // NHWC is handled by transposing around the operator -- see Opset7.
  if (padding_algorithm_ == "EXPLICIT") {
    if (paddings_.size() != 2 && paddings_.size() != 4) {
      Error() << "While padding_algorithm is EXPLICIT, size of paddings should "
                 "be 2 or 4."
              << std::endl;
      return -1;
    }
  }
  if (dilations_[0] != 1 || dilations_[1] != 1) {
    if (padding_algorithm_ == "SAME") {
      Error() << "While dilations != 1, cannot support padding = 'SAME'."
              << std::endl;
      return -1;
    }
  }
  return 7;
}

void Conv2dMapper::Opset7() {
  auto input_info = GetInput("Input");
  auto kernel_info = GetInput("Filter");
  auto output_info = GetOutput("Output");

  // ONNX's Conv is NCHW only, so an NHWC activation is transposed in and back
  // out again -- the same shape Conv3dTransposeMapper uses for NDHWC. The
  // filter is NOT transposed: Paddle stores it as [out, in, kh, kw] whatever
  // data_format says, which is already the layout ONNX wants (and is why
  // kernel_shape below still reads shape[2] and shape[3]).
  std::string input = input_info[0].name;
  const bool nhwc = data_format_ == "NHWC";
  if (nhwc) {
    input = helper_->Transpose(input, {0, 3, 1, 2});  // NHWC -> NCHW
  }

  // On the NCHW path the node still writes the graph output directly, so that
  // path emits exactly what it emitted before this branch existed.
  auto node =
      nhwc ? helper_->MakeNode("Conv", {input, kernel_info[0].name})
           : helper_->MakeNode("Conv",
                               {input, kernel_info[0].name},
                               {output_info[0].name});
  AddAttribute(node, "dilations", dilations_);
  std::vector<int64_t> kernel_shape = {kernel_info[0].shape[2],
                                       kernel_info[0].shape[3]};
  AddAttribute(node, "kernel_shape", kernel_shape);
  AddAttribute(node, "strides", strides_);
  AddAttribute(node, "group", groups_);
  if (padding_algorithm_ == "SAME") {
    std::string auto_pad = "SAME_UPPER";
    AddAttribute(node, "auto_pad", auto_pad);
  } else if (padding_algorithm_ == "VALID") {
    std::string auto_pad = "VALID";
    AddAttribute(node, "auto_pad", auto_pad);
  } else {
    std::vector<int64_t> paddings;
    if (paddings_.size() == 2) {
      paddings.insert(paddings.begin(), paddings_.begin(), paddings_.end());
      paddings.insert(paddings.begin(), paddings_.begin(), paddings_.end());
    } else {
      paddings.assign(paddings_.begin(), paddings_.end());
      paddings[1] = paddings_[2];
      paddings[2] = paddings_[1];
    }
    AddAttribute(node, "pads", paddings);
  }

  if (nhwc) {
    helper_->Transpose(
        node->output(0), output_info[0].name, {0, 2, 3, 1});  // NCHW -> NHWC
  }
}

}  // namespace paddle2onnx
