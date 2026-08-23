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

#include "paddle2onnx/mapper/nn/pool2d.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(pool2d, Pool2dMapper)
REGISTER_MAPPER(max_pool2d_with_index, Pool2dMapper)

REGISTER_PIR_MAPPER(pool2d, Pool2dMapper)
REGISTER_PIR_MAPPER(max_pool2d_with_index, Pool2dMapper)

namespace {
// Every shape index in this file is written for NCHW -- `shape[2]` and
// `shape[3]` are read as H and W in both GetMinOpsetVersion and Opset7, and the
// pooling attributes (ksize, strides, paddings) are in H,W order whatever
// data_format says. Rather than teach four emission paths about the layout, an
// NHWC shape is permuted into NCHW here and the activation itself is transposed
// once in Opset7. Nothing else in the file has to know.
inline std::vector<int64_t> NhwcShapeToNchw(const std::vector<int64_t>& shape) {
  if (shape.size() != 4) return shape;
  return {shape[0], shape[3], shape[1], shape[2]};
}
}  // namespace

void Pool2dMapper::ExactAdaptiveAvgPool(
    const std::vector<TensorInfo>& input_info,
    const std::vector<TensorInfo>& output_info) {
  // Paddle computes adaptive window i on each axis as
  // [floor(i*in/out), ceil((i+1)*in/out)). When out does not evenly divide
  // in, those windows have non-uniform strides and overlap, and no
  // AveragePool(kernel, stride) samples them — the kernel/stride emission
  // below averages different pixels and is silently wrong (found by numeric
  // parity on PaddleSeg's PSP/SPPM heads: worst delta 2.8e-01 at 30->4
  // against 3.3e-06 for the same weights at a divisible size).
  //
  // The pooling is separable, so it is emitted exactly instead: each output
  // row/column is a fixed average of input rows/columns, i.e. a matrix
  // product against a constant averaging matrix on each axis.
  //   Y = A_h · X · A_w,  A_h [out_h, in_h], A_w [in_w, out_w]
  // MatMul broadcasts the 2-D constants over [N, C, ·, ·], so two nodes and
  // two small initializers cover any batch and channel count.
  int64_t input_h = input_info[0].shape[2];
  int64_t input_w = input_info[0].shape[3];
  int64_t output_h = output_info[0].shape[2];
  int64_t output_w = output_info[0].shape[3];

  auto window = [](int64_t i, int64_t in, int64_t out, int64_t* start,
                   int64_t* end) {
    *start =
        static_cast<int64_t>(std::floor(static_cast<double>(i * in) / out));
    *end = static_cast<int64_t>(
        std::ceil(static_cast<double>((i + 1) * in) / out));
  };

  std::vector<float> row_weights(output_h * input_h, 0.0f);
  for (int64_t i = 0; i < output_h; ++i) {
    int64_t start = 0;
    int64_t end = 0;
    window(i, input_h, output_h, &start, &end);
    for (int64_t k = start; k < end; ++k) {
      row_weights[i * input_h + k] = 1.0f / static_cast<float>(end - start);
    }
  }
  std::vector<float> col_weights(input_w * output_w, 0.0f);
  for (int64_t j = 0; j < output_w; ++j) {
    int64_t start = 0;
    int64_t end = 0;
    window(j, input_w, output_w, &start, &end);
    for (int64_t k = start; k < end; ++k) {
      col_weights[k * output_w + j] = 1.0f / static_cast<float>(end - start);
    }
  }

  std::vector<int64_t> row_shape = {output_h, input_h};
  std::vector<int64_t> col_shape = {input_w, output_w};
  std::string rows = helper_->Constant(
      row_shape, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, row_weights);
  std::string cols = helper_->Constant(
      col_shape, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, col_weights);

  std::string input = input_info[0].name;
  bool needs_cast = kNoNeedCastTypesOpSet7.find(input_info[0].dtype) ==
                    kNoNeedCastTypesOpSet7.end();
  if (needs_cast) {
    input = helper_->AutoCast(input, input_info[0].dtype, P2ODataType::FP32);
  }
  auto pooled_rows = helper_->MakeNode("MatMul", {rows, input});
  if (needs_cast) {
    auto pooled = helper_->MakeNode("MatMul", {pooled_rows->output(0), cols});
    helper_->AutoCast(pooled->output(0),
                      output_info[0].name,
                      P2ODataType::FP32,
                      output_info[0].dtype);
  } else {
    helper_->MakeNode(
        "MatMul", {pooled_rows->output(0), cols}, {output_info[0].name});
  }
}

void Pool2dMapper::AdaptivePool(const std::vector<TensorInfo>& input_info,
                                const std::vector<TensorInfo>& output_info) {
  int64_t input_h = input_info[0].shape[2];
  int64_t input_w = input_info[0].shape[3];
  int64_t output_h = output_info[0].shape[2];
  int64_t output_w = output_info[0].shape[3];
  int64_t stride_h = std::floor(input_h / output_h);
  int64_t stride_w = std::floor(input_w / output_w);
  int64_t kernel_h = input_h - (output_h - 1) * stride_h;
  int64_t kernel_w = input_w - (output_w - 1) * stride_w;
  std::string onnx_pool_type;
  if (convert_pir_op_name(OpType()) == "max_pool2d_with_index") {
    onnx_pool_type = "MaxPool";
  } else {
    auto iter = op_mapper_.find(pooling_type_);
    Assert(iter != op_mapper_.end(), "Pooling not found");
    onnx_pool_type = iter->second[0];
  }
  std::shared_ptr<ONNX_NAMESPACE::NodeProto> node(nullptr);
  if (kNoNeedCastTypesOpSet7.find(input_info[0].dtype) !=
      kNoNeedCastTypesOpSet7.end()) {
    node = helper_->MakeNode(
        onnx_pool_type, {input_info[0].name}, {output_info[0].name});
  } else {
    auto input = helper_->AutoCast(
        input_info[0].name, input_info[0].dtype, P2ODataType::FP32);
    node = helper_->MakeNode(onnx_pool_type, {input});
    helper_->AutoCast(node->output(0),
                      output_info[0].name,
                      P2ODataType::FP32,
                      output_info[0].dtype);
  }

  std::vector<int64_t> kernel_size = {kernel_h, kernel_w};
  AddAttribute(node, "kernel_shape", kernel_size);
  std::vector<int64_t> strides = {stride_h, stride_w};
  AddAttribute(node, "strides", strides);
  // AddAttribute(node, "kernel_shape", k_size_);
  // AddAttribute(node, "strides", strides_);

  if (helper_->GetOpsetVersion() > 10) {
    AddAttribute(node, "ceil_mode", static_cast<int64_t>(ceil_mode_));
  }

  std::string auto_pad = "NOTSET";
  if (padding_algorithm_ == "SAME") {
    auto_pad = "SAME_UPPER";
  } else if (padding_algorithm_ == "VALID") {
    auto_pad = "VALID";
  }
  AddAttribute(node, "auto_pad", auto_pad);
  if (pooling_type_ == "avg") {
    AddAttribute(node, "count_include_pad", static_cast<int64_t>(exclusive_));
  }
}

void Pool2dMapper::NoAdaptivePool(const std::vector<TensorInfo>& input_info,
                                  const std::vector<TensorInfo>& output_info) {
  std::vector<int64_t> input_shape = input_info[0].shape;
  if (pads_.size() == 2) {
    pads_.push_back(pads_[0]);
    pads_.push_back(pads_[1]);
  } else if (pads_.size() == 4) {
    std::vector<int64_t> index = {0, 2, 1, 3};
    std::vector<int64_t> copy = pads_;
    for (auto i = 0; i < index.size(); ++i) {
      pads_[i] = copy[index[i]];
    }
  }
  if (input_shape[2] > 0 && input_shape[2] + pads_[0] + pads_[2] < k_size_[0]) {
    k_size_[0] = input_shape[2] + pads_[0] + pads_[2];
  }
  if (input_shape[3] > 0 && input_shape[3] + pads_[1] + pads_[3] < k_size_[1]) {
    k_size_[1] = input_shape[3] + pads_[1] + pads_[3];
  }

  int64_t max_ksize = *std::max_element(std::begin(k_size_), std::end(k_size_));
  int64_t max_pads = *std::max_element(std::begin(pads_), std::end(pads_));
  std::string input_x = input_info[0].name;
  if (kNoNeedCastTypesOpSet7.find(input_info[0].dtype) ==
      kNoNeedCastTypesOpSet7.end()) {
    input_x = helper_->AutoCast(
        input_info[0].name, input_info[0].dtype, P2ODataType::FP32);
  }
  if (max_ksize <= max_pads) {
    std::vector<int64_t> onnx_paddings = {
        0, 0, pads_[0], pads_[1], 0, 0, pads_[2], pads_[3]};
    std::vector<std::string> inputs_names = {input_x};
    if (helper_->GetOpsetVersion() >= 11) {
      std::string paddings_node =
          helper_->Constant(GetOnnxDtype(P2ODataType::INT64), onnx_paddings);
      inputs_names.push_back(paddings_node);
      std::vector<float> val = {0.0};
      std::string val_node =
          helper_->Constant(GetOnnxDtype(P2ODataType::FP32), val);
      inputs_names.push_back(val_node);
    }
    auto node = helper_->MakeNode("Pad", inputs_names);
    std::string mode = "constant";
    AddAttribute(node, "mode", mode);
    if (helper_->GetOpsetVersion() < 11) {
      AddAttribute(node, "pads", onnx_paddings);
      float val = 0.0;
      AddAttribute(node, "value", val);
    }
    input_x = node->output(0);
    pads_.clear();
    pads_.resize(4, 0);
  }
  std::string onnx_pool_type;
  if (convert_pir_op_name(OpType()) == "max_pool2d_with_index") {
    onnx_pool_type = "MaxPool";
  } else {
    auto iter = op_mapper_.find(pooling_type_);
    Assert(iter != op_mapper_.end(), "Pooling not found");
    onnx_pool_type = iter->second[0];
  }
  std::shared_ptr<ONNX_NAMESPACE::NodeProto> node(nullptr);
  if (kNoNeedCastTypesOpSet7.find(input_info[0].dtype) !=
      kNoNeedCastTypesOpSet7.end()) {
    node = helper_->MakeNode(onnx_pool_type, {input_x}, {output_info[0].name});
  } else {
    node = helper_->MakeNode(onnx_pool_type, {input_x});
    helper_->AutoCast(node->output(0),
                      output_info[0].name,
                      P2ODataType::FP32,
                      output_info[0].dtype);
  }

  AddAttribute(node, "kernel_shape", k_size_);
  AddAttribute(node, "strides", strides_);
  std::string auto_pad = "NOTSET";
  if (padding_algorithm_ == "SAME") {
    auto_pad = "SAME_UPPER";
    AddAttribute(node, "auto_pad", auto_pad);
  } else if (padding_algorithm_ == "VALID") {
    auto_pad = "VALID";
    AddAttribute(node, "auto_pad", auto_pad);
  } else {
    AddAttribute(node, "pads", pads_);
  }
  // TODO(qinzhongyu): Need double check
  // if (OpType() != "max_pool2d_with_index" && helper_->GetOpsetVersion() >=
  // 10) {
  //   AddAttribute(node, "ceil_mode", static_cast<int64_t>(ceil_mode_));
  // }
  // if (OpType() != "max_pool2d_with_index" && pooling_type_ == "avg") {
  //   AddAttribute(node, "count_include_pad",
  //   static_cast<int64_t>(exclusive_));
  // }
  if (helper_->GetOpsetVersion() >= 10) {
    AddAttribute(node, "ceil_mode", static_cast<int64_t>(ceil_mode_));
  }
  if (pooling_type_ == "avg") {
    AddAttribute(node, "count_include_pad", static_cast<int64_t>(exclusive_));
  }
}

int32_t Pool2dMapper::GetMinOpsetVersion(bool verbose) {
  // NHWC is handled by transposing around the whole operator -- see Opset7.
  // The shapes are permuted here too: the adaptive checks below divide
  // shape[2]/shape[3] as H and W, and under NHWC those are W and C.
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (data_format_ == "NHWC") {
    input_info[0].shape = NhwcShapeToNchw(input_info[0].shape);
    output_info[0].shape = NhwcShapeToNchw(output_info[0].shape);
  }
  if (in_pir_mode) {
    if (convert_pir_op_name(OpType()) != "max_pool2d_with_index") {
      // TODO(qinzhongyu): For PIR, kernel size is in inputs
      auto ksize = GetInput("ksize")[0];
      Assert(IsConstantInput("ksize"), "ksize's type is not constant.");
      // for (auto i = 0; i < ksize.shape.size(); ++ i) {
      //   k_size_.push_back(ksize.shape[i]);
      // }
      TryGetInputValue("ksize", &k_size_);
    } else {
      GetAttr("kernel_size", &k_size_);
    }
  } else {
    if (IsAttrVar("ksize")) {
      Error() << "While Attribute(ksize)'s type is Tensor, it's not "
                 "supported."
              << std::endl;
      return -1;
    } else {
      GetAttr("ksize", &k_size_);
    }
  }

  if (global_pooling_ || (k_size_[0] == 1 && k_size_[1] == 1)) {
    if (ceil_mode_) {
      Logger(verbose, 10) << "While ceil_model is True, " << RequireOpset(10)
                          << std::endl;
      return 10;
    }
    return 7;
  }

  if (adaptive_) {
    for (auto one_input : input_info) {
      for (auto i = 2; i < one_input.shape.size(); ++i) {
        if (one_input.shape[i] == -1) {
          Error() << "Adaptive only support static input shape." << std::endl;
          return -1;
        }
      }
    }
    int64_t input_h = input_info[0].shape[2];
    int64_t input_w = input_info[0].shape[3];
    int64_t output_h = output_info[0].shape[2];
    int64_t output_w = output_info[0].shape[3];
    if (output_h == -1 || output_w == -1) {
      Error() << "Cannot convert adaptive pool with input_size: " << input_h
              << " " << input_w << " output_size: " << output_h << " "
              << output_w << std::endl;
      return -1;
    }
    // Divisibility is the only condition under which kernel = stride = in/out
    // reproduces Paddle's [floor(i*in/out), ceil((i+1)*in/out)) windows. The
    // predecessor of this check compared window *sizes* computed with integer
    // division, which both truncated the ratio to a constant (so every window
    // looked identical) and said nothing about strides — 30 -> 4 passed it
    // and converted to a pool that averages different pixels than Paddle's.
    // Non-divisible average pooling is emitted exactly instead (see
    // ExactAdaptiveAvgPool); max pooling is not linear and has no such form,
    // so it is refused rather than approximated.
    bool divisible = (input_h % output_h == 0) && (input_w % output_w == 0);
    bool is_average = convert_pir_op_name(OpType()) != "max_pool2d_with_index" &&
                      pooling_type_ == "avg";
    if (!divisible && !is_average) {
      Error() << "Adaptive max pool whose output does not evenly divide its "
                 "input cannot be expressed as an ONNX MaxPool. input_size: "
              << input_h << " " << input_w << " output_size: " << output_h
              << " " << output_w << std::endl;
      return -1;
    }
  }
  if (convert_pir_op_name(OpType()) == "max_pool2d_with_index") {
    return 9;
  }
  auto iter = op_mapper_.find(pooling_type_);
  if (op_mapper_.end() == iter) {
    Error() << "Cannot find " << pooling_type_ << " in pool op_mapper."
            << std::endl;
    return -1;
  }

  if (ceil_mode_) {
    Logger(verbose, 10) << "While ceil_model is True, " << RequireOpset(10)
                        << std::endl;
    return 10;
  }
  return 7;
}

void Pool2dMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  // NHWC: transpose the activation into NCHW, let the four emission paths below
  // run exactly as they do for NCHW, then transpose the result back. The
  // pooling itself is layout-agnostic once the tensor is in the layout ONNX's
  // MaxPool/AveragePool require, which is NCHW only.
  const bool nhwc = data_format_ == "NHWC";
  std::string nhwc_final_output;
  if (nhwc) {
    input_info[0].name = helper_->Transpose(input_info[0].name, {0, 3, 1, 2});
    input_info[0].shape = NhwcShapeToNchw(input_info[0].shape);
    // The paths below write straight to output_info[0].name, so they are given
    // a temporary and the real graph output is produced by the transpose.
    nhwc_final_output = output_info[0].name;
    output_info[0].name = MapperHelper::Get()->GenName("pool2d.nchw");
    output_info[0].shape = NhwcShapeToNchw(output_info[0].shape);
  }

  if (in_pir_mode) {
    /**
    // TODO: For PIR, kernel size is in inputs
    auto ksize = GetInput("ksize")[0];
    for (auto i = 0; i < ksize.shape.size(); ++ i) {
      k_size_.push_back(ksize.shape[i]);
    }
    */
    // k_size_ = GetInputAttrVar("ksize", "value");
    if (convert_pir_op_name(OpType()) != "max_pool2d_with_index")
      TryGetInputValue("ksize", &k_size_);
    else
      GetAttr("kernel_size", &k_size_);
  } else {
    GetAttr("ksize", &k_size_);
  }

  bool is_1x1_kernel = true;
  for (auto i : k_size_) {
    if (i != 1) {
      is_1x1_kernel = false;
    }
  }

  if (global_pooling_ || (adaptive_ && is_1x1_kernel)) {
    std::string onnx_pool_type;
    if (convert_pir_op_name(OpType()) == "max_pool2d_with_index") {
      onnx_pool_type = "GlobalMaxPool";
    } else {
      auto iter = op_mapper_.find(pooling_type_);
      onnx_pool_type = iter->second[1];
    }
    if (kNoNeedCastTypesOpSet7.find(input_info[0].dtype) !=
        kNoNeedCastTypesOpSet7.end()) {
      auto output = helper_->MakeNode(
          onnx_pool_type, {input_info[0].name}, {output_info[0].name});
    } else {
      auto input = helper_->AutoCast(
          input_info[0].name, input_info[0].dtype, P2ODataType::FP32);
      auto output = helper_->MakeNode(onnx_pool_type, {input})->output(0);
      helper_->AutoCast(
          output, output_info[0].name, P2ODataType::FP32, output_info[0].dtype);
    }
  } else if (adaptive_) {
    int64_t input_h = input_info[0].shape[2];
    int64_t input_w = input_info[0].shape[3];
    int64_t output_h = output_info[0].shape[2];
    int64_t output_w = output_info[0].shape[3];
    if (input_h % output_h == 0 && input_w % output_w == 0) {
      AdaptivePool(input_info, output_info);
    } else {
      // GetMinOpsetVersion only lets average pooling through here.
      ExactAdaptiveAvgPool(input_info, output_info);
    }
  } else {
    NoAdaptivePool(input_info, output_info);
  }

  if (nhwc) {
    helper_->Transpose(
        output_info[0].name, nhwc_final_output, {0, 2, 3, 1});  // NCHW -> NHWC
  }
}

}  // namespace paddle2onnx
