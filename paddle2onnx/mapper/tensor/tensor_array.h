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

#pragma once
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class CreateArrayMapper : public Mapper {
 public:
  CreateArrayMapper(const PaddlePirParser& p,
                    OnnxHelper* helper,
                    int64_t op_id,
                    bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    GetAttr("dtype", &dtype_);
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;

 private:
  int64_t dtype_;
};

class ArrayLengthMapper : public Mapper {
 public:
  ArrayLengthMapper(const PaddlePirParser& p,
                    OnnxHelper* helper,
                    int64_t op_id,
                    bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {}

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;
};

class ArrayWriteMapper : public Mapper {
 public:
  ArrayWriteMapper(const PaddlePirParser& p,
                   OnnxHelper* helper,
                   int64_t op_id,
                   bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {}

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;
};

class ArrayReadMapper : public Mapper {
 public:
  ArrayReadMapper(const PaddlePirParser& p,
                  OnnxHelper* helper,
                  int64_t op_id,
                  bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {}

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;
};

// pd_op.slice_array_dense — reads one element out of a TensorArray as a dense
// tensor. Paddle emits it wherever dy2static turns a Python list into a
// TensorArray and the result is indexed, which is what the RCNN family does
// when exported without `export_onnx=True`.
class SliceArrayDenseMapper : public Mapper {
 public:
  SliceArrayDenseMapper(const PaddlePirParser& p,
                        OnnxHelper* helper,
                        int64_t op_id,
                        bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {}

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;
};

}  // namespace paddle2onnx
