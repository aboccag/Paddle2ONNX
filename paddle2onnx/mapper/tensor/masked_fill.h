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
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class MaskedFillMapper : public Mapper {
 public:
  MaskedFillMapper(const PaddlePirParser& p,
                   OnnxHelper* helper,
                   int64_t op_id,
                   bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    in_pir_mode = true;
  }

  int32_t GetMinOpsetVersion(bool verbose) override {
    // Where is opset 9; its numpy-style broadcasting, which a scalar `value`
    // relies on, arrived with it.
    Logger(verbose, 9) << RequireOpset(9) << std::endl;
    return 9;
  }

  void Opset9() override;
};

}  // namespace paddle2onnx
