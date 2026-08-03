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

#include "paddle2onnx/mapper/detection/matrix_nms.h"

#include <string>
#include <vector>

namespace paddle2onnx {

REGISTER_MAPPER(matrix_nms, MatrixNmsMapper)
REGISTER_PIR_MAPPER(matrix_nms, MatrixNmsMapper)

namespace {
// Stands in for "this pair does not exist"; it must never win a minimum.
constexpr float kNoPredecessor = 1e9f;
}  // namespace

int32_t MatrixNmsMapper::GetMinOpsetVersion(bool verbose) {
  // NonZero (9), Where (9), TopK with a tensor K and sorted/largest (11),
  // GatherND (11).
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

std::string MatrixNmsMapper::Reduce(const std::string& op_type,
                                    const std::string& input,
                                    int64_t axis,
                                    bool keepdims) {
  int32_t axes_as_input = (op_type == "ReduceSum") ? 13 : 18;
  if (helper_->GetOpsetVersion() >= axes_as_input) {
    auto axes = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                  std::vector<int64_t>{axis});
    auto node = helper_->MakeNode(op_type, {input, axes});
    AddAttribute(node, "keepdims", static_cast<int64_t>(keepdims));
    return node->output(0);
  }
  auto node = helper_->MakeNode(op_type, {input});
  AddAttribute(node, "axes", std::vector<int64_t>{axis});
  AddAttribute(node, "keepdims", static_cast<int64_t>(keepdims));
  return node->output(0);
}

void MatrixNmsMapper::Opset11() {
  auto box_info = HasInput("BBoxes") ? GetInput("BBoxes") : GetInput("bboxes");
  auto score_info = HasInput("Scores") ? GetInput("Scores") : GetInput("scores");

  P2OLogger() << "[OP: " << (in_pir_mode ? "pd_op.matrix_nms" : "matrix_nms")
              << "] The exported ONNX model only supports inference with "
                 "input batch_size == 1."
              << std::endl;

  const auto f32 = ONNX_NAMESPACE::TensorProto::FLOAT;
  const auto i64 = ONNX_NAMESPACE::TensorProto::INT64;
  // An explicit variable, not a braced literal: `Constant({}, dtype, v)` is
  // ambiguous against the vector-valued overload.
  const std::vector<int64_t> scalar_shape;
  auto scalar_f = [&](float v) {
    return helper_->Constant(scalar_shape, f32, v);
  };
  auto scalar_i = [&](int64_t v) {
    return helper_->Constant(scalar_shape, i64, v);
  };

  // [1, C, M] -> [C, M] and [1, M, 4] -> [M, 4]. Everything below is written
  // against dynamic C, M and K, so no static shape is needed.
  auto scores = helper_->Squeeze(score_info[0].name, {0});
  auto boxes = helper_->Squeeze(box_info[0].name, {0});
  if (score_info[0].dtype != P2ODataType::FP32) {
    scores = helper_->AutoCast(scores, score_info[0].dtype, P2ODataType::FP32);
  }
  if (box_info[0].dtype != P2ODataType::FP32) {
    boxes = helper_->AutoCast(boxes, box_info[0].dtype, P2ODataType::FP32);
  }

  // ---- the candidate set: every (class, box) pair above score_threshold ----
  auto keep_mask =
      helper_->MakeNode("Greater", {scores, scalar_f(score_threshold_)})
          ->output(0);
  if (background_label_ >= 0) {
    auto num_classes = helper_->Slice(
        helper_->MakeNode("Shape", {scores})->output(0), {0}, {0}, {1});
    auto class_ids =
        helper_->MakeNode("Range", {scalar_i(0),
                                    helper_->Squeeze(num_classes, {0}),
                                    scalar_i(1)})
            ->output(0);
    auto is_bg =
        helper_->MakeNode("Equal", {helper_->Unsqueeze(class_ids, {1}),
                                    scalar_i(background_label_)})
            ->output(0);
    auto not_bg = helper_->MakeNode("Not", {is_bg})->output(0);
    keep_mask = helper_->MakeNode("And", {keep_mask, not_bg})->output(0);
  }

  // NonZero gives [2, K]: row 0 the class index, row 1 the box index.
  auto pairs = helper_->MakeNode("NonZero", {keep_mask})->output(0);
  auto pairs_t = helper_->Transpose(pairs, {1, 0});  // [K, 2]
  auto sel_scores = helper_->MakeNode("GatherND", {scores, pairs_t})->output(0);

  auto gather_col = [&](const std::string& x, int64_t col) {
    auto node = helper_->MakeNode("Gather", {x, scalar_i(col)});
    AddAttribute(node, "axis", static_cast<int64_t>(1));
    return node->output(0);
  };
  auto cand_class = gather_col(pairs_t, 0);  // [K]
  auto cand_box = gather_col(pairs_t, 1);    // [K]

  // ---- one global descending sort; it preserves each class's own order ----
  auto num_cand = helper_->MakeNode("Shape", {sel_scores})->output(0);  // [1]
  auto topk_all = helper_->MakeNode("TopK", {sel_scores, num_cand}, 2);
  AddAttribute(topk_all, "axis", static_cast<int64_t>(0));
  AddAttribute(topk_all, "largest", static_cast<int64_t>(1));
  AddAttribute(topk_all, "sorted", static_cast<int64_t>(1));
  auto s_scores = topk_all->output(0);
  auto s_order = topk_all->output(1);

  auto gather0 = [&](const std::string& x, const std::string& idx) {
    auto node = helper_->MakeNode("Gather", {x, idx});
    AddAttribute(node, "axis", static_cast<int64_t>(0));
    return node->output(0);
  };
  auto s_class = gather0(cand_class, s_order);
  auto s_box = gather0(cand_box, s_order);
  auto s_boxes = gather0(boxes, s_box);  // [K, 4]

  // ---- which (i, j) pairs take part: j < i, same class, both still active ---
  auto k_scalar = helper_->Squeeze(num_cand, {0});
  auto rank =
      helper_->MakeNode("Range", {scalar_i(0), k_scalar, scalar_i(1)})
          ->output(0);
  auto rank_i = helper_->Unsqueeze(rank, {1});
  auto rank_j = helper_->Unsqueeze(rank, {0});
  auto lower = helper_->MakeNode("Greater", {rank_i, rank_j})->output(0);
  auto same_class =
      helper_->MakeNode("Equal", {helper_->Unsqueeze(s_class, {1}),
                                  helper_->Unsqueeze(s_class, {0})})
          ->output(0);
  auto pair = helper_->MakeNode("And", {lower, same_class})->output(0);

  // nms_top_k cuts each class's list *before* the matrix is built, so it is a
  // rank within the class, not a global one.
  std::string active;
  if (nms_top_k_ > -1) {
    auto counted = helper_->AutoCast(pair, P2ODataType::BOOL, P2ODataType::INT64);
    auto rank_in_class = Reduce("ReduceSum", counted, 1, false);
    active = helper_->MakeNode("Less", {rank_in_class, scalar_i(nms_top_k_)})
                 ->output(0);
    auto both = helper_->MakeNode("And", {helper_->Unsqueeze(active, {1}),
                                          helper_->Unsqueeze(active, {0})})
                    ->output(0);
    pair = helper_->MakeNode("And", {pair, both})->output(0);
  }

  // ---- the overlap matrix ----
  auto col = [&](int64_t c) { return helper_->Slice(s_boxes, {1}, {c}, {c + 1}); };
  auto x1 = col(0), y1 = col(1), x2 = col(2), y2 = col(3);
  auto t = [&](const std::string& x) { return helper_->Transpose(x, {1, 0}); };
  auto x1t = t(x1), y1t = t(y1), x2t = t(x2), y2t = t(y2);

  // Paddle returns 0 for disjoint boxes *before* widening by the +1 that
  // normalized=false applies, so the test has to come first rather than being
  // folded into a clamp.
  auto disjoint =
      helper_->MakeNode(
                  "Or",
                  {helper_->MakeNode("Or",
                                     {helper_->MakeNode("Greater", {x1t, x2})
                                          ->output(0),
                                      helper_->MakeNode("Less", {x2t, x1})
                                          ->output(0)})
                       ->output(0),
                   helper_->MakeNode("Or",
                                     {helper_->MakeNode("Greater", {y1t, y2})
                                          ->output(0),
                                      helper_->MakeNode("Less", {y2t, y1})
                                          ->output(0)})
                       ->output(0)})
          ->output(0);

  float eps = normalized_ ? 0.f : 1.f;
  auto widen = [&](const std::string& hi, const std::string& lo) {
    auto d = helper_->MakeNode("Sub", {hi, lo})->output(0);
    if (eps == 0.f) return d;
    return helper_->MakeNode("Add", {d, scalar_f(eps)})->output(0);
  };
  auto inter_w = widen(helper_->MakeNode("Min", {x2, x2t})->output(0),
                       helper_->MakeNode("Max", {x1, x1t})->output(0));
  auto inter_h = widen(helper_->MakeNode("Min", {y2, y2t})->output(0),
                       helper_->MakeNode("Max", {y1, y1t})->output(0));
  auto inter = helper_->MakeNode(
                          "Where",
                          {disjoint, scalar_f(0.f),
                           helper_->MakeNode("Mul", {inter_w, inter_h})
                               ->output(0)})
                   ->output(0);
  auto area = helper_->MakeNode("Mul", {widen(x2, x1), widen(y2, y1)})->output(0);
  auto union_ = helper_->MakeNode(
                            "Sub",
                            {helper_->MakeNode("Add", {area, t(area)})->output(0),
                             inter})
                     ->output(0);
  auto positive = helper_->MakeNode("Greater", {union_, scalar_f(0.f)})->output(0);
  auto safe_union =
      helper_->MakeNode("Where", {positive, union_, scalar_f(1.f)})->output(0);
  auto iou = helper_->MakeNode(
                         "Where",
                         {positive,
                          helper_->MakeNode("Div", {inter, safe_union})->output(0),
                          scalar_f(0.f)})
                 ->output(0);
  iou = helper_->MakeNode("Where", {pair, iou, scalar_f(0.f)})->output(0);

  // ---- the decay ----
  // iou_max is a row maximum, but the decay of box i reads it at the *column*
  // index j -- the strongest overlap the earlier box j already had.
  auto iou_max = Reduce("ReduceMax", iou, 1, false);
  auto iou_max_j = helper_->Unsqueeze(iou_max, {0});
  std::string decay;
  if (use_gaussian_) {
    // Paddle multiplies by gaussian_sigma: it is an inverse variance, not the
    // sigma of the Matrix NMS paper's exp(-iou^2 / sigma).
    auto sq = [&](const std::string& x) {
      return helper_->MakeNode("Mul", {x, x})->output(0);
    };
    auto diff =
        helper_->MakeNode("Sub", {sq(iou_max_j), sq(iou)})->output(0);
    decay = helper_->MakeNode(
                        "Exp",
                        {helper_->MakeNode("Mul", {diff, scalar_f(gaussian_sigma_)})
                             ->output(0)})
                ->output(0);
  } else {
    auto one = scalar_f(1.f);
    auto denom = helper_->MakeNode("Sub", {one, iou_max_j})->output(0);
    auto zero_denom =
        helper_->MakeNode("Equal", {denom, scalar_f(0.f)})->output(0);
    auto safe_denom =
        helper_->MakeNode("Where", {zero_denom, scalar_f(1e-30f), denom})
            ->output(0);
    decay = helper_->MakeNode(
                        "Div",
                        {helper_->MakeNode("Sub", {one, iou})->output(0),
                         safe_denom})
                ->output(0);
  }
  decay = helper_->MakeNode("Where", {pair, decay, scalar_f(kNoPredecessor)})
              ->output(0);
  auto min_decay =
      helper_->MakeNode("Min", {Reduce("ReduceMin", decay, 1, false),
                                scalar_f(1.f)})
          ->output(0);
  auto decayed = helper_->MakeNode("Mul", {s_scores, min_decay})->output(0);

  // ---- keep what survives post_threshold, then the global keep_top_k ----
  auto survive =
      helper_->MakeNode("Greater", {decayed, scalar_f(post_threshold_)})
          ->output(0);
  if (nms_top_k_ > -1) {
    survive = helper_->MakeNode("And", {survive, active})->output(0);
  }
  auto kept = helper_->Squeeze(
      helper_->MakeNode("NonZero", {survive})->output(0), {0});
  auto f_scores = gather0(decayed, kept);
  auto f_class = gather0(s_class, kept);
  auto f_box = gather0(s_box, kept);

  auto num_kept = helper_->MakeNode("Shape", {f_scores})->output(0);
  std::string final_k = num_kept;
  if (keep_top_k_ > -1) {
    final_k = helper_->MakeNode(
                          "Min",
                          {num_kept,
                           helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                             std::vector<int64_t>{keep_top_k_})})
                  ->output(0);
  }
  auto topk_final = helper_->MakeNode("TopK", {f_scores, final_k}, 2);
  AddAttribute(topk_final, "axis", static_cast<int64_t>(0));
  AddAttribute(topk_final, "largest", static_cast<int64_t>(1));
  AddAttribute(topk_final, "sorted", static_cast<int64_t>(1));
  auto out_scores = topk_final->output(0);
  auto out_order = topk_final->output(1);
  auto out_class = gather0(f_class, out_order);
  auto out_box = gather0(f_box, out_order);

  // ---- the three results ----
  auto out_info = HasOutput("Out") ? GetOutput("Out") : GetOutput("out");
  auto class_f =
      helper_->AutoCast(out_class, P2ODataType::INT64, P2ODataType::FP32);
  helper_->Concat({helper_->Unsqueeze(class_f, {1}),
                   helper_->Unsqueeze(out_scores, {1}),
                   gather0(boxes, out_box)},
                  out_info[0].name, 1);

  std::string index_key = HasOutput("Index") ? "Index" : "index";
  if (HasOutput(index_key)) {
    auto index_info = GetOutput(index_key);
    auto idx = helper_->Unsqueeze(out_box, {1});
    helper_->AutoCast(idx, index_info[0].name, P2ODataType::INT64,
                      index_info[0].dtype);
  }
  std::string rois_key = HasOutput("RoisNum") ? "RoisNum" : "roisnum";
  if (HasOutput(rois_key)) {
    auto rois_info = GetOutput(rois_key);
    auto count = helper_->MakeNode("Shape", {out_scores})->output(0);
    helper_->AutoCast(count, rois_info[0].name, P2ODataType::INT64,
                      rois_info[0].dtype);
  }
}

}  // namespace paddle2onnx
