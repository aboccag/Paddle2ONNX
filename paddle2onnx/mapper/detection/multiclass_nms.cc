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

#include "paddle2onnx/mapper/detection/multiclass_nms.h"

namespace paddle2onnx {

REGISTER_MAPPER(multiclass_nms3, NMSMapper);
REGISTER_PIR_MAPPER(multiclass_nms3, NMSMapper);

int32_t NMSMapper::GetMinOpsetVersion(bool verbose) {
  auto boxes_info = GetInput("BBoxes");
  auto score_info = GetInput("Scores");
  if (score_info[0].Rank() == 2) {
    // LoD form, emitted by the RCNN family: BBoxes [M, C, 4], Scores [M, C].
    // Handled by ExportForLodInput(), which needs NonZero (opset 9) and
    // NonMaxSuppression (opset 10).
    if (boxes_info[0].Rank() != 3) {
      Error() << "For LoD input, boxes are expected to be a 3-D tensor of "
                 "shape [M, C, 4], but the rank is "
              << boxes_info[0].Rank() << "." << std::endl;
      return -1;
    }
    if (score_info[0].shape[1] <= 0) {
      Error() << "The number of classes (2nd dimension of scores) must be "
                 "static, but it is "
              << score_info[0].shape[1] << "." << std::endl;
      return -1;
    }
    Logger(verbose, 10) << RequireOpset(10) << std::endl;
    return 10;
  }
  if (score_info[0].Rank() != 3) {
    Error() << "Lod Tensor input is not supported, which means the shape of "
               "input(scores) is [M, C] now, but Paddle2ONNX only support [N, "
               "C, M]."
            << std::endl;
    return -1;
  }
  if (boxes_info[0].Rank() != 3) {
    Error() << "Only support input boxes as 3-D Tensor, but now it's rank is "
            << boxes_info[0].Rank() << "." << std::endl;
    return -1;
  }
  if (score_info[0].shape[1] <= 0) {
    Error() << "The 2nd-dimension of score should be fixed(means the number of "
               "classes), but now it's "
            << score_info[0].shape[1] << "." << std::endl;
    return -1;
  }

  if (this->deploy_backend == "tensorrt") {
    return 7;
  }

  Logger(verbose, 10) << RequireOpset(10) << std::endl;
  return 10;
}

void NMSMapper::KeepTopK(const std::string& selected_indices) {
  auto boxes_info = GetInput("BBoxes");
  auto score_info = GetInput("Scores");
  auto out_info = GetOutput("Out");
  auto index_info = GetOutput("Index");
  auto num_rois_info = GetOutput("NmsRoisNum");
  auto value_0 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(0));
  auto value_1 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(1));
  auto value_2 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(2));
  auto value_neg_1 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(-1));

  auto class_id = helper_->MakeNode("Gather", {selected_indices, value_1});
  AddAttribute(class_id, "axis", int64_t(1));

  auto box_id = helper_->MakeNode("Gather", {selected_indices, value_2});
  AddAttribute(box_id, "axis", int64_t(1));

  auto filtered_class_id = class_id->output(0);
  auto filtered_box_id = box_id->output(0);
  if (background_label_ >= 0) {
    auto filter_indices = MapperHelper::Get()->GenName("nms.filter_background");
    auto squeezed_class_id =
        helper_->Squeeze(filtered_class_id, std::vector<int64_t>(1, 1));
    if (background_label_ > 0) {
      auto background = helper_->Constant(
          {1}, ONNX_NAMESPACE::TensorProto::INT64, background_label_);
      auto diff = helper_->MakeNode("Sub", {squeezed_class_id, background});
      helper_->MakeNode("NonZero", {diff->output(0)}, {filter_indices});
    } else if (background_label_ == 0) {
      helper_->MakeNode("NonZero", {filtered_class_id}, {filter_indices});
    }
    auto new_class_id =
        helper_->MakeNode("Gather", {filtered_class_id, filter_indices});
    AddAttribute(new_class_id, "axis", int64_t(0));
    auto new_box_id =
        helper_->MakeNode("Gather", {filtered_box_id, filter_indices});
    AddAttribute(new_box_id, "axis", int64_t(0));
    filtered_class_id = new_class_id->output(0);
    filtered_box_id = new_box_id->output(0);
  }

  // Here is a little complicated
  // Since we need to gather all the scores for the final boxes to filter the
  // top-k boxes Now we have the follow inputs
  //    - scores: [N, C, M] N means batch size(but now it will be regarded as
  //    1); C means number of classes; M means number of boxes for each classes
  //    - selected_indices: [num_selected_indices, 3], and 3 means [batch,
  //    class_id, box_id]. We will use this inputs to gather score
  // So now we will first flatten `scores` to shape of [1 * C * M], then we
  // gather scores by each elements in `selected_indices` The index need be
  // calculated as
  //    `gather_index = class_id * M + box_id`
  auto flatten_score = helper_->Flatten(score_info[0].name);
  // M (boxes per class) is only known statically for fixed-size graphs. SSD
  // exports scores as [N, C, -1], and folding -1 into the flat gather index
  // computed below silently produced scores read from the wrong offsets — the
  // boxes were right, the scores were not. Fall back to the runtime shape.
  std::string num_boxes_each_class;
  if (score_info[0].shape[2] > 0) {
    num_boxes_each_class = helper_->Constant(
        {1}, ONNX_NAMESPACE::TensorProto::INT64, score_info[0].shape[2]);
  } else {
    num_boxes_each_class = helper_->Slice(
        helper_->MakeNode("Shape", {score_info[0].name})->output(0),
        {0},
        {2},
        {3});
  }
  auto gather_indices_0 =
      helper_->MakeNode("Mul", {filtered_class_id, num_boxes_each_class});
  auto gather_indices_1 =
      helper_->MakeNode("Add", {gather_indices_0->output(0), filtered_box_id});
  auto gather_indices = helper_->Flatten(gather_indices_1->output(0));
  auto gathered_scores =
      helper_->MakeNode("Gather", {flatten_score, gather_indices});
  AddAttribute(gathered_scores, "axis", int64_t(0));

  // Now we will perform keep_top_k process
  // First we need to check if the number of remaining boxes is greater than
  // keep_top_k Otherwise, we will downgrade the keep_top_k to number of
  // remaining boxes
  auto final_classes = filtered_class_id;  // shape of [num_selected_indices, 1]
  auto final_boxes_id = filtered_box_id;
  auto final_scores =
      gathered_scores->output(0);  // shape of [num_selected_indices]
  if (keep_top_k_ > 0) {
    // get proper topk
    auto shape_of_scores = helper_->MakeNode("Shape", {final_scores});
    auto num_of_boxes = helper_->Slice(shape_of_scores->output(0),
                                       std::vector<int64_t>(1, 0),
                                       std::vector<int64_t>(1, 0),
                                       std::vector<int64_t>(1, 1));
    auto top_k =
        helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, keep_top_k_);
    auto ensemble_value = helper_->MakeNode("Concat", {num_of_boxes, top_k});
    AddAttribute(ensemble_value, "axis", int64_t(0));

    std::shared_ptr<ONNX_NAMESPACE::NodeProto> new_top_k;
    if (OnnxHelper::GetOpsetVersion() >= 18) {
      std::string reduce_min_axis = helper_->Constant(
          {1}, ONNX_NAMESPACE::TensorProto::INT64, static_cast<int64_t>(0));
      new_top_k = helper_->MakeNode(
          "ReduceMin", {ensemble_value->output(0), reduce_min_axis});

    } else {
      new_top_k = helper_->MakeNode("ReduceMin", {ensemble_value->output(0)});
      AddAttribute(new_top_k, "axes", std::vector<int64_t>(1, 0));
    }
    AddAttribute(new_top_k, "keepdims", int64_t(1));

    // the output is topk_scores, topk_score_indices
    auto topk_node =
        helper_->MakeNode("TopK", {final_scores, new_top_k->output(0)}, 2);
    auto topk_scores =
        helper_->MakeNode("Gather", {final_scores, topk_node->output(1)});
    AddAttribute(topk_scores, "axis", int64_t(0));
    filtered_class_id = helper_->Flatten(filtered_class_id);
    auto topk_classes =
        helper_->MakeNode("Gather", {filtered_class_id, topk_node->output(1)});
    AddAttribute(topk_classes, "axis", int64_t(0));
    filtered_box_id = helper_->Flatten(filtered_box_id);
    auto topk_boxes_id =
        helper_->MakeNode("Gather", {filtered_box_id, topk_node->output(1)});
    AddAttribute(topk_boxes_id, "axis", int64_t(0));

    final_boxes_id = topk_boxes_id->output(0);
    final_scores = topk_scores->output(0);
    final_classes = topk_classes->output(0);

    auto topk_class_asc =
        helper_->MakeNode("TopK", {final_classes, new_top_k->output(0)}, 2);
    AddAttribute(topk_class_asc, "axis", int64_t(0));
    AddAttribute(topk_class_asc, "largest", int64_t(0));
    final_classes = topk_class_asc->output(0);
    final_boxes_id =
        helper_->MakeNode("Gather", {final_boxes_id, topk_class_asc->output(1)})
            ->output(0);
    final_scores =
        helper_->MakeNode("Gather", {final_scores, topk_class_asc->output(1)})
            ->output(0);
  }

  auto flatten_boxes_id = helper_->Flatten({final_boxes_id});
  auto gathered_selected_boxes =
      helper_->MakeNode("Gather", {boxes_info[0].name, flatten_boxes_id});
  AddAttribute(gathered_selected_boxes, "axis", int64_t(1));

  auto float_classes = helper_->MakeNode("Cast", {final_classes});
  AddAttribute(float_classes, "to", ONNX_NAMESPACE::TensorProto::FLOAT);

  std::vector<int64_t> shape{1, -1, 1};
  auto unsqueezed_scores = helper_->Reshape({final_scores}, shape);

  auto unsqueezed_class = helper_->Reshape({float_classes->output(0)}, shape);

  auto box_result = helper_->MakeNode("Concat",
                                      {unsqueezed_class,
                                       unsqueezed_scores,
                                       gathered_selected_boxes->output(0)});
  AddAttribute(box_result, "axis", int64_t(2));
  helper_->Squeeze(
      {box_result->output(0)}, {out_info[0].name}, std::vector<int64_t>(1, 0));

  // other outputs, we don't use sometimes
  // there's lots of Cast in exporting
  // TODO(jiangjiajun) A pass to eleminate all the useless Cast is needed
  auto reshaped_index_result =
      helper_->Reshape({flatten_boxes_id}, {int64_t(-1), int64_t(1)});
  auto index_result =
      helper_->MakeNode("Cast", {reshaped_index_result}, {index_info[0].name});
  AddAttribute(index_result, "to", GetOnnxDtype(index_info[0].dtype));

  auto out_box_shape = helper_->MakeNode("Shape", {out_info[0].name});
  auto num_rois_result = helper_->Slice({out_box_shape->output(0)},
                                        std::vector<int64_t>(1, 0),
                                        std::vector<int64_t>(1, 0),
                                        std::vector<int64_t>(1, 1));
  auto int32_num_rois_result = helper_->AutoCast(num_rois_result,
                                                 num_rois_info[0].name,
                                                 P2ODataType::INT64,
                                                 num_rois_info[0].dtype);
}

// Two-stage detectors run NMS over per-class boxes: BBoxes is [M, C, 4] and
// Scores is [M, C], with no batch dimension (batch size is 1 by construction).
//
// ONNX NonMaxSuppression takes boxes shared across classes, [B, M, 4], so the
// per-class boxes are folded into the *batch* axis instead: transposing to
// [C, M, 4] with scores [C, 1, M] makes ONNX run one independent single-class
// NMS per class, which is exactly Paddle's semantics. The returned "batch"
// index is then the class id.
void NMSMapper::ExportForLodInput() {
  auto boxes_info = GetInput("BBoxes");   // [M, C, 4]
  auto score_info = GetInput("Scores");   // [M, C]
  auto out_info = GetOutput("Out");
  auto index_info = GetOutput("Index");
  auto num_rois_info = GetOutput("NmsRoisNum");

  const int64_t num_classes = score_info[0].shape[1];
  auto i64 = [&](int64_t v) {
    return helper_->Constant(
        {1}, ONNX_NAMESPACE::TensorProto::INT64, v);
  };

  auto boxes_by_class = helper_->Transpose(boxes_info[0].name, {1, 0, 2});
  if (!normalized_) {
    // Paddle measures IoU on inclusive pixel coordinates when the boxes are
    // not normalised; widen xmax/ymax by one to match.
    auto one = helper_->Constant(
        {1}, GetOnnxDtype(boxes_info[0].dtype), static_cast<float>(1.0));
    auto parts =
        helper_->Split(boxes_by_class, std::vector<int64_t>(4, 1), 2);
    auto xmax = helper_->MakeNode("Add", {parts[2], one})->output(0);
    auto ymax = helper_->MakeNode("Add", {parts[3], one})->output(0);
    auto widened =
        helper_->MakeNode("Concat", {parts[0], parts[1], xmax, ymax});
    AddAttribute(widened, "axis", int64_t(2));
    boxes_by_class = widened->output(0);
  }
  // [M, C] -> [C, M] -> [C, 1, M]
  auto scores_by_class =
      helper_->Unsqueeze(helper_->Transpose(score_info[0].name, {1, 0}), {1});

  // Paddle spells "no per-class limit" as nms_top_k <= 0, but ONNX reads a
  // non-positive max_output_boxes_per_class as "select nothing" — so fall back
  // to the actual box count instead of passing the negative value through.
  auto num_boxes = helper_->Slice(
      helper_->MakeNode("Shape", {boxes_by_class})->output(0), {0}, {1}, {2});
  auto max_per_class = nms_top_k_ > 0 ? i64(nms_top_k_) : num_boxes;

  auto selected =
      helper_->MakeNode("NonMaxSuppression",
                        {boxes_by_class,
                         scores_by_class,
                         max_per_class,
                         helper_->Constant({1},
                                           ONNX_NAMESPACE::TensorProto::FLOAT,
                                           nms_threshold_),
                         helper_->Constant({1},
                                           ONNX_NAMESPACE::TensorProto::FLOAT,
                                           score_threshold_)})
          ->output(0);  // [K, 3] as (class, 0, box)

  auto gather_col = [&](const std::string& data, int64_t col) {
    auto n = helper_->MakeNode("Gather", {data, i64(col)});
    AddAttribute(n, "axis", int64_t(1));
    return n->output(0);  // [K, 1]
  };
  auto class_id = gather_col(selected, 0);
  auto box_id = gather_col(selected, 2);

  // ── Drop the background class ────────────────────────────────────────────
  if (background_label_ >= 0) {
    auto squeezed = helper_->Squeeze(class_id, {1});
    std::string keep;
    if (background_label_ == 0) {
      keep = helper_->MakeNode("NonZero", {squeezed})->output(0);
    } else {
      auto diff =
          helper_->MakeNode("Sub", {squeezed, i64(background_label_)})
              ->output(0);
      keep = helper_->MakeNode("NonZero", {diff})->output(0);
    }
    auto keep_1d = helper_->Reshape(keep, {-1});
    auto take = [&](const std::string& d) {
      auto n = helper_->MakeNode("Gather", {d, keep_1d});
      AddAttribute(n, "axis", int64_t(0));
      return n->output(0);
    };
    class_id = take(class_id);
    box_id = take(box_id);
  }

  // ── Scores and boxes of the survivors ────────────────────────────────────
  // Both Scores [M, C] and BBoxes [M, C, 4] are indexed by (box, class), so a
  // single flat index box_id * C + class_id serves for both.
  auto flat_index = helper_->Reshape(
      helper_->MakeNode(
                  "Add",
                  {helper_->MakeNode("Mul", {box_id, i64(num_classes)})
                       ->output(0),
                   class_id})
          ->output(0),
      {-1});
  auto flat_scores = helper_->Reshape(score_info[0].name, {-1});
  auto gather0 = [&](const std::string& data, const std::string& idx) {
    auto n = helper_->MakeNode("Gather", {data, idx});
    AddAttribute(n, "axis", int64_t(0));
    return n->output(0);
  };
  auto final_scores = gather0(flat_scores, flat_index);
  auto flat_boxes = helper_->Reshape(boxes_info[0].name, {-1, 4});
  auto final_boxes = gather0(flat_boxes, flat_index);
  auto final_classes = helper_->Reshape(class_id, {-1});
  auto final_box_ids = helper_->Reshape(box_id, {-1});

  // ── keep_top_k over everything that survived ─────────────────────────────
  auto flat_index_kept = flat_index;
  if (keep_top_k_ > 0) {
    auto num_left = helper_->Slice(
        helper_->MakeNode("Shape", {final_scores})->output(0), {0}, {0}, {1});
    auto k = helper_->MakeNode("Min", {num_left, i64(keep_top_k_)})->output(0);

    auto topk = helper_->MakeNode("TopK", {final_scores, k}, 2);
    AddAttribute(topk, "axis", int64_t(0));
    AddAttribute(topk, "largest", int64_t(1));
    AddAttribute(topk, "sorted", int64_t(1));
    auto order = topk->output(1);
    final_scores = topk->output(0);
    final_boxes = gather0(final_boxes, order);
    final_classes = gather0(final_classes, order);
    final_box_ids = gather0(final_box_ids, order);
    flat_index_kept = gather0(flat_index_kept, order);
  }

  // ── Match Paddle's output ordering: class ascending, box index ascending ──
  // Sorting by the composite key class * M + box_id reproduces it in one pass.
  {
    auto key = helper_->MakeNode(
                           "Add",
                           {helper_->MakeNode("Mul", {final_classes, num_boxes})
                                ->output(0),
                            final_box_ids})
                   ->output(0);
    auto n_final = helper_->Slice(
        helper_->MakeNode("Shape", {key})->output(0), {0}, {0}, {1});
    auto ordered = helper_->MakeNode("TopK", {key, n_final}, 2);
    AddAttribute(ordered, "axis", int64_t(0));
    AddAttribute(ordered, "largest", int64_t(0));
    AddAttribute(ordered, "sorted", int64_t(1));
    auto o = ordered->output(1);
    final_scores = gather0(final_scores, o);
    final_boxes = gather0(final_boxes, o);
    final_classes = gather0(final_classes, o);
    flat_index_kept = gather0(flat_index_kept, o);
  }

  // ── Assemble Out as [num, 6] = (class, score, xmin, ymin, xmax, ymax) ────
  auto class_f = helper_->MakeNode("Cast", {final_classes});
  AddAttribute(class_f, "to", GetOnnxDtype(boxes_info[0].dtype));
  auto out = helper_->MakeNode("Concat",
                               {helper_->Reshape(class_f->output(0), {-1, 1}),
                                helper_->Reshape(final_scores, {-1, 1}),
                                final_boxes});
  AddAttribute(out, "axis", int64_t(1));
  helper_->MakeNode("Identity", {out->output(0)}, {out_info[0].name});

  // Paddle's Index is the flattened (box, class) position, not the box id.
  helper_->AutoCast(helper_->Reshape(flat_index_kept, {-1, 1}),
                    index_info[0].name,
                    P2ODataType::INT64,
                    index_info[0].dtype);

  auto num = helper_->Slice(
      helper_->MakeNode("Shape", {out->output(0)})->output(0), {0}, {0}, {1});
  helper_->AutoCast(
      num, num_rois_info[0].name, P2ODataType::INT64, num_rois_info[0].dtype);
}

void NMSMapper::Opset10() {
  if (this->deploy_backend == "tensorrt") {
    return ExportForTensorRT();
  }
  auto boxes_info = GetInput("BBoxes");
  auto score_info = GetInput("Scores");
  if (score_info[0].Rank() == 2) {
    return ExportForLodInput();
  }
  if (boxes_info[0].shape[0] != 1) {
    Warn() << "Due to the operator multiclass_nms3, the exported ONNX model "
              "will only supports inference with input batch_size == 1."
           << std::endl;
  }
  int64_t num_classes = score_info[0].shape[1];
  auto score_threshold = helper_->Constant(
      {1}, ONNX_NAMESPACE::TensorProto::FLOAT, score_threshold_);
  auto nms_threshold = helper_->Constant(
      {1}, ONNX_NAMESPACE::TensorProto::FLOAT, nms_threshold_);
  auto nms_top_k =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, nms_top_k_);

  auto selected_box_index = MapperHelper::Get()->GenName("nms.selected_index");
  if (normalized_) {
    helper_->MakeNode("NonMaxSuppression",
                      {boxes_info[0].name,
                       score_info[0].name,
                       nms_top_k,
                       nms_threshold,
                       score_threshold},
                      {selected_box_index});
  } else {
    auto value_1 = helper_->Constant(
        {1}, GetOnnxDtype(boxes_info[0].dtype), static_cast<float>(1.0));
    auto split_boxes = helper_->Split(
        boxes_info[0].name, std::vector<int64_t>(4, 1), int64_t(2));
    auto xmax = helper_->MakeNode("Add", {split_boxes[2], value_1});
    auto ymax = helper_->MakeNode("Add", {split_boxes[3], value_1});
    auto new_boxes = helper_->MakeNode(
        "Concat",
        {split_boxes[0], split_boxes[1], xmax->output(0), ymax->output(0)});
    AddAttribute(new_boxes, "axis", int64_t(2));
    helper_->MakeNode("NonMaxSuppression",
                      {new_boxes->output(0),
                       score_info[0].name,
                       nms_top_k,
                       nms_threshold,
                       score_threshold},
                      {selected_box_index});
  }
  KeepTopK(selected_box_index);
}

void NMSMapper::ExportForTensorRT() {
  auto boxes_info = GetInput("BBoxes");
  auto score_info = GetInput("Scores");
  auto out_info = GetOutput("Out");
  auto index_info = GetOutput("Index");
  auto num_rois_info = GetOutput("NmsRoisNum");

  auto scores = helper_->Transpose(score_info[0].name, {0, 2, 1});
  auto boxes = helper_->Unsqueeze(boxes_info[0].name, {2});
  int64_t num_classes = score_info[0].shape[1];
  auto repeats =
      helper_->Constant(GetOnnxDtype(P2ODataType::INT64),
                        std::vector<int64_t>({1, 1, num_classes, 1}));
  boxes = helper_->MakeNode("Tile", {boxes, repeats})->output(0);

  auto nms_node =
      helper_->MakeNode("BatchedNMSDynamic_TRT", {boxes, scores}, 4);
  AddAttribute(nms_node, "shareLocation", int64_t(0));
  AddAttribute(nms_node, "backgroundLabelId", background_label_);
  AddAttribute(nms_node, "numClasses", num_classes);
  int64_t nms_top_k = nms_top_k_;
  int64_t keep_top_k = keep_top_k_;
  if (nms_top_k > 4096) {
    Warn()
        << "Paramter nms_top_k:" << nms_top_k
        << " is exceed limit in TensorRT BatchedNMS plugin, will force to 4096."
        << std::endl;
    nms_top_k = 4096;
  }
  if (keep_top_k > 4096) {
    Warn()
        << "Parameter keep_top_k:" << keep_top_k
        << " is exceed limit in TensorRT BatchedNMS plugin, will force to 4096."
        << std::endl;
    keep_top_k = 4096;
  }
  AddAttribute(nms_node, "topK", nms_top_k);
  AddAttribute(nms_node, "keepTopK", keep_top_k);
  AddAttribute(nms_node, "scoreThreshold", score_threshold_);
  AddAttribute(nms_node, "iouThreshold", nms_threshold_);
  if (normalized_) {
    AddAttribute(nms_node, "isNormalized", int64_t(1));
  } else {
    AddAttribute(nms_node, "isNormalized", int64_t(0));
  }
  AddAttribute(nms_node, "clipBoxes", int64_t(0));
  nms_node->set_domain("Paddle");

  auto num_rois = helper_->Reshape(nms_node->output(0), {-1});
  helper_->AutoCast(num_rois,
                    num_rois_info[0].name,
                    P2ODataType::INT32,
                    num_rois_info[0].dtype);

  auto out_classes = helper_->Reshape(nms_node->output(3), {-1, 1});
  auto out_scores = helper_->Reshape(nms_node->output(2), {-1, 1});
  auto out_boxes = helper_->Reshape(nms_node->output(1), {-1, 4});
  out_classes =
      helper_->AutoCast(out_classes, P2ODataType::INT32, P2ODataType::FP32);
  helper_->Concat({out_classes, out_scores, out_boxes}, {out_info[0].name}, 1);

  //  EfficientNMS_TRT cannot get the same result, so disable now
  //  auto nms_node = helper_->MakeNode("EfficientNMS_TRT", {boxes_info[0].name,
  //  score}, 4);
  //  AddAttribute(nms_node, "plugin_version", "1");
  //  AddAttribute(nms_node, "background_class", background_label_);
  //  AddAttribute(nms_node, "max_output_boxes", nms_top_k_);
  //  AddAttribute(nms_node, "score_threshold", score_threshold_);
  //  AddAttribute(nms_node, "iou_threshold", nms_threshold_);
  //  AddAttribute(nms_node, "score_activation", int64_t(0));
  //  AddAttribute(nms_node, "box_coding", int64_t(0));
  //  nms_node->set_domain("Paddle");
  //
  //  auto num_rois = helper_->Reshape(nms_node->output(0), {-1});
  //  helper_->AutoCast(num_rois, num_rois_info[0].name, P2ODataType::INT32,
  //  num_rois_info[0].dtype);
  //
  //  auto out_classes = helper_->Reshape(nms_node->output(3), {-1, 1});
  //  auto out_scores = helper_->Reshape(nms_node->output(2), {-1, 1});
  //  auto out_boxes = helper_->Reshape(nms_node->output(1), {-1, 4});
  //  out_classes = helper_->AutoCast(out_classes, P2ODataType::INT32,
  //  P2ODataType::FP32);
  //  helper_->Concat({out_classes, out_scores, out_boxes}, {out_info[0].name},
  //  1);
}

}  // namespace paddle2onnx
