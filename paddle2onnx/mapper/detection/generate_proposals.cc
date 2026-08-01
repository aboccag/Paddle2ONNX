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

#include "paddle2onnx/mapper/detection/generate_proposals.h"

#include <cmath>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(generate_proposals, GenerateProposalsMapper)

// Paddle clamps the width/height deltas before exp() to keep the exponential
// from overflowing: phi::funcs::kBBoxClipDefault == std::log(1000.0 / 16.0).
static const float kBBoxClipDefault = 4.135166556742356f;

int32_t GenerateProposalsMapper::GetMinOpsetVersion(bool verbose) {
  // NonMaxSuppression (opset 10) plus Clip with tensor min/max (opset 11).
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void GenerateProposalsMapper::Opset11() {
  auto scores_info = GetInput(0);     // [N, A, H, W]
  auto deltas_info = GetInput(1);     // [N, 4*A, H, W]
  auto img_size_info = GetInput(2);   // [N, 2] as (height, width)
  auto anchors_info = GetInput(3);    // [H, W, A, 4]
  auto variances_info = GetInput(4);  // [H, W, A, 4]

  auto rois_info = GetOutput(0);       // [M, 4]
  auto probs_info = GetOutput(1);      // [M, 1]
  auto rois_num_info = GetOutput(2);   // [N]

  int32_t dtype = scores_info[0].dtype;
  auto onnx_dtype = GetOnnxDtype(dtype);
  const float offset = pixel_offset_ ? 1.0f : 0.0f;

  auto f_const = [&](float v) {
    return helper_->Constant({1}, onnx_dtype, v);
  };
  auto i64_const = [&](int64_t v) {
    return helper_->Constant(
        ONNX_NAMESPACE::TensorProto::INT64, std::vector<int64_t>(1, v));
  };

  // ── Flatten every per-anchor quantity to [H*W*A, ...] ─────────────────────
  // Paddle transposes NCHW -> NHWC and then reshapes, so the trailing 4*A axis
  // is anchor-major with the 4 box deltas contiguous — matching the [H,W,A,4]
  // layout of the anchors.
  auto scores_t = helper_->Transpose(scores_info[0].name, {0, 2, 3, 1});
  auto scores_flat = helper_->Reshape(scores_t, {-1});

  auto deltas_t = helper_->Transpose(deltas_info[0].name, {0, 2, 3, 1});
  auto deltas_2d = helper_->Reshape(deltas_t, {-1, 4});

  auto anchors_2d = helper_->Reshape(anchors_info[0].name, {-1, 4});
  auto variances_2d = helper_->Reshape(variances_info[0].name, {-1, 4});

  // ── Pre-NMS top-k by objectness ───────────────────────────────────────────
  // k = min(pre_nms_top_n, num_anchors); a non-positive attribute means "all".
  auto num_anchors = helper_->MakeNode("Shape", {scores_flat})->output(0);
  std::string k = num_anchors;
  if (pre_nms_top_n_ > 0) {
    k = helper_->MakeNode("Min", {num_anchors, i64_const(pre_nms_top_n_)})
            ->output(0);
  }
  auto topk = helper_->MakeNode("TopK", {scores_flat, k}, 2);
  AddAttribute(topk, "axis", static_cast<int64_t>(0));
  AddAttribute(topk, "largest", static_cast<int64_t>(1));
  AddAttribute(topk, "sorted", static_cast<int64_t>(1));
  auto sel_scores = topk->output(0);  // [k]
  auto sel_idx = topk->output(1);     // [k]

  auto gather = [&](const std::string& data, const std::string& idx) {
    auto n = helper_->MakeNode("Gather", {data, idx});
    AddAttribute(n, "axis", static_cast<int64_t>(0));
    return n->output(0);
  };
  auto sel_deltas = gather(deltas_2d, sel_idx);
  auto sel_anchors = gather(anchors_2d, sel_idx);
  auto sel_variances = gather(variances_2d, sel_idx);

  // ── Decode anchors + deltas into boxes (box_coder, decode_center_size) ────
  auto a = helper_->Split(sel_anchors, std::vector<int64_t>(4, 1), 1);
  auto d = helper_->Split(sel_deltas, std::vector<int64_t>(4, 1), 1);
  auto v = helper_->Split(sel_variances, std::vector<int64_t>(4, 1), 1);

  auto sub = [&](const std::string& x, const std::string& y) {
    return helper_->MakeNode("Sub", {x, y})->output(0);
  };
  auto add = [&](const std::string& x, const std::string& y) {
    return helper_->MakeNode("Add", {x, y})->output(0);
  };
  auto mul = [&](const std::string& x, const std::string& y) {
    return helper_->MakeNode("Mul", {x, y})->output(0);
  };

  auto offset_c = f_const(offset);
  auto half = f_const(0.5f);

  // anchor width/height/centre
  auto aw = add(sub(a[2], a[0]), offset_c);
  auto ah = add(sub(a[3], a[1]), offset_c);
  auto acx = add(a[0], mul(half, aw));
  auto acy = add(a[1], mul(half, ah));

  // predicted centre and size; the size deltas are clipped before exp()
  auto pcx = add(acx, mul(mul(d[0], v[0]), aw));
  auto pcy = add(acy, mul(mul(d[1], v[1]), ah));
  auto clip_c = f_const(kBBoxClipDefault);
  auto dw = helper_->MakeNode("Min", {mul(d[2], v[2]), clip_c})->output(0);
  auto dh = helper_->MakeNode("Min", {mul(d[3], v[3]), clip_c})->output(0);
  auto pw = mul(helper_->MakeNode("Exp", {dw})->output(0), aw);
  auto ph = mul(helper_->MakeNode("Exp", {dh})->output(0), ah);

  auto x1 = sub(pcx, mul(half, pw));
  auto y1 = sub(pcy, mul(half, ph));
  auto x2 = sub(add(pcx, mul(half, pw)), offset_c);
  auto y2 = sub(add(pcy, mul(half, ph)), offset_c);

  // ── Clip boxes to the image ───────────────────────────────────────────────
  // img_size is (height, width); take row 0 — batch size is 1 here.
  auto img_hw = helper_->Slice(img_size_info[0].name, {0}, {0}, {1});  // [1,2]
  auto img_h = helper_->Reshape(helper_->Slice(img_hw, {1}, {0}, {1}), {1});
  auto img_w = helper_->Reshape(helper_->Slice(img_hw, {1}, {1}, {2}), {1});
  if (img_size_info[0].dtype != dtype) {
    img_h = helper_->AutoCast(img_h, img_size_info[0].dtype, dtype);
    img_w = helper_->AutoCast(img_w, img_size_info[0].dtype, dtype);
  }
  auto max_x = sub(img_w, offset_c);
  auto max_y = sub(img_h, offset_c);
  auto zero_c = f_const(0.0f);

  auto clip_xy = [&](const std::string& val, const std::string& hi) {
    auto lo_clipped = helper_->MakeNode("Min", {val, hi})->output(0);
    return helper_->MakeNode("Max", {lo_clipped, zero_c})->output(0);
  };
  x1 = clip_xy(x1, max_x);
  y1 = clip_xy(y1, max_y);
  x2 = clip_xy(x2, max_x);
  y2 = clip_xy(y2, max_y);

  auto boxes = helper_->Concat({x1, y1, x2, y2}, 1);  // [k, 4]

  // ── Drop boxes smaller than min_size ──────────────────────────────────────
  // Paddle raises min_size to at least 1.0 before comparing.
  float min_size = std::max(min_size_, 1.0f);
  auto min_size_c = f_const(min_size);
  auto bw = add(sub(x2, x1), offset_c);
  auto bh = add(sub(y2, y1), offset_c);
  auto keep = helper_->MakeNode(
                          "And",
                          {helper_->MakeNode("GreaterOrEqual", {bw, min_size_c})
                               ->output(0),
                           helper_->MakeNode("GreaterOrEqual", {bh, min_size_c})
                               ->output(0)})
                  ->output(0);
  if (pixel_offset_) {
    // With the pixel offset convention Paddle additionally requires the box
    // centre to fall inside the image.
    auto cx = add(x1, mul(half, bw));
    auto cy = add(y1, mul(half, bh));
    auto inside =
        helper_->MakeNode(
                    "And",
                    {helper_->MakeNode("LessOrEqual", {cx, img_w})->output(0),
                     helper_->MakeNode("LessOrEqual", {cy, img_h})->output(0)})
            ->output(0);
    keep = helper_->MakeNode("And", {keep, inside})->output(0);
  }
  auto keep_1d = helper_->Reshape(keep, {-1});
  auto keep_idx = helper_->Reshape(
      helper_->MakeNode("NonZero", {keep_1d})->output(0), {-1});

  auto kept_boxes = gather(boxes, keep_idx);
  auto kept_scores = gather(sel_scores, keep_idx);

  // ── NMS ───────────────────────────────────────────────────────────────────
  // ONNX NonMaxSuppression documents its boxes as (y1, x1, y2, x2), but IoU of
  // axis-aligned boxes is invariant under swapping the two axes, so (x1, y1,
  // x2, y2) is fed straight through — as the multiclass_nms mapper already
  // does. Scores must be [batch, class, num_boxes].
  auto nms_boxes = helper_->Unsqueeze(kept_boxes, {0});      // [1, m, 4]
  auto nms_scores = helper_->Reshape(kept_scores, {1, 1, -1});
  if (dtype != P2ODataType::FP32) {
    nms_boxes = helper_->AutoCast(nms_boxes, dtype, P2ODataType::FP32);
    nms_scores = helper_->AutoCast(nms_scores, dtype, P2ODataType::FP32);
  }
  // Paddle reads post_nms_top_n <= 0 as "keep everything", whereas ONNX reads a
  // non-positive max_output_boxes_per_class as "keep nothing" — so pass the
  // candidate count in that case.
  auto num_candidates = helper_->Slice(
      helper_->MakeNode("Shape", {kept_boxes})->output(0), {0}, {0}, {1});
  auto max_out =
      post_nms_top_n_ > 0 ? i64_const(post_nms_top_n_) : num_candidates;
  auto iou_thr =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, nms_thresh_);
  auto selected =
      helper_->MakeNode("NonMaxSuppression",
                        {nms_boxes, nms_scores, max_out, iou_thr})
          ->output(0);  // [K, 3] as (batch, class, box index)

  // NonMaxSuppression emits candidates in descending score order, so simply
  // taking the box-index column preserves Paddle's ordering.
  auto box_col = helper_->Slice(selected, {1}, {2}, {3});
  auto nms_idx = helper_->Reshape(box_col, {-1});

  auto final_boxes = gather(kept_boxes, nms_idx);
  auto final_scores = gather(kept_scores, nms_idx);

  helper_->MakeNode("Identity", {final_boxes}, {rois_info[0].name});
  helper_->Reshape(final_scores, probs_info[0].name, {-1, 1});

  // rpn_rois_num: one element per image (batch size 1 → the proposal count).
  auto count = helper_->MakeNode("Shape", {final_boxes})->output(0);
  count = helper_->Slice(count, {0}, {0}, {1});
  helper_->AutoCast(count, rois_num_info[0].name, P2ODataType::INT64,
                    rois_num_info[0].dtype);
}

}  // namespace paddle2onnx
