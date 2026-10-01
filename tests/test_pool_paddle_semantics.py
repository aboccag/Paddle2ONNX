# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pooling behaviours ONNX's MaxPool / AveragePool attributes cannot express,
# one deterministic case each (paddle2onnx/mapper/nn/pool_exact.h):
#
# - under ceil_mode Paddle keeps a last window that starts in the right
#   padding, which onnxruntime drops (PyTorch's rule);
# - a non-exclusive average under ceil_mode divides by the window clipped to
#   the explicit padding, not to the overflow beyond it;
# - padding as large as the kernel (refused by onnxruntime, and 0 is the
#   wrong pad value for a max pool);
# - adaptive max pooling whose output does not divide its input, and adaptive
#   3-D average pooling likewise.
#
# Inputs are drawn negative: a max pool padded with 0 instead of -FLT_MAX would
# return 0 where Paddle returns the window's (negative) maximum.

import paddle
import paddle.nn.functional as F
from onnxbase import APIOnnx
from onnxbase import randtool
from onnxbase import _test_only_pir


class Pool(paddle.nn.Layer):
    def __init__(self, fn):
        super(Pool, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _run(fn, name, shape, opsets=(10, 14)):
    op = Pool(fn)
    op.eval()
    obj = APIOnnx(op, name, list(opsets))
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -5, -1, shape).astype("float32")),
    )
    obj.run()


# ── ceil_mode: a last window starting in the right padding is kept ─────────
@_test_only_pir
def test_max_pool2d_ceil_keeps_padding_window():
    # in 19, k 10, s 10, p 1: ceil gives 3 windows. The third starts at 19, in
    # the right padding; Paddle keeps it, onnxruntime would return 2.
    _run(lambda x: F.max_pool2d(x, 10, 10, 1, ceil_mode=True),
         "pool_max2d_ceil_keep", [2, 3, 19, 19])


@_test_only_pir
def test_avg_pool2d_ceil_keeps_padding_window():
    _run(lambda x: F.avg_pool2d(x, 7, 7, 3, ceil_mode=True),
         "pool_avg2d_ceil_keep", [2, 3, 10, 10])


@_test_only_pir
def test_max_pool3d_ceil_keeps_padding_window():
    _run(lambda x: F.max_pool3d(x, 10, 10, 1, ceil_mode=True),
         "pool_max3d_ceil_keep", [1, 2, 10, 10, 19])


@_test_only_pir
def test_avg_pool3d_ceil_keeps_padding_window():
    _run(lambda x: F.avg_pool3d(x, 7, 7, 3, ceil_mode=True),
         "pool_avg3d_ceil_keep", [1, 2, 10, 10, 10])


# ── ceil_mode + non-exclusive average: overflow is not in the divisor ───────
@_test_only_pir
def test_avg_pool2d_ceil_inclusive_divisor():
    # in 10, k 3, s 2, p 1: the last window covers one input row, one padding
    # row and one overflow row; Paddle divides by 2 rows, not 3.
    _run(lambda x: F.avg_pool2d(x, 3, 2, 1, ceil_mode=True, exclusive=False),
         "pool_avg2d_ceil_incl", [2, 3, 10, 10])


@_test_only_pir
def test_avg_pool3d_ceil_inclusive_divisor():
    _run(lambda x: F.avg_pool3d(x, 3, 2, 1, ceil_mode=True, exclusive=False),
         "pool_avg3d_ceil_incl", [1, 2, 10, 10, 10])


# ── padding as large as the kernel ───────────────────────────────────────────
@_test_only_pir
def test_max_pool2d_padding_not_smaller_than_kernel():
    # Corner windows lie entirely in the padding: Paddle returns -FLT_MAX.
    _run(lambda x: F.max_pool2d(x, 3, 2, 3), "pool_max2d_pad_ge_k",
         [2, 3, 10, 10])


@_test_only_pir
def test_avg_pool2d_padding_not_smaller_than_kernel_exclusive():
    # Corner windows hold no input: Paddle returns NaN (0 / 0).
    _run(lambda x: F.avg_pool2d(x, 3, 2, 3, exclusive=True),
         "pool_avg2d_pad_ge_k_excl", [2, 3, 10, 10])


@_test_only_pir
def test_avg_pool2d_padding_not_smaller_than_kernel_inclusive():
    _run(lambda x: F.avg_pool2d(x, 2, 1, 3, exclusive=False),
         "pool_avg2d_pad_ge_k_incl", [2, 3, 7, 7])


# ── adaptive pooling whose output does not divide its input ─────────────────
@_test_only_pir
def test_adaptive_max_pool2d_non_divisible():
    _run(lambda x: F.adaptive_max_pool2d(x, 3), "pool_adaptive_max2d_nondiv",
         [2, 3, 10, 7])


@_test_only_pir
def test_adaptive_max_pool3d_non_divisible():
    _run(lambda x: F.adaptive_max_pool3d(x, 3), "pool_adaptive_max3d_nondiv",
         [1, 2, 10, 12, 7])


@_test_only_pir
def test_adaptive_avg_pool3d_non_divisible():
    _run(lambda x: F.adaptive_avg_pool3d(x, 4), "pool_adaptive_avg3d_nondiv",
         [1, 2, 10, 7, 12])
