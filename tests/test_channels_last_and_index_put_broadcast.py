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

# Regression cases for two mapper gaps that the auto-scan tests never reached:
#
# - batch_norm with a channels-last layout of rank 3 or 5. F.batch_norm maps
#   NLC and NDHWC onto the same data_format "NHWC" as rank 4, so a mapper that
#   assumes rank 4 emits a malformed Transpose. With L == C the pre-NHWC mapper
#   was worse: it converted, ran, and normalised over the wrong axis.
# - index_put whose index tensors broadcast rather than agree ([n] against
#   [m, 1]), with a value that does not already carry the broadcast shape.
#   test_auto_scan_index_put.py only generates indices of equal shape.

import numpy as np
import paddle
from onnxbase import APIOnnx
from onnxbase import randtool
from onnxbase import _test_with_pir
from onnxbase import _test_only_pir


def _bn(layer_cls, c, data_format):
    layer = layer_cls(c, data_format=data_format)
    rng = np.random.default_rng(0)
    # Non-trivial running statistics: with mean 0 / variance 1 a batch_norm
    # applied over the wrong axis can still look right.
    layer._mean.set_value(paddle.to_tensor(rng.normal(size=c).astype("float32")))
    layer._variance.set_value(
        paddle.to_tensor(rng.uniform(0.5, 2.0, size=c).astype("float32"))
    )
    layer.weight.set_value(paddle.to_tensor(rng.normal(size=c).astype("float32")))
    layer.bias.set_value(paddle.to_tensor(rng.normal(size=c).astype("float32")))
    layer.eval()
    return layer


def _run_bn(layer, name, shape):
    obj = APIOnnx(layer, name, [7, 14])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, shape).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_BatchNorm1D_NLC():
    _run_bn(_bn(paddle.nn.BatchNorm1D, 4, "NLC"), "nn_BatchNorm1D_NLC", [2, 7, 4])


@_test_with_pir
def test_BatchNorm1D_NLC_length_equals_channels():
    _run_bn(
        _bn(paddle.nn.BatchNorm1D, 4, "NLC"), "nn_BatchNorm1D_NLC_LeqC", [2, 4, 4]
    )


@_test_with_pir
def test_BatchNorm2D_NHWC():
    _run_bn(_bn(paddle.nn.BatchNorm2D, 4, "NHWC"), "nn_BatchNorm2D_NHWC", [2, 6, 7, 4])


@_test_with_pir
def test_BatchNorm3D_NDHWC():
    _run_bn(
        _bn(paddle.nn.BatchNorm3D, 4, "NDHWC"),
        "nn_BatchNorm3D_NDHWC",
        [2, 3, 5, 6, 4],
    )


# paddle.index_put only builds under PIR -- the old IR refuses it before any
# conversion -- so these run under PIR alone, which is also all Orion exports.
class IndexPutBroadcast(paddle.nn.Layer):
    def __init__(self, value_shape, accumulate=False):
        super(IndexPutBroadcast, self).__init__()
        self.value_shape = value_shape
        self.accumulate = accumulate

    def forward(self, x, i, j):
        v = paddle.full(self.value_shape, 3.5, dtype=x.dtype)
        return paddle.index_put(x, (i, j), v, accumulate=self.accumulate)


def _run_index_put(value_shape, accumulate, name):
    op = IndexPutBroadcast(value_shape, accumulate)
    op.eval()
    obj = APIOnnx(op, name, [16] if accumulate else [11, 16])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [4, 5, 6]).astype("float32")),
        # [3] against [2, 1]: they broadcast to [2, 3], they are not equal.
        paddle.to_tensor(np.array([0, 1, 3], dtype="int64")),
        paddle.to_tensor(np.array([[1], [4]], dtype="int64")),
    )
    obj.run()


@_test_only_pir
def test_index_put_broadcast_indices_scalar_value():
    _run_index_put([1], False, "index_put_bcast_scalar")


@_test_only_pir
def test_index_put_broadcast_indices_accumulate():
    _run_index_put([1], True, "index_put_bcast_accumulate")


@_test_only_pir
def test_index_put_broadcast_indices_full_value():
    _run_index_put([2, 3, 6], False, "index_put_bcast_full")
