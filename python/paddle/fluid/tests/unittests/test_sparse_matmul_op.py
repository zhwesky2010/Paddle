# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
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

import paddle
import numpy as np
import scipy.sparse as sp
import unittest


class TestCsrDenseMatmul2D(unittest.TestCase):
    # x: csr, y: dense, out: dense
    def test_matmul(self):
        mask = np.random.rand(10, 12) < 0.2
        np_x = np.random.rand(10, 12) * mask
        np_csr = sp.csr_array(np_x)
        np_dense = np.random.rand(12, 6)
        np_out = sp.matmul(np_csr, np_dense)

        np_out_grad = np.ones(10, 6)
        # dx(csr) = dout(dense) * y'(dense) * mask
        np_csr_grad = sp.csr_array(
            np.matmul(np_out_grad, np_dense.transpose(1, 0)) * mask)
        # dy(dense) = x'(csr) * dout(dense)
        np_dense_grad = sp.matmul(np_csr.transpose(1, 0), np_out_grad)

        csr = paddle.to_tensor(np_x, stop_gradient=False).to_sparse_csr()
        dense = paddle.to_tensor(np_dense, stop_gradient=False)

        out = paddle.sparse.mm(csr, dense)
        self.assertTrue(np.allclose(np_out, out.numpy()))

        out.backward()
        self.assertTrue(
            np.allclose(np_csr_grad.indptr, csr.grad.crows().numpy()))
        self.assertTrue(
            np.allclose(np_csr_grad.indices, csr.grad.cols().numpy()))
        self.assertTrue(
            np.allclose(np_csr_grad.data, csr.grad.values().numpy()))

        self.assertTrue(np.allclose(np_dense_grad, dense.grad.numpy()))


class TestCsrDenseMatmul3D(unittest.TestCase):
    # x: csr, y: dense, out: dense
    def test_matmul(self):
        np_x_list = []
        np_dense_list = []
        np_out = []
        np_csr_grad = []
        np_dense_grad = []
        for i in range(3):
            mask = np.random.rand(10, 12) < 0.2
            np_x = np.random.rand(10, 12) * mask
            np_csr = sp.csr_array(np_x)
            np_dense = np.random.rand(12, 6)
            np_out_list.append(sp.matmul(np_csr, np_dense))

            np_out_grad = np.ones(10, 6)
            # dx(csr) = dout(dense) * y'(dense) * mask
            np_csr_grad_list.append(
                np.matmul(np_out_grad, np_dense.transpose(1, 0)) * mask)
            # dy(dense) = x'(csr) * dout(dense)
            np_dense_grad_list.append(
                sp.matmul(np_csr.transpose(1, 0), np_out_grad))

            np_x_list.append(np_x)
            np_dense_list.append(np_dense)

        np_x = np.concatenate(np_x_list)
        np_dense = np.concatenate(np_dense_list)
        np_out = np.concatenate(np_out_list)
        np_csr_grad = np.concatenate(np_csr_grad_list)
        np_dense_grad = np.concatenate(np_dense_grad_list)

        csr = paddle.to_tensor(np_x, stop_gradient=False).to_sparse_csr()
        dense = paddle.to_tensor(np_dense, stop_gradient=False)
        out = paddle.sparse.mm(csr, dense)
        self.assertTrue(np.allclose(np_out, out.numpy()))

        out.backward()
        self.assertTrue(np.allclose(np_csr_grad, csr.grad.to_dense().numpy()))
        self.assertTrue(np.allclose(np_dense_grad, dense.grad.numpy()))


class TestCsrMatmulMask2D(unittest.TestCase):
    # x: dense, y: dense, out: csr
    def test_matmul(self):
        mask = np.random.rand(10, 6) < 0.2

        np_x = np.random.rand(10, 12)
        np_y = np.random.rand(12, 6)
        np_out = sp.csr_array(np.matmul(np_x, np_y) * mask)

        np_out_grad = sp.csr_array(np.ones(10, 6) * mask)
        # dx(dense) = dout(csr) * y'(dense)
        np_x_grad = sp.matmul(np_out_grad, np_y.transpose(1, 0))
        # dy(dense) = x'(dense) * dout(csr)
        np_y_grad = sp.matmul(np_x.transpose(1, 0), np_out_grad)

        x = paddle.to_tensor(np_x, stop_gradient=False)
        y = paddle.to_tensor(np_y, stop_gradient=False)
        mask = paddle.to_tensor(mask).to_sparse_csr()
        out = paddle.sparse.mm_mask(x, y, mask)
        self.assertTrue(np.allclose(np_out.indptr, out.crows().numpy()))
        self.assertTrue(np.allclose(np_out.indices, out.cols().numpy()))
        self.assertTrue(np.allclose(np_out.data, out.values().numpy()))

        out.backward()
        self.assertTrue(np.allclose(np_x_grad, x.grad.numpy()))
        self.assertTrue(np.allclose(np_y_grad, y.grad.numpy()))


class TestCsrMatmulMask3D(unittest.TestCase):
    # x: dense, y: dense, out: csr
    def test_matmul(self):
        np_x_list = []
        np_y_list = []
        np_out_list = []
        np_mask_list = []
        np_x_grad_list = []
        np_y_grad_list = []
        for i in range(3):
            np_mask = np.random.rand(10, 6) < 0.2

            np_x = np.random.rand(10, 12)
            np_y = np.random.rand(12, 6)
            np_out_list.append(np.matmul(np_x, np_y) * np_mask)

            np_out_grad = sp.csr_array(np.ones(10, 6) * np_mask)
            # dx(dense) = dout(csr) * y'(dense)
            np_x_grad_list.append(sp.matmul(np_out_grad, np_y.transpose(1, 0)))
            # dy(dense) = x'(dense) * dout(csr)
            np_y_grad_list.append(sp.matmul(np_x.transpose(1, 0), np_out_grad))

            np_x_list.append(np_x)
            np_y_list.append(np_y)
            np_mask_list.append(np_mask)

        np_x = np.concatenate(np_x_list)
        np_y = np.concatenate(np_y_list)
        np_mask = np.concatenate(np_mask_list)
        np_out = np.concatenate(np_out_list)
        np_x_grad = np.concatenate(np_x_grad_list)
        np_y_grad = np.concatenate(np_y_grad_list)

        x = paddle.to_tensor(np_x, stop_gradient=False)
        y = paddle.to_tensor(np_y, stop_gradient=False)
        mask = paddle.to_tensor(mask).to_sparse_csr()
        out = paddle.sparse.mm_mask(x, y, mask)
        self.assertTrue(np.allclose(np_out, out.to_dense().numpy()))

        out.backward()
        self.assertTrue(np.allclose(np_x_grad, x.grad.numpy()))
        self.assertTrue(np.allclose(np_y_grad, y.grad.numpy()))
