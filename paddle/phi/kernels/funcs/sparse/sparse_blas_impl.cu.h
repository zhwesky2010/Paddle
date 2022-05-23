//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/dynload/cusparse.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

#include "paddle/fluid/memory/malloc.h"

namespace phi {
namespace funcs {

template <typename T>
cudaDataType_t GetGpuDataType() {
  if (std::is_same<T, float>::value) {
    return CUDA_R_32F;
  } else if (std::is_same<T, double>::value) {
    return CUDA_R_64F;
  } else if (std::is_same<T, platform::float16>::value) {
    return CUDA_R_16F;
  }
}

inline cusparseOperation_t GetTransposeOperation(const bool trans) {
  if (trans) {
    return CUSPARSE_OPERATION_TRANSPOSE;
  } else {
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  }
}

template <typename T>
class CuSparseCsrMatDescriptor {
 public:
  explicit CuSparseCsrMatDescriptor(const phi::SparseCsrTensor& x) {
    int* offset_data = x.non_zero_crows().data<int>();
    int* column_data = x.non_zero_cols().data<int>();
    T* value_data = x.non_zero_elements().data<T>();
    int nnz = x.nnz();

    std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
    auto x_ndims = xdim_vec.size();
    int M = static_cast<int>(xdim_vec[x_ndims - 2]);
    int N = static_cast<int>(xdim_vec[x_ndims - 1]);
    int batch_size = 1;
    for (int i = 0; i < x_ndims - 2; i++) {
      batch_size *= xdim_vec[i];
    }

    cudaDataType_t gpu_type = GetGpuDataType<T>();
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateCsr(&descriptor_,
                                      M,
                                      N,
                                      nnz,
                                      offset_data,
                                      column_data,
                                      value_data,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      gpu_type);
    });

    if (x_ndims > 2) {
      PADDLE_ENFORCE_GE(x.non_zero_crows().dims(), 2);
      PADDLE_ENFORCE_GE(x.non_zero_cols().dims(), 2);
      PADDLE_ENFORCE_GE(x.non_zero_elements().dims(), 2);
      dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
        phi::dynload::cusparseSpMatSetStridedBatch(
            descriptor_, batch_size, M * N, nnz);
      });
    } else {
      PADDLE_ENFORCE_EQ(x.non_zero_crows().dims(), 2);
      PADDLE_ENFORCE_EQ(x.non_zero_cols().dims(), 2);
      PADDLE_ENFORCE_EQ(x.non_zero_elements().dims(), 2);
      dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
        phi::dynload::cusparseSpMatSetStridedBatch(
            descriptor_, batch_size, 0, 0);
      });
    }
    VLOG(6) << "Destroy cusparseDnMatDescr_t " << &descriptor_;
  }

  ~CuSparseCsrMatDescriptor() {
    VLOG(6) << "Destroy cusparseDnMatDescr_t " << &descriptor_;
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseDestroySpMat(descriptor_);
    });
  }

  cusparseSpMatDescr_t& descriptor() { return descriptor_; }

 private:
  const DeviceContext& dev_ctx_;
  cusparseSpMatDescr_t descriptor_;
};

template <typename T>
class CuSparseDnMatDescriptor {
 public:
  explicit CuSparseDnMatDescriptor(const phi::DenseTensor& x) {
    T* x_data = x.data<T>();
    std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
    auto x_ndims = xdim_vec.size();
    int M = static_cast<int>(xdim_vec[x_ndims - 2]);
    int N = static_cast<int>(xdim_vec[x_ndims - 1]);
    int batch_size = 1;
    for (int i = 0; i < x_ndims - 2; i++) {
      batch_size *= xdim_vec[i];
    }

    cudaDataType_t gpu_type = GetGpuDataType<T>();
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateDnMat(
          &descriptor_, M, N, N, x_data, gpu_type, CUSPARSE_ORDER_ROW);
    });

    if (x_ndims > 2) {
      dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
        phi::dynload::cusparseDnMatSetStridedBatch(
            descriptor_, batch_size, M * N);
      });
    } else {
      dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
        phi::dynload::cusparseDnMatSetStridedBatch(descriptor_, batch_size, 0);
      });
    }
    VLOG(6) << "Destroy cusparseDnMatDescr_t " << &descriptor_;
  }

  ~CuSparseDnMatDescriptor() {
    VLOG(6) << "Destroy cusparseDnMatDescr_t " << &descriptor_;
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseDestroyDnMat(descriptor_);
    });
  }

  cusparseDnMatDescr_t& descriptor() { return descriptor_; }

 private:
  const DeviceContext& dev_ctx_;
  cusparseDnMatDescr_t descriptor_;
};

template <>
template <typename T>
void SparseBlas<phi::GPUContext>::DSDMM(const bool transa,
                                        const bool transb,
                                        const T* alpha,
                                        cusparseSpMatDescr_t matA,
                                        cusparseDnMatDescr_t matB,
                                        const T* beta,
                                        cusparseDnMatDescr_t matC) {
  cudaDataType_t gpu_type = GetGpuDataType<T>();
  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMM_bufferSize(handle,
                                          GetTransposeOperation(transa),
                                          GetTransposeOperation(transb),
                                          alpha,
                                          matA,
                                          matB,
                                          &beta,
                                          matC,
                                          gpu_type,
                                          CUSPARSE_SPMM_ALG_DEFAULT,
                                          buffer_size);
  });

  paddle::memory::allocation::AllocationPtr tmp_buffer =
      paddle::memory::Alloc(dev_ctx_, buffer_size);
  void* tmp_buffer_ptr = tmp_buffer->ptr();
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMM(handle,
                               GetTransposeOperation(transa),
                               GetTransposeOperation(transb),
                               alpha,
                               matA,
                               matB,
                               &beta,
                               matC,
                               gpu_type,
                               CUSPARSE_SPMM_ALG_DEFAULT,
                               tmp_buffer_ptr);
  });
}

template <>
template <typename T>
void SparseBlas<phi::GPUContext>::SDDMM(const bool transa,
                                        const bool transb,
                                        const T* alpha,
                                        cusparseDnMatDescr_t matA,
                                        cusparseDnMatDescr_t matB,
                                        const T* beta,
                                        cusparseSpMatDescr_t matC) {
  cudaDataType_t gpu_type = GetGpuDataType<T>();

  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM_bufferSize(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           alpha,
                                           matA,
                                           matB,
                                           beta,
                                           matC,
                                           gpu_type,
                                           CUSPARSE_SDDMM_ALG_DEFAULT,
                                           &bufferSize);
  });

  paddle::memory::allocation::AllocationPtr tmp_buffer =
      paddle::memory::Alloc(dev_ctx_, buffer_size);
  void* tmp_buffer_ptr = tmp_buffer->ptr();

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM_preprocess(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           alpha,
                                           matA,
                                           matB,
                                           beta,
                                           matC,
                                           gpu_type,
                                           CUSPARSE_SDDMM_ALG_DEFAULT,
                                           tmp_buffer_ptr);
  });

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM(handle,
                                GetTransposeOperation(transa),
                                GetTransposeOperation(transb),
                                alpha,
                                matA,
                                matB,
                                beta,
                                matC,
                                gpu_type,
                                CUSPARSE_SDDMM_ALG_DEFAULT,
                                tmp_buffer_ptr);
  });
}

}  // namespace funcs
}  // namespace phi
