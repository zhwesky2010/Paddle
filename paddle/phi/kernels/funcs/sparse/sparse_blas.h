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

#include "paddle/phi/core/device_context.h"

namespace phi {
namespace funcs {

template <typename DeviceContext>
class SparseBlas {
 public:
  explicit SparseBlas(const DeviceContext& dev_ctx) : dev_ctx_(dev_ctx) {}

  template <typename T>
  void DSDMM(const bool transa,
             const bool transb,
             const T* alpha,
             cusparseSpMatDescr_t matA,
             cusparseDnMatDescr_t matB,
             const T* beta,
             cusparseDnMatDescr_t matC) const;

 private:
  const DeviceContext& dev_ctx_;
};

template <typename DeviceContext, typename T>
class SparseBlasT : private SparseBlas<DeviceContext> {
 public:
  using SparseBlas<DeviceContext>::SparseBlas;

  template <typename... ARGS>
  void DSDMM(ARGS... args) const {
    Base()->template DSDMM<T>(args...);
  }

 private:
  const SparseBlas<DeviceContext>* Base() const {
    return static_cast<const SparseBlas<DeviceContext>*>(this);
  }
};

template <typename DeviceContext, typename T>
inline SparseBlasT<DeviceContext, T> GetSparseBlas(
    const DeviceContext& dev_ctx) {
  return SparseBlasT<DeviceContext, T>(dev_ctx);
}

}  // namespace funcs
}  // namespace phi

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/kernels/funcs/sparse/sparse_blas_impl.cu.h"
#endif
