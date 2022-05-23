/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/sparse_mm_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void CsrDenseMatmulGradKernel(const Context& dev_ctx,
                              const SparseCsrTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              SparseCsrTensor* dx,
                              DenseTensor* dy) {
  auto x_descriptor = CuSparseCsrMatDescriptor<T>(x);
  auto y_descriptor = CuSparseDnMatDescriptor<T>(y);
  auto dout_descriptor = CuSparseDnMatDescriptor<T>(*dout);

  auto sparse_blas = GetSparseBlas<Context, T>(dev_ctx);

  // dx{SparseCsr} = dout{Dense} * y'{Dense}
  if (dx) {
    // Set Meta of SparseCsrTensor 'dx'
    DenseTensor crows = phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_crows());
    phi::Copy(dev_ctx, x.non_zero_crows(), dev_ctx.GetPlace(), false, &crows);

    DenseTensor cols = phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_cols());
    phi::Copy(dev_ctx, x.non_zero_cols(), dev_ctx.GetPlace(), false, &cols);

    DenseTensor values =
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_elements());

    dx->SetMember(crows, cols, values, x.dims());

    auto dx_descriptor = CuSparseCsrMatDescriptor<T>(*dx);

    sparse_blas.SDDMM(false,
                      true,
                      static_cast<T>(1),
                      dout_descriptor.descriptor(),
                      y_descriptor.descriptor(),
                      static_cast<T>(0),
                      dx_descriptor.descriptor());
  }

  // dy{Dense} = x'{SparseCsr} * dout{Dense}
  if (dy) {
    // Set Meta of DenseTensor 'dy'
    dy->Resize(y.dims());
    auto dy_descriptor = CuSparseCsrMatDescriptor<T>(*dy);

    sparse_blas.DSDMM(true,
                      false,
                      static_cast<T>(1),
                      x_descriptor.descriptor(),
                      dout_descriptor.descriptor(),
                      static_cast<T>(0),
                      dy_descriptor.descriptor());
  }
}

template <typename T, typename Context>
void CsrMatmulMaskGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             const SparseCsrTensor& dout,
                             DenseTensor* dx,
                             DenseTensor* dy) {
  auto x_descriptor = CuSparseDnMatDescriptor<T>(x);
  auto y_descriptor = CuSparseDnMatDescriptor<T>(y);
  auto dout_descriptor = CuSparseCsrMatDescriptor<T>(*dout);

  auto sparse_blas = GetSparseBlas<Context, T>(dev_ctx);

  // dx{Dense} = dout{SparseCsr} * y'{Dense}
  if (dx) {
    // Set Meta of DenseTensor 'dx'
    dx->Resize(x.dims());
    auto dx_descriptor = CuSparseDnMatDescriptor<T>(*dx);

    sparse_blas.DSDMM(false,
                      true,
                      static_cast<T>(1),
                      dout_descriptor.descriptor(),
                      dy_descriptor.descriptor(),
                      static_cast<T>(0),
                      dx_descriptor.descriptor());
  }

  // dy{Dense} = x'{Dense} * dout{SparseCsr}
  if (dy) {
    // Set Meta of DenseTensor 'dy'
    dy->Resize(y.dims());
    auto dy_descriptor = CuSparseDnMatDescriptor<T>(*dy);

    sparse_blas.DSDMM(true,
                      false,
                      static_cast<T>(1),
                      x_descriptor.descriptor(),
                      dout_descriptor.descriptor(),
                      static_cast<T>(0),
                      dy_descriptor.descriptor());
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(csr_dense_matmul_grad,
                   GPU,
                   SPARSE_CSR,
                   phi::sparse::CsrDenseMatmulGradKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(csr_dense_matmul_grad,
                   GPU,
                   SPARSE_CSR,
                   phi::sparse::CsrMatmulMaskGradKernel,
                   float,
                   double) {}
