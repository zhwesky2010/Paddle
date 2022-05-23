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

#include "paddle/phi/kernels/sparse/sparse_mm_kernel.h"

#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void CsrDenseMatmulKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
  std::vector<int64_t> ydim_vec = phi::vectorize(y.dims());
  auto x_ndims = xdim_vec.size();
  auto y_ndims = ydim_vec.size();
  PADDLE_ENFORCE_EQ(
      x_ndims,
      y_ndims,
      phi::errors::PreconditionNotMet("The dims size of Input(x) and Input(y) "
                                      "should be equal, But received X's "
                                      "dimensions=%d, Y's dimensions=%d.",
                                      x_ndims,
                                      y_ndims));
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      phi::errors::InvalidArgument("the dims size of Input(x) and "
                                   "Input(y) must be greater than "
                                   "or eaqual to 2."));

  for (size_t i = 0; i < x_ndims - 2; ++i) {
    PADDLE_ENFORCE_EQ(xdim_vec[i],
                      ydim_vec[i],
                      phi::errors::InvalidArgument(
                          "x.dim[%d] and x.dim[%d] must match.", i, i));
  }

  PADDLE_ENFORCE_GE(
      xdim_vec[x_ndims - 1],
      ydim_vec[y_ndims - 2],
      phi::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, x_dim[-1] must be eaqual to y_dim[-2]."));

  // Set Meta of DenseTensor 'out'
  int M = xdim_vec[x_ndims - 2];
  int N = xdim_vec[x_ndims - 1];
  int K = ydim_vec[y_ndims - 1];
  std::vector<int64_t> out_dim_vec(ydim_vec);
  out_dim_vec[y_ndims - 1] = M;
  out_dim_vec[y_ndims - 2] = K;
  out.Resize(phi::make_ddim(out_dim_vec));
  dev_ctx.template Alloc<T>(out);

  auto x_descriptor = CuSparseCsrMatDescriptor<T>(x);
  auto y_descriptor = CuSparseDnMatDescriptor<T>(y);
  auto out_descriptor = CuSparseDnMatDescriptor<T>(*out);

  auto sparse_blas = GetSparseBlas<Context, T>(dev_ctx);
  sparse_blas.DSDMM(false,
                    false,
                    static_cast<T>(1),
                    x_descriptor.descriptor(),
                    y_descriptor.descriptor(),
                    static_cast<T>(0),
                    out_descriptor.descriptor());
}

template <typename T, typename Context>
void CsrMatmulMaskKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const SparseCsrTensor& mask,
                         SparseCsrTensor* out) {
  std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
  std::vector<int64_t> ydim_vec = phi::vectorize(y.dims());
  std::vector<int64_t> maskdim_vec = phi::vectorize(y.dims());

  auto x_ndims = xdim_vec.size();
  auto y_ndims = ydim_vec.size();
  auto mask_ndims = ydim_vec.size();

  PADDLE_ENFORCE_EQ(
      x_ndims,
      y_ndims,
      phi::errors::PreconditionNotMet("The dims size of Input(x) and Input(y) "
                                      "should be equal, But received X's "
                                      "dimensions=%d, Y's dimensions=%d.",
                                      x_ndims,
                                      y_ndims));
  PADDLE_ENFORCE_EQ(x_ndims,
                    mask_ndims,
                    phi::errors::PreconditionNotMet(
                        "The dims size of Input(x) and Input(mask) "
                        "should be equal, But received X's "
                        "dimensions=%d, mask's dimensions=%d.",
                        x_ndims,
                        mask_ndims));
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      phi::errors::InvalidArgument("the dims size of Input(x) and "
                                   "Input(y) must be greater than "
                                   "or eaqual to 2."));

  for (size_t i = 0; i < x_ndims - 2; ++i) {
    PADDLE_ENFORCE_EQ(xdim_vec[i],
                      ydim_vec[i],
                      phi::errors::InvalidArgument(
                          "x.dim[%d] and x.dim[%d] must match.", i, i));
    PADDLE_ENFORCE_EQ(xdim_vec[i],
                      maskdim_vec[i],
                      phi::errors::InvalidArgument(
                          "x.dim[%d] and mask.dim[%d] must match.", i, i));
  }

  PADDLE_ENFORCE_GE(
      xdim_vec[x_ndims - 1],
      ydim_vec[y_ndims - 2],
      phi::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, x_dim[-1] must be eaqual to y_dim[-2]."));

  PADDLE_ENFORCE_GE(
      maskdim_vec[mask_ndims - 2],
      xdim_vec[x_ndims - 2],
      phi::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, x_dim[-1] must be eaqual to y_dim[-2]."));

  PADDLE_ENFORCE_GE(
      maskdim_vec[mask_ndims - 1],
      ydim_vec[y_ndims - 1],
      phi::errors::PreconditionNotMet(
          "The shape of Input(x) and Input(y) is not suitable for matmul "
          "opetation, x_dim[-1] must be eaqual to y_dim[-2]."));

  // Set Meta of SparseCsrTensor 'out'
  phi::Copy(dev_ctx,
            mask.non_zero_crows(),
            dev_ctx.GetPlace(),
            false,
            out->mutable_non_zero_crows());
  phi::Copy(dev_ctx,
            mask.non_zero_cols(),
            dev_ctx.GetPlace(),
            false,
            out->mutable_non_zero_cols());

  DenseTensor* values = out->mutable_non_zero_elements();
  values->ResizeAndAllocate(mask.mutable_non_zero_elements().dims());

  out->set_dims(mask.dims());

  auto x_descriptor = CuSparseDnMatDescriptor<T>(x);
  auto y_descriptor = CuSparseDnMatDescriptor<T>(y);
  auto out_descriptor = CuSparseCsrMatDescriptor<T>(*out);

  auto sparse_blas = GetSparseBlas<Context, T>(dev_ctx);
  sparse_blas.SDDMM(false,
                    false,
                    static_cast<T>(1),
                    x_descriptor.descriptor(),
                    y_descriptor.descriptor(),
                    static_cast<T>(0),
                    out_descriptor.descriptor());
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(csr_dense_matmul,
                   GPU,
                   SPARSE_CSR,
                   phi::sparse::CsrDenseMatmulKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(csr_mm_mask,
                   GPU,
                   SPARSE_CSR,
                   phi::sparse::CsrMatmulMaskKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
