#pragma once

#include <torch/autograd.h>
#include <torch/extension.h>

namespace gaussian_peft::cuda_field {

inline void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_float32_tensor(const torch::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

inline void check_index_tensor(const torch::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt32 || tensor.scalar_type() == torch::kInt64,
      name,
      " must be int32 or int64");
}

}  // namespace gaussian_peft::cuda_field
