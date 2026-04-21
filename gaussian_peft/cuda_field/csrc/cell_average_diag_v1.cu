#include "field.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>

#include "common.h"

namespace gaussian_peft::cuda_field {

namespace {

constexpr float kInvSqrt2 = 0.70710678118654752440f;
constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
constexpr int kCellAverageThreads = 256;

__device__ inline float normal_pdf_standard(float z) {
  return kInvSqrt2Pi * expf(-0.5f * z * z);
}

__device__ inline float normal_cdf_standard(float z) {
  return 0.5f * (1.0f + erff(z * kInvSqrt2));
}

void validate_cell_average_diag_v1_inputs(
    const torch::Tensor& mu_raw,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    int64_t out_features,
    int64_t in_features,
    double sigma_min) {
  check_float32_tensor(mu_raw, "mu_raw");
  check_float32_tensor(chol_raw, "chol_raw");
  check_float32_tensor(amp, "amp");

  TORCH_CHECK(mu_raw.dim() == 2 && mu_raw.size(1) == 2, "mu_raw must have shape [K, 2]");
  TORCH_CHECK(chol_raw.dim() == 2 && chol_raw.size(1) == 3, "chol_raw must have shape [K, 3]");
  TORCH_CHECK(
      (amp.dim() == 1) || (amp.dim() == 2 && amp.size(1) == 1),
      "amp must have shape [K] or [K, 1]");
  TORCH_CHECK(mu_raw.size(0) == chol_raw.size(0), "mu_raw and chol_raw must agree on K");
  TORCH_CHECK(mu_raw.size(0) == amp.size(0), "mu_raw and amp must agree on K");
  TORCH_CHECK(out_features > 0, "out_features must be positive");
  TORCH_CHECK(in_features > 0, "in_features must be positive");
  TORCH_CHECK(sigma_min > 0.0, "sigma_min must be positive");
}

__global__ void interval_average_kernel(
    const float* mu_ptr,
    const float* sigma_ptr,
    int64_t num_gaussians,
    int64_t num_bins,
    float* avg_ptr) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_gaussians * num_bins;
  if (idx >= total) {
    return;
  }

  const int64_t gaussian_id = idx / num_bins;
  const int64_t bin_id = idx % num_bins;
  const float mu = mu_ptr[gaussian_id];
  const float sigma = sigma_ptr[gaussian_id];
  const float width = 2.0f / static_cast<float>(num_bins);
  const float left = -1.0f + width * static_cast<float>(bin_id);
  const float right = left + width;
  const float z_left = (left - mu) / sigma;
  const float z_right = (right - mu) / sigma;
  const float cdf_left = normal_cdf_standard(z_left);
  const float cdf_right = normal_cdf_standard(z_right);
  avg_ptr[idx] = (cdf_right - cdf_left) / width;
}

__global__ void interval_param_grad_kernel(
    const float* mu_ptr,
    const float* sigma_ptr,
    const float* upstream_ptr,
    int64_t num_gaussians,
    int64_t num_bins,
    float* d_mu_ptr,
    float* d_sigma_ptr) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_gaussians * num_bins;
  if (idx >= total) {
    return;
  }

  const int64_t gaussian_id = idx / num_bins;
  const int64_t bin_id = idx % num_bins;
  const float upstream = upstream_ptr[idx];
  if (upstream == 0.0f) {
    return;
  }

  const float mu = mu_ptr[gaussian_id];
  const float sigma = sigma_ptr[gaussian_id];
  const float width = 2.0f / static_cast<float>(num_bins);
  const float left = -1.0f + width * static_cast<float>(bin_id);
  const float right = left + width;
  const float z_left = (left - mu) / sigma;
  const float z_right = (right - mu) / sigma;
  const float pdf_left = normal_pdf_standard(z_left);
  const float pdf_right = normal_pdf_standard(z_right);
  const float inv_sigma_width = 1.0f / (sigma * width);
  const float d_avg_d_mu = (pdf_left - pdf_right) * inv_sigma_width;
  const float d_avg_d_sigma =
      (z_left * pdf_left - z_right * pdf_right) * inv_sigma_width;

  atomicAdd(d_mu_ptr + gaussian_id, upstream * d_avg_d_mu);
  atomicAdd(d_sigma_ptr + gaussian_id, upstream * d_avg_d_sigma);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
materialize_v1_parameters(
    const torch::Tensor& mu_raw,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    double sigma_min) {
  auto mu_raw_c = mu_raw.contiguous();
  auto chol_raw_c = chol_raw.contiguous();
  auto amp_c = amp.contiguous();

  auto mu = torch::tanh(mu_raw_c);
  auto sigma_x = torch::zeros({mu_raw.size(0)}, mu_raw.options());
  auto sigma_y = torch::zeros({mu_raw.size(0)}, mu_raw.options());
  const float sigma_floor = static_cast<float>(sigma_min);

  // Keep sigma materialization on CUDA but avoid an extra tensor-expression graph.
  auto sigma_x_view = chol_raw_c.select(1, 0);
  auto sigma_y_view = chol_raw_c.select(1, 2);
  sigma_x.copy_(torch::log1p(torch::exp(-torch::abs(sigma_x_view))) + torch::clamp_min(sigma_x_view, 0.0) + sigma_floor);
  sigma_y.copy_(torch::log1p(torch::exp(-torch::abs(sigma_y_view))) + torch::clamp_min(sigma_y_view, 0.0) + sigma_floor);

  return {chol_raw_c, amp_c, mu, sigma_x, sigma_y};
}

torch::Tensor compute_interval_average_axis(
    const torch::Tensor& mu_axis,
    const torch::Tensor& sigma_axis,
    int64_t num_bins) {
  auto avg = torch::empty({mu_axis.size(0), num_bins}, mu_axis.options());
  const int64_t total = mu_axis.size(0) * num_bins;
  const int blocks = static_cast<int>((total + kCellAverageThreads - 1) / kCellAverageThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  interval_average_kernel<<<blocks, kCellAverageThreads, 0, stream>>>(
      mu_axis.data_ptr<float>(),
      sigma_axis.data_ptr<float>(),
      mu_axis.size(0),
      num_bins,
      avg.data_ptr<float>());
  C10_CUDA_CHECK(cudaGetLastError());
  return avg;
}

std::pair<torch::Tensor, torch::Tensor> accumulate_axis_param_grads(
    const torch::Tensor& mu_axis,
    const torch::Tensor& sigma_axis,
    const torch::Tensor& upstream,
    int64_t num_bins) {
  auto d_mu = torch::zeros({mu_axis.size(0)}, mu_axis.options());
  auto d_sigma = torch::zeros({mu_axis.size(0)}, mu_axis.options());
  const int64_t total = mu_axis.size(0) * num_bins;
  const int blocks = static_cast<int>((total + kCellAverageThreads - 1) / kCellAverageThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  interval_param_grad_kernel<<<blocks, kCellAverageThreads, 0, stream>>>(
      mu_axis.data_ptr<float>(),
      sigma_axis.data_ptr<float>(),
      upstream.data_ptr<float>(),
      mu_axis.size(0),
      num_bins,
      d_mu.data_ptr<float>(),
      d_sigma.data_ptr<float>());
  C10_CUDA_CHECK(cudaGetLastError());
  return {d_mu, d_sigma};
}

}  // namespace

torch::Tensor cell_average_diag_v1_forward(
    const torch::Tensor& mu_raw,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    int64_t out_features,
    int64_t in_features,
    double sigma_min) {
  validate_cell_average_diag_v1_inputs(mu_raw, chol_raw, amp, out_features, in_features, sigma_min);
  c10::cuda::CUDAGuard device_guard(mu_raw.device());

  auto [chol_raw_c, amp_c, mu, sigma_x, sigma_y] =
      materialize_v1_parameters(mu_raw, chol_raw, amp, sigma_min);
  (void)chol_raw_c;

  auto mu_x = mu.select(1, 0).contiguous();
  auto mu_y = mu.select(1, 1).contiguous();
  auto avg_x = compute_interval_average_axis(mu_x, sigma_x, in_features);
  auto avg_y = compute_interval_average_axis(mu_y, sigma_y, out_features);
  auto amp_flat = amp_c.reshape({amp_c.size(0), 1});
  auto weighted_y = avg_y * amp_flat;
  auto delta = torch::matmul(weighted_y.transpose(0, 1), avg_x);
  return delta.contiguous();
}

std::vector<torch::Tensor> cell_average_diag_v1_backward(
    const torch::Tensor& grad_delta,
    const torch::Tensor& mu_raw,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    int64_t out_features,
    int64_t in_features,
    double sigma_min) {
  validate_cell_average_diag_v1_inputs(mu_raw, chol_raw, amp, out_features, in_features, sigma_min);
  c10::cuda::CUDAGuard device_guard(mu_raw.device());

  auto grad_matrix = grad_delta.contiguous();
  check_float32_tensor(grad_matrix, "grad_delta");
  if (grad_matrix.dim() == 1) {
    TORCH_CHECK(
        grad_matrix.numel() == out_features * in_features,
        "grad_delta flattened shape does not match [out_features, in_features]");
    grad_matrix = grad_matrix.view({out_features, in_features});
  } else {
    TORCH_CHECK(
        grad_matrix.dim() == 2 && grad_matrix.size(0) == out_features &&
            grad_matrix.size(1) == in_features,
        "grad_delta must have shape [out_features, in_features]");
  }

  auto [chol_raw_c, amp_c, mu, sigma_x, sigma_y] =
      materialize_v1_parameters(mu_raw, chol_raw, amp, sigma_min);
  auto mu_x = mu.select(1, 0).contiguous();
  auto mu_y = mu.select(1, 1).contiguous();
  auto avg_x = compute_interval_average_axis(mu_x, sigma_x, in_features);
  auto avg_y = compute_interval_average_axis(mu_y, sigma_y, out_features);

  auto proj_y = torch::matmul(grad_matrix, avg_x.transpose(0, 1));
  auto proj_y_t = proj_y.transpose(0, 1).contiguous();
  auto proj_x = torch::matmul(avg_y, grad_matrix).contiguous();

  auto amp_flat = amp_c.reshape({amp_c.size(0), 1});
  auto d_avg_y = proj_y_t * amp_flat;
  auto d_avg_x = proj_x * amp_flat;
  auto d_amp = (avg_y * proj_y_t).sum(1, true).contiguous();

  auto [d_mu_x, d_sigma_x] = accumulate_axis_param_grads(mu_x, sigma_x, d_avg_x, in_features);
  auto [d_mu_y, d_sigma_y] = accumulate_axis_param_grads(mu_y, sigma_y, d_avg_y, out_features);

  auto d_mu = torch::stack({d_mu_x, d_mu_y}, 1);
  auto d_mu_raw = d_mu * (1.0f - mu * mu);

  auto d_chol_raw = torch::zeros_like(chol_raw_c);
  d_chol_raw.select(1, 0).copy_(d_sigma_x * torch::sigmoid(chol_raw_c.select(1, 0)));
  d_chol_raw.select(1, 2).copy_(d_sigma_y * torch::sigmoid(chol_raw_c.select(1, 2)));

  return {
      d_mu_raw.contiguous(),
      d_chol_raw.contiguous(),
      d_amp.contiguous(),
  };
}

}  // namespace gaussian_peft::cuda_field
