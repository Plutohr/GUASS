#include "field.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>

#include "common.h"

namespace gaussian_peft::cuda_field {

namespace {

constexpr double kCholeskyEps = 1e-5;
constexpr double kCovarianceFloor = 1e-8;
constexpr double kTwoPi = 6.28318530717958647692;
constexpr int kBackwardThreads = 256;
constexpr int64_t kPartialGradBytesLimit = 256LL * 1024LL * 1024LL;
constexpr double kOccupancyImbalanceRatio = 4.0;
constexpr int64_t kOccupancyMaxCountFloor = 64;

__device__ inline float softplus_stable_scalar(float x) {
  return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}

__device__ inline float sigmoid_scalar(float x) {
  return 1.0f / (1.0f + expf(-x));
}

int64_t compute_num_tile_rows(int64_t out_features, int64_t tile_out) {
  return (out_features + tile_out - 1) / tile_out;
}

int64_t compute_num_tile_cols(int64_t in_features, int64_t tile_in) {
  return (in_features + tile_in - 1) / tile_in;
}

BinningState make_saved_binning_state(const SavedTensorBundle& saved) {
  auto empty = torch::empty({0}, saved.tile_ptr.options());
  return BinningState{
      empty,
      empty,
      empty,
      empty,
      empty,
      empty,
      empty,
      saved.gaussian_ids_sorted,
      saved.tile_ptr,
      saved.gaussian_ids_sorted.numel(),
      saved.gaussian_ids_sorted.numel(),
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cub_sort_pairs_int64(
    const torch::Tensor& keys_in,
    const torch::Tensor& values_in) {
  auto keys_out = torch::empty_like(keys_in);
  auto values_out = torch::empty_like(values_in);
  size_t temp_storage_bytes = 0;
  auto stream = at::cuda::getDefaultCUDAStream();
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_storage_bytes,
      keys_in.data_ptr<int64_t>(),
      keys_out.data_ptr<int64_t>(),
      values_in.data_ptr<int64_t>(),
      values_out.data_ptr<int64_t>(),
      keys_in.numel(),
      0,
      sizeof(int64_t) * 8,
      stream));
  auto temp_storage = torch::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      keys_in.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp_storage.data_ptr(),
      temp_storage_bytes,
      keys_in.data_ptr<int64_t>(),
      keys_out.data_ptr<int64_t>(),
      values_in.data_ptr<int64_t>(),
      values_out.data_ptr<int64_t>(),
      keys_in.numel(),
      0,
      sizeof(int64_t) * 8,
      stream));
  return {keys_out, values_out, temp_storage};
}

std::pair<torch::Tensor, torch::Tensor> cub_reduce_sum_by_key_int64_float(
    const torch::Tensor& keys_in,
    const torch::Tensor& values_in) {
  auto stream = at::cuda::getDefaultCUDAStream();
  auto keys_out = torch::empty_like(keys_in);
  auto values_out = torch::empty_like(values_in);
  auto num_segments = torch::zeros({1}, keys_in.options().dtype(torch::kInt64));
  size_t temp_storage_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
      nullptr,
      temp_storage_bytes,
      keys_in.data_ptr<int64_t>(),
      keys_out.data_ptr<int64_t>(),
      values_in.data_ptr<float>(),
      values_out.data_ptr<float>(),
      num_segments.data_ptr<int64_t>(),
      cub::Sum{},
      keys_in.numel(),
      stream));
  auto temp_storage = torch::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      keys_in.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
      temp_storage.data_ptr(),
      temp_storage_bytes,
      keys_in.data_ptr<int64_t>(),
      keys_out.data_ptr<int64_t>(),
      values_in.data_ptr<float>(),
      values_out.data_ptr<float>(),
      num_segments.data_ptr<int64_t>(),
      cub::Sum{},
      keys_in.numel(),
      stream));
  return {keys_out, values_out};
}

struct BackwardChunkPlan {
  int64_t num_pairs;
  int64_t pair_bytes;
  int64_t chunk_pairs;
  int64_t estimated_total_bytes;
  int64_t max_tile_count;
  double mean_tile_count;
  double imbalance_ratio;
  bool use_chunking;
};

int64_t estimate_pair_gradient_bytes(int64_t num_pairs) {
  constexpr int64_t kBytesPerPair =
      static_cast<int64_t>(sizeof(int64_t)) * 2 +
      static_cast<int64_t>(sizeof(float)) * (1 + 2 + 4 + 1);
  return num_pairs * kBytesPerPair;
}

BackwardChunkPlan make_backward_chunk_plan(
    const torch::Tensor& tile_ptr_i64,
    int64_t num_pairs) {
  const auto tile_ptr_cpu = tile_ptr_i64.to(torch::kCPU).contiguous();
  const auto* tile_ptr = tile_ptr_cpu.data_ptr<int64_t>();
  const int64_t num_tiles = std::max<int64_t>(0, tile_ptr_cpu.numel() - 1);

  int64_t max_tile_count = 0;
  int64_t total = 0;
  for (int64_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
    const int64_t count = tile_ptr[tile_id + 1] - tile_ptr[tile_id];
    total += count;
    if (count > max_tile_count) {
      max_tile_count = count;
    }
  }
  const double mean_tile_count =
      num_tiles > 0 ? static_cast<double>(total) / static_cast<double>(num_tiles) : 0.0;
  const double imbalance_ratio =
      mean_tile_count > 0.0 ? static_cast<double>(max_tile_count) / mean_tile_count : 0.0;

  const int64_t estimated_total_bytes = estimate_pair_gradient_bytes(num_pairs);
  const bool exceeds_bytes = estimated_total_bytes > kPartialGradBytesLimit;
  const bool exceeds_occupancy =
      max_tile_count > kOccupancyMaxCountFloor &&
      imbalance_ratio > kOccupancyImbalanceRatio;

  int64_t chunk_pairs = num_pairs;
  if (exceeds_bytes) {
    chunk_pairs = std::max<int64_t>(1, kPartialGradBytesLimit / estimate_pair_gradient_bytes(1));
  } else if (exceeds_occupancy) {
    chunk_pairs = std::max<int64_t>(1, num_pairs / 2);
  }
  chunk_pairs = std::min<int64_t>(chunk_pairs, std::max<int64_t>(1, num_pairs));

  return BackwardChunkPlan{
      num_pairs,
      estimate_pair_gradient_bytes(1),
      chunk_pairs,
      estimated_total_bytes,
      max_tile_count,
      mean_tile_count,
      imbalance_ratio,
      exceeds_bytes || exceeds_occupancy,
  };
}

PairGradientBuffer allocate_pair_gradient_buffer(
    const torch::Tensor& gaussian_ids_chunk,
    const torch::Tensor& grad_delta) {
  const auto num_pairs = gaussian_ids_chunk.numel();
  auto float_opts = grad_delta.options().dtype(torch::kFloat32);
  auto index_opts = gaussian_ids_chunk.options().dtype(torch::kInt64);
  return PairGradientBuffer{
      gaussian_ids_chunk.to(torch::kInt64).contiguous(),
      torch::empty({num_pairs}, index_opts),
      torch::zeros({num_pairs}, float_opts),
      torch::zeros({num_pairs, 2}, float_opts),
      torch::zeros({num_pairs, 2, 2}, float_opts),
      torch::zeros({num_pairs}, float_opts),
  };
}

__global__ void fill_pair_tile_ids_range_kernel(
    const int64_t* tile_ptr_ptr,
    int64_t* pair_tile_id_ptr,
    int64_t num_tiles,
    int64_t pair_start,
    int64_t num_pairs_chunk) {
  const int64_t local_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_idx >= num_pairs_chunk) {
    return;
  }

  const int64_t global_pair_idx = pair_start + local_idx;
  int64_t lo = 0;
  int64_t hi = num_tiles;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (tile_ptr_ptr[mid + 1] <= global_pair_idx) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  pair_tile_id_ptr[local_idx] = lo;
}

__global__ void backward_pair_partial_kernel(
    const float* row_coords_ptr,
    const float* col_coords_ptr,
    const float* mu_ptr,
    const float* amp_ptr,
    const float* inv_cov_ptr,
    const float* det_cov_ptr,
    const int64_t* pair_tile_id_ptr,
    const int64_t* pair_gaussian_id_ptr,
    const float* grad_delta_ptr,
    float* partial_amp_ptr,
    float* partial_mu_ptr,
    float* partial_inv_cov_ptr,
    float* partial_det_cov_ptr,
    int64_t out_features,
    int64_t in_features,
    int64_t tile_out,
    int64_t tile_in,
    int64_t num_tile_cols,
    bool normalize,
    bool has_clamp_quad,
    double clamp_quad,
    int64_t num_pairs) {
  __shared__ float sh_amp[kBackwardThreads];
  __shared__ float sh_mu0[kBackwardThreads];
  __shared__ float sh_mu1[kBackwardThreads];
  __shared__ float sh_inv00[kBackwardThreads];
  __shared__ float sh_inv01[kBackwardThreads];
  __shared__ float sh_inv10[kBackwardThreads];
  __shared__ float sh_inv11[kBackwardThreads];
  __shared__ float sh_det[kBackwardThreads];

  const int64_t pair_id = static_cast<int64_t>(blockIdx.x);
  if (pair_id >= num_pairs) {
    return;
  }

  const int64_t tile_id = pair_tile_id_ptr[pair_id];
  const int64_t gaussian_id = pair_gaussian_id_ptr[pair_id];
  const int64_t tile_row = tile_id / num_tile_cols;
  const int64_t tile_col = tile_id % num_tile_cols;
  const int64_t row_start = tile_row * tile_out;
  const int64_t row_end = min(row_start + tile_out, out_features);
  const int64_t col_start = tile_col * tile_in;
  const int64_t col_end = min(col_start + tile_in, in_features);
  const int64_t tile_rows = row_end - row_start;
  const int64_t tile_cols = col_end - col_start;
  const int64_t tile_elems = tile_rows * tile_cols;

  const float mu_row = mu_ptr[gaussian_id * 2 + 0];
  const float mu_col = mu_ptr[gaussian_id * 2 + 1];
  const float amp = amp_ptr[gaussian_id];
  const float inv00 = inv_cov_ptr[gaussian_id * 4 + 0];
  const float inv01 = inv_cov_ptr[gaussian_id * 4 + 1];
  const float inv10 = inv_cov_ptr[gaussian_id * 4 + 2];
  const float inv11 = inv_cov_ptr[gaussian_id * 4 + 3];
  const float det_cov = det_cov_ptr[gaussian_id];

  float acc_amp = 0.0f;
  float acc_mu0 = 0.0f;
  float acc_mu1 = 0.0f;
  float acc_inv00 = 0.0f;
  float acc_inv01 = 0.0f;
  float acc_inv10 = 0.0f;
  float acc_inv11 = 0.0f;
  float acc_det = 0.0f;

  for (int64_t pos = threadIdx.x; pos < tile_elems; pos += blockDim.x) {
    const int64_t local_row = pos / tile_cols;
    const int64_t local_col = pos % tile_cols;
    const int64_t global_row = row_start + local_row;
    const int64_t global_col = col_start + local_col;
    const float coord_row = row_coords_ptr[global_row];
    const float coord_col = col_coords_ptr[global_col];
    const float grad_out = grad_delta_ptr[global_row * in_features + global_col];
    if (grad_out == 0.0f) {
      continue;
    }

    const float dx = coord_row - mu_row;
    const float dy = coord_col - mu_col;
    float quad = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy);
    quad = fmaxf(quad, 0.0f);
    bool q_grad_active = true;
    if (has_clamp_quad && quad >= static_cast<float>(clamp_quad)) {
      quad = static_cast<float>(clamp_quad);
      q_grad_active = false;
    }

    const float basis_exp = expf(-0.5f * quad);
    float basis = basis_exp;
    if (normalize) {
      const float safe_det = fmaxf(det_cov, static_cast<float>(kCovarianceFloor));
      basis *= 1.0f / (static_cast<float>(kTwoPi) * sqrtf(safe_det));
    }

    acc_amp += grad_out * basis;
    const float common = grad_out * amp * basis;
    if (normalize) {
      const float safe_det = fmaxf(det_cov, static_cast<float>(kCovarianceFloor));
      acc_det += common * (-0.5f / safe_det);
    }
    if (!q_grad_active) {
      continue;
    }

    const float coeff = common * -0.5f;
    const float sym01 = inv01 + inv10;
    const float dq_dmu0 = -(2.0f * inv00 * dx + sym01 * dy);
    const float dq_dmu1 = -(sym01 * dx + 2.0f * inv11 * dy);
    acc_mu0 += coeff * dq_dmu0;
    acc_mu1 += coeff * dq_dmu1;

    const float dxdy = dx * dy;
    acc_inv00 += coeff * dx * dx;
    acc_inv01 += coeff * dxdy;
    acc_inv10 += coeff * dxdy;
    acc_inv11 += coeff * dy * dy;
  }

  sh_amp[threadIdx.x] = acc_amp;
  sh_mu0[threadIdx.x] = acc_mu0;
  sh_mu1[threadIdx.x] = acc_mu1;
  sh_inv00[threadIdx.x] = acc_inv00;
  sh_inv01[threadIdx.x] = acc_inv01;
  sh_inv10[threadIdx.x] = acc_inv10;
  sh_inv11[threadIdx.x] = acc_inv11;
  sh_det[threadIdx.x] = acc_det;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_amp[threadIdx.x] += sh_amp[threadIdx.x + stride];
      sh_mu0[threadIdx.x] += sh_mu0[threadIdx.x + stride];
      sh_mu1[threadIdx.x] += sh_mu1[threadIdx.x + stride];
      sh_inv00[threadIdx.x] += sh_inv00[threadIdx.x + stride];
      sh_inv01[threadIdx.x] += sh_inv01[threadIdx.x + stride];
      sh_inv10[threadIdx.x] += sh_inv10[threadIdx.x + stride];
      sh_inv11[threadIdx.x] += sh_inv11[threadIdx.x + stride];
      sh_det[threadIdx.x] += sh_det[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    partial_amp_ptr[pair_id] = sh_amp[0];
    partial_mu_ptr[pair_id * 2 + 0] = sh_mu0[0];
    partial_mu_ptr[pair_id * 2 + 1] = sh_mu1[0];
    partial_inv_cov_ptr[pair_id * 4 + 0] = sh_inv00[0];
    partial_inv_cov_ptr[pair_id * 4 + 1] = sh_inv01[0];
    partial_inv_cov_ptr[pair_id * 4 + 2] = sh_inv10[0];
    partial_inv_cov_ptr[pair_id * 4 + 3] = sh_inv11[0];
    partial_det_cov_ptr[pair_id] = sh_det[0];
  }
}

__global__ void scatter_reduced_scalar_kernel(
    const int64_t* gaussian_id_ptr,
    const float* reduced_ptr,
    float* out_ptr,
    int64_t num_segments) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_segments) {
    return;
  }
  atomicAdd(out_ptr + gaussian_id_ptr[idx], reduced_ptr[idx]);
}

__global__ void scatter_reduced_vec2_kernel(
    const int64_t* gaussian_id_ptr,
    const float* reduced0_ptr,
    const float* reduced1_ptr,
    float* out_ptr,
    int64_t num_segments) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_segments) {
    return;
  }
  const int64_t gaussian_id = gaussian_id_ptr[idx];
  atomicAdd(out_ptr + gaussian_id * 2 + 0, reduced0_ptr[idx]);
  atomicAdd(out_ptr + gaussian_id * 2 + 1, reduced1_ptr[idx]);
}

__global__ void scatter_reduced_mat2_kernel(
    const int64_t* gaussian_id_ptr,
    const float* reduced00_ptr,
    const float* reduced01_ptr,
    const float* reduced10_ptr,
    const float* reduced11_ptr,
    float* out_ptr,
    int64_t num_segments) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_segments) {
    return;
  }
  const int64_t gaussian_id = gaussian_id_ptr[idx];
  atomicAdd(out_ptr + gaussian_id * 4 + 0, reduced00_ptr[idx]);
  atomicAdd(out_ptr + gaussian_id * 4 + 1, reduced01_ptr[idx]);
  atomicAdd(out_ptr + gaussian_id * 4 + 2, reduced10_ptr[idx]);
  atomicAdd(out_ptr + gaussian_id * 4 + 3, reduced11_ptr[idx]);
}

__global__ void cov_chain_to_chol_raw_kernel(
    const float* chol_raw_ptr,
    const float* inv_cov_ptr,
    const float* det_cov_ptr,
    const float* d_inv_cov_ptr,
    const float* d_det_cov_ptr,
    float* d_chol_raw_ptr,
    int64_t k) {
  const int64_t gaussian_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gaussian_id >= k) {
    return;
  }

  const float raw11 = chol_raw_ptr[gaussian_id * 3 + 0];
  const float raw21 = chol_raw_ptr[gaussian_id * 3 + 1];
  const float raw22 = chol_raw_ptr[gaussian_id * 3 + 2];

  const float l11 = softplus_stable_scalar(raw11) + static_cast<float>(kCholeskyEps);
  const float l21 = raw21;
  const float l22 = softplus_stable_scalar(raw22) + static_cast<float>(kCholeskyEps);

  const float inv00 = inv_cov_ptr[gaussian_id * 4 + 0];
  const float inv01 = inv_cov_ptr[gaussian_id * 4 + 1];
  const float inv10 = inv_cov_ptr[gaussian_id * 4 + 2];
  const float inv11 = inv_cov_ptr[gaussian_id * 4 + 3];
  const float det_cov = det_cov_ptr[gaussian_id];

  const float g00 = d_inv_cov_ptr[gaussian_id * 4 + 0];
  const float g01 = d_inv_cov_ptr[gaussian_id * 4 + 1];
  const float g10 = d_inv_cov_ptr[gaussian_id * 4 + 2];
  const float g11 = d_inv_cov_ptr[gaussian_id * 4 + 3];
  const float gdet = d_det_cov_ptr[gaussian_id];

  const float at00 = inv00;
  const float at01 = inv10;
  const float at10 = inv01;
  const float at11 = inv11;

  const float b00 = g00 * at00 + g01 * at10;
  const float b01 = g00 * at01 + g01 * at11;
  const float b10 = g10 * at00 + g11 * at10;
  const float b11 = g10 * at01 + g11 * at11;

  float gcov00 = -(at00 * b00 + at01 * b10);
  float gcov01 = -(at00 * b01 + at01 * b11);
  float gcov10 = -(at10 * b00 + at11 * b10);
  float gcov11 = -(at10 * b01 + at11 * b11);

  gcov00 += gdet * det_cov * at00;
  gcov01 += gdet * det_cov * at01;
  gcov10 += gdet * det_cov * at10;
  gcov11 += gdet * det_cov * at11;

  const float d_l11 = gcov00 * (2.0f * l11) + gcov01 * l21 + gcov10 * l21;
  const float d_l21 = gcov01 * l11 + gcov10 * l11 + gcov11 * (2.0f * l21);
  const float d_l22 = gcov11 * (2.0f * l22);

  d_chol_raw_ptr[gaussian_id * 3 + 0] = d_l11 * sigmoid_scalar(raw11);
  d_chol_raw_ptr[gaussian_id * 3 + 1] = d_l21;
  d_chol_raw_ptr[gaussian_id * 3 + 2] = d_l22 * sigmoid_scalar(raw22);
}

}  // namespace

std::vector<torch::Tensor> backward_reference_autograd(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config) {
  TORCH_WARN_ONCE(
      "gaussian_field_backward_reference currently executes the reference replay path. "
      "It does not implement the planned custom CUDA backward chain or pair-based "
      "partial gradient reduction.");
  torch::autograd::AutoGradMode enable_grad(true);
  auto mu = saved.mu.detach().clone().set_requires_grad(true);
  auto chol_raw = saved.chol_raw.detach().clone().set_requires_grad(true);
  auto amp = saved.amp.detach().clone().set_requires_grad(true);

  auto geometry = preprocess_geometry(
      saved.row_coords,
      saved.col_coords,
      mu,
      chol_raw,
      config);
  auto binning = make_saved_binning_state(saved);
  auto delta = forward_accumulate_tiles(
      saved.row_coords,
      saved.col_coords,
      mu,
      amp,
      geometry,
      binning,
      config);
  auto grad_flat = grad_delta.contiguous().view_as(delta);

  auto grads = torch::autograd::grad(
      {delta},
      {mu, chol_raw, amp},
      {grad_flat},
      false,
      false,
      false);

  return {grads[0].contiguous(), grads[1].contiguous(), grads[2].contiguous()};
}

std::vector<torch::Tensor> backward_pair_reduce_train(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config) {
  TORCH_WARN_ONCE(
      "gaussian_field_backward is running the cuda_field_train backward path. "
      "This path now uses pair-buffer partial gradients plus grouped reduce-by-key. "
      "It is the active training backend; remaining work is performance tuning "
      "and broader regression coverage.");

  auto guard = at::cuda::CUDAGuard(saved.mu.device());
  auto geometry = preprocess_geometry(
      saved.row_coords,
      saved.col_coords,
      saved.mu,
      saved.chol_raw,
      config);
  auto row_coords_f = saved.row_coords.to(torch::kFloat32).contiguous();
  auto col_coords_f = saved.col_coords.to(torch::kFloat32).contiguous();
  auto mu_f = saved.mu.to(torch::kFloat32).contiguous();
  auto amp_f = saved.amp.to(torch::kFloat32).contiguous().view({-1});
  auto inv_cov_f = geometry.inv_cov.contiguous().view({-1});
  auto det_cov_f = geometry.det_cov.contiguous();
  auto tile_ptr_i64 = saved.tile_ptr.to(torch::kInt64).contiguous();
  auto gaussian_ids_i64 = saved.gaussian_ids_sorted.to(torch::kInt64).contiguous();
  auto grad_delta_f = grad_delta.to(torch::kFloat32).contiguous().view({-1});

  const auto out_features = row_coords_f.size(0);
  const auto in_features = col_coords_f.size(0);
  const auto k = mu_f.size(0);
  const auto num_tile_rows = compute_num_tile_rows(out_features, config.tile_out);
  const auto num_tile_cols = compute_num_tile_cols(in_features, config.tile_in);
  const auto num_tiles = num_tile_rows * num_tile_cols;

  auto d_amp_flat = torch::zeros({k}, amp_f.options());
  auto d_mu = torch::zeros({k, 2}, mu_f.options());
  auto d_inv_cov = torch::zeros({k, 2, 2}, mu_f.options());
  auto d_det_cov = torch::zeros({k}, mu_f.options());
  auto d_chol_raw = torch::zeros({k, 3}, mu_f.options());
  auto d_mu_flat = d_mu.view({-1});
  auto d_inv_cov_flat = d_inv_cov.view({-1});
  auto chol_raw_f = saved.chol_raw.to(torch::kFloat32).contiguous();
  const auto num_pairs = gaussian_ids_i64.numel();
  const auto chunk_plan = make_backward_chunk_plan(tile_ptr_i64, num_pairs);
  if (chunk_plan.use_chunking) {
    TORCH_WARN_ONCE(
        "gaussian_field_backward enabled chunked pair-buffer protection for "
        "cuda_field_train. The custom CUDA backward path remains active, but "
        "pair gradients are reduced in chunks to stay within capacity limits.");
  }

  for (int64_t pair_start = 0; pair_start < num_pairs; pair_start += chunk_plan.chunk_pairs) {
    const auto pair_count = std::min<int64_t>(chunk_plan.chunk_pairs, num_pairs - pair_start);
    auto gaussian_ids_chunk = gaussian_ids_i64.narrow(0, pair_start, pair_count).contiguous();
    auto pair_buffer = allocate_pair_gradient_buffer(gaussian_ids_chunk, grad_delta_f);

    const int fill_blocks = static_cast<int>((pair_count + kBackwardThreads - 1) / kBackwardThreads);
    fill_pair_tile_ids_range_kernel<<<fill_blocks, kBackwardThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
        tile_ptr_i64.data_ptr<int64_t>(),
        pair_buffer.pair_tile_id.data_ptr<int64_t>(),
        num_tiles,
        pair_start,
        pair_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    backward_pair_partial_kernel<<<
        static_cast<unsigned int>(pair_count),
        kBackwardThreads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
        row_coords_f.data_ptr<float>(),
        col_coords_f.data_ptr<float>(),
        mu_f.data_ptr<float>(),
        amp_f.data_ptr<float>(),
        inv_cov_f.data_ptr<float>(),
        det_cov_f.data_ptr<float>(),
        pair_buffer.pair_tile_id.data_ptr<int64_t>(),
        pair_buffer.pair_gaussian_id.data_ptr<int64_t>(),
        grad_delta_f.data_ptr<float>(),
        pair_buffer.partial_amp.data_ptr<float>(),
        pair_buffer.partial_mu.view({-1}).data_ptr<float>(),
        pair_buffer.partial_inv_cov.view({-1}).data_ptr<float>(),
        pair_buffer.partial_det_cov.data_ptr<float>(),
        out_features,
        in_features,
        config.tile_out,
        config.tile_in,
        num_tile_cols,
        config.normalize,
        config.has_clamp_quad,
        config.clamp_quad,
        pair_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto pair_order = torch::arange(pair_count, tile_ptr_i64.options());
    auto [sorted_gaussian_ids, sorted_pair_order, _sort_workspace] =
        cub_sort_pairs_int64(pair_buffer.pair_gaussian_id, pair_order);
    auto partial_amp_sorted = pair_buffer.partial_amp.index_select(0, sorted_pair_order);
    auto partial_mu_sorted = pair_buffer.partial_mu.index_select(0, sorted_pair_order);
    auto partial_inv_cov_sorted = pair_buffer.partial_inv_cov.index_select(0, sorted_pair_order);
    auto partial_det_cov_sorted = pair_buffer.partial_det_cov.index_select(0, sorted_pair_order);
    auto partial_mu0_sorted = partial_mu_sorted.select(1, 0).contiguous();
    auto partial_mu1_sorted = partial_mu_sorted.select(1, 1).contiguous();
    auto partial_inv00_sorted = partial_inv_cov_sorted.select(1, 0).select(1, 0).contiguous();
    auto partial_inv01_sorted = partial_inv_cov_sorted.select(1, 0).select(1, 1).contiguous();
    auto partial_inv10_sorted = partial_inv_cov_sorted.select(1, 1).select(1, 0).contiguous();
    auto partial_inv11_sorted = partial_inv_cov_sorted.select(1, 1).select(1, 1).contiguous();

    auto [amp_keys, amp_values] =
        cub_reduce_sum_by_key_int64_float(sorted_gaussian_ids, partial_amp_sorted.contiguous());
    auto [mu0_keys, mu0_values] =
        cub_reduce_sum_by_key_int64_float(sorted_gaussian_ids, partial_mu0_sorted);
    auto [mu1_keys, mu1_values] =
        cub_reduce_sum_by_key_int64_float(sorted_gaussian_ids, partial_mu1_sorted);
    auto [inv00_keys, inv00_values] =
        cub_reduce_sum_by_key_int64_float(sorted_gaussian_ids, partial_inv00_sorted);
    auto [inv01_keys, inv01_values] =
        cub_reduce_sum_by_key_int64_float(sorted_gaussian_ids, partial_inv01_sorted);
    auto [inv10_keys, inv10_values] =
        cub_reduce_sum_by_key_int64_float(sorted_gaussian_ids, partial_inv10_sorted);
    auto [inv11_keys, inv11_values] =
        cub_reduce_sum_by_key_int64_float(sorted_gaussian_ids, partial_inv11_sorted);
    auto [det_keys, det_values] =
        cub_reduce_sum_by_key_int64_float(sorted_gaussian_ids, partial_det_cov_sorted.contiguous());

    auto unique_mask = torch::ones({pair_count}, sorted_gaussian_ids.options().dtype(torch::kBool));
    if (pair_count > 1) {
      unique_mask.slice(0, 1).copy_(sorted_gaussian_ids.slice(0, 1) != sorted_gaussian_ids.slice(0, 0, -1));
    }
    const auto num_segments = static_cast<int64_t>(unique_mask.sum().item<int64_t>());
    auto segment_ids = sorted_gaussian_ids.masked_select(unique_mask);

    const int scatter_blocks = static_cast<int>((num_segments + kBackwardThreads - 1) / kBackwardThreads);
    scatter_reduced_scalar_kernel<<<scatter_blocks, kBackwardThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
        segment_ids.data_ptr<int64_t>(),
        amp_values.data_ptr<float>(),
        d_amp_flat.data_ptr<float>(),
        num_segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scatter_reduced_vec2_kernel<<<scatter_blocks, kBackwardThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
        segment_ids.data_ptr<int64_t>(),
        mu0_values.data_ptr<float>(),
        mu1_values.data_ptr<float>(),
        d_mu_flat.data_ptr<float>(),
        num_segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scatter_reduced_mat2_kernel<<<scatter_blocks, kBackwardThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
        segment_ids.data_ptr<int64_t>(),
        inv00_values.data_ptr<float>(),
        inv01_values.data_ptr<float>(),
        inv10_values.data_ptr<float>(),
        inv11_values.data_ptr<float>(),
        d_inv_cov_flat.data_ptr<float>(),
        num_segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scatter_reduced_scalar_kernel<<<scatter_blocks, kBackwardThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
        segment_ids.data_ptr<int64_t>(),
        det_values.data_ptr<float>(),
        d_det_cov.data_ptr<float>(),
        num_segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int chain_blocks = static_cast<int>((k + kBackwardThreads - 1) / kBackwardThreads);
  cov_chain_to_chol_raw_kernel<<<chain_blocks, kBackwardThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
      chol_raw_f.data_ptr<float>(),
      inv_cov_f.data_ptr<float>(),
      det_cov_f.data_ptr<float>(),
      d_inv_cov_flat.data_ptr<float>(),
      d_det_cov.data_ptr<float>(),
      d_chol_raw.data_ptr<float>(),
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {d_mu.contiguous(), d_chol_raw.contiguous(), d_amp_flat.view({k, 1}).contiguous()};
}

}  // namespace gaussian_peft::cuda_field
