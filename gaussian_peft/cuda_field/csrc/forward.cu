#include "field.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>

namespace gaussian_peft::cuda_field {

namespace {

constexpr double kCholeskyEps = 1e-5;
constexpr double kCovarianceFloor = 1e-8;
constexpr double kTwoPi = 6.28318530717958647692;
constexpr int kPreprocessThreads = 256;
constexpr int kBinningThreads = 256;
constexpr int kAccumulateThreadsMax = 256;
constexpr int kAccumulateGaussianChunk = 32;
constexpr int64_t kAccumulateCandidatesPerWorkBlock = 64;
constexpr int64_t kAccumulateMaxBlocksPerTile = 4;

__device__ inline float softplus_stable_scalar(float x) {
  return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}

int64_t compute_num_tile_rows(int64_t out_features, int64_t tile_out) {
  return (out_features + tile_out - 1) / tile_out;
}

int64_t compute_num_tile_cols(int64_t in_features, int64_t tile_in) {
  return (in_features + tile_in - 1) / tile_in;
}

int choose_accumulate_threads(int64_t tile_out, int64_t tile_in) {
  const int64_t tile_elems = tile_out * tile_in;
  if (tile_elems <= 64) {
    return 64;
  }
  if (tile_elems <= 128) {
    return 128;
  }
  return kAccumulateThreadsMax;
}

int64_t read_total_pairs_from_scan(
    const torch::Tensor& point_offsets_i64,
    const torch::Tensor& tiles_touched_i64) {
  const auto num_gaussians = tiles_touched_i64.size(0);
  if (num_gaussians == 0) {
    return 0;
  }

  int64_t last_offset = 0;
  int64_t last_touched = 0;
  auto stream = at::cuda::getDefaultCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      &last_offset,
      point_offsets_i64.data_ptr<int64_t>() + (num_gaussians - 1),
      sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      &last_touched,
      tiles_touched_i64.data_ptr<int64_t>() + (num_gaussians - 1),
      sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return last_offset + last_touched;
}

std::pair<torch::Tensor, torch::Tensor> cub_exclusive_sum_int64(const torch::Tensor& input_i64) {
  auto output = torch::empty_like(input_i64);
  size_t temp_storage_bytes = 0;
  auto stream = at::cuda::getDefaultCUDAStream();
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      nullptr,
      temp_storage_bytes,
      input_i64.data_ptr<int64_t>(),
      output.data_ptr<int64_t>(),
      input_i64.numel(),
      stream));
  auto temp_storage = torch::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      input_i64.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      temp_storage.data_ptr(),
      temp_storage_bytes,
      input_i64.data_ptr<int64_t>(),
      output.data_ptr<int64_t>(),
      input_i64.numel(),
      stream));
  return {output, temp_storage};
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

__global__ void preprocess_geometry_kernel(
    const float* mu_ptr,
    const float* chol_raw_ptr,
    float* inv_cov_ptr,
    float* det_cov_ptr,
    int64_t* tile_r0_ptr,
    int64_t* tile_r1_ptr,
    int64_t* tile_c0_ptr,
    int64_t* tile_c1_ptr,
    int64_t* tiles_touched_ptr,
    int64_t out_features,
    int64_t in_features,
    int64_t tile_out,
    int64_t tile_in,
    int64_t num_tile_rows,
    int64_t num_tile_cols,
    double sigma_multiplier,
    int64_t k) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= k) {
    return;
  }

  const float mu_row = mu_ptr[idx * 2 + 0];
  const float mu_col = mu_ptr[idx * 2 + 1];
  const float raw11 = chol_raw_ptr[idx * 3 + 0];
  const float raw21 = chol_raw_ptr[idx * 3 + 1];
  const float raw22 = chol_raw_ptr[idx * 3 + 2];

  const float l11 = softplus_stable_scalar(raw11) + static_cast<float>(kCholeskyEps);
  const float l21 = raw21;
  const float l22 = softplus_stable_scalar(raw22) + static_cast<float>(kCholeskyEps);

  const float cov00 = l11 * l11;
  const float cov01 = l11 * l21;
  const float cov10 = cov01;
  const float cov11 = l21 * l21 + l22 * l22;
  const float det_cov = fmaxf(cov00 * cov11 - cov01 * cov10, static_cast<float>(kCovarianceFloor));

  inv_cov_ptr[idx * 4 + 0] = cov11 / det_cov;
  inv_cov_ptr[idx * 4 + 1] = -cov01 / det_cov;
  inv_cov_ptr[idx * 4 + 2] = -cov10 / det_cov;
  inv_cov_ptr[idx * 4 + 3] = cov00 / det_cov;
  det_cov_ptr[idx] = det_cov;

  const float std_row = sqrtf(fmaxf(cov00, static_cast<float>(kCovarianceFloor)));
  const float std_col = sqrtf(fmaxf(cov11, static_cast<float>(kCovarianceFloor)));

  const float row_scale = 0.5f * static_cast<float>(max(out_features - 1, static_cast<int64_t>(1)));
  const float col_scale = 0.5f * static_cast<float>(max(in_features - 1, static_cast<int64_t>(1)));
  const float row_center = (mu_row + 1.0f) * row_scale;
  const float col_center = (mu_col + 1.0f) * col_scale;
  const float row_radius = static_cast<float>(sigma_multiplier) * std_row * row_scale;
  const float col_radius = static_cast<float>(sigma_multiplier) * std_col * col_scale;

  int64_t row_min = static_cast<int64_t>(floorf(row_center - row_radius));
  int64_t row_max = static_cast<int64_t>(ceilf(row_center + row_radius));
  int64_t col_min = static_cast<int64_t>(floorf(col_center - col_radius));
  int64_t col_max = static_cast<int64_t>(ceilf(col_center + col_radius));

  row_min = max(static_cast<int64_t>(0), min(row_min, out_features - 1));
  row_max = max(static_cast<int64_t>(0), min(row_max, out_features - 1));
  col_min = max(static_cast<int64_t>(0), min(col_min, in_features - 1));
  col_max = max(static_cast<int64_t>(0), min(col_max, in_features - 1));

  const int64_t tile_r0 = max(static_cast<int64_t>(0), min(row_min / tile_out, num_tile_rows - 1));
  const int64_t tile_r1 = max(static_cast<int64_t>(0), min(row_max / tile_out, num_tile_rows - 1));
  const int64_t tile_c0 = max(static_cast<int64_t>(0), min(col_min / tile_in, num_tile_cols - 1));
  const int64_t tile_c1 = max(static_cast<int64_t>(0), min(col_max / tile_in, num_tile_cols - 1));

  tile_r0_ptr[idx] = tile_r0;
  tile_r1_ptr[idx] = tile_r1;
  tile_c0_ptr[idx] = tile_c0;
  tile_c1_ptr[idx] = tile_c1;
  tiles_touched_ptr[idx] = (tile_r1 - tile_r0 + 1) * (tile_c1 - tile_c0 + 1);
}

__global__ void emit_tile_pairs_kernel(
    const int64_t* point_offsets_ptr,
    const int64_t* tile_r0_ptr,
    const int64_t* tile_r1_ptr,
    const int64_t* tile_c0_ptr,
    const int64_t* tile_c1_ptr,
    int64_t* tile_keys_unsorted_ptr,
    int64_t* gaussian_ids_unsorted_ptr,
    int64_t num_tile_cols,
    int64_t k) {
  const int64_t gaussian_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gaussian_id >= k) {
    return;
  }

  const int64_t base = point_offsets_ptr[gaussian_id];
  const int64_t tile_r0 = tile_r0_ptr[gaussian_id];
  const int64_t tile_r1 = tile_r1_ptr[gaussian_id];
  const int64_t tile_c0 = tile_c0_ptr[gaussian_id];
  const int64_t tile_c1 = tile_c1_ptr[gaussian_id];

  int64_t local = 0;
  for (int64_t tile_row = tile_r0; tile_row <= tile_r1; ++tile_row) {
    for (int64_t tile_col = tile_c0; tile_col <= tile_c1; ++tile_col) {
      const int64_t pair_idx = base + local;
      tile_keys_unsorted_ptr[pair_idx] = tile_row * num_tile_cols + tile_col;
      gaussian_ids_unsorted_ptr[pair_idx] = gaussian_id;
      ++local;
    }
  }
}

__global__ void accumulate_tile_counts_kernel(
    const int64_t* tile_keys_ptr,
    int64_t* counts_ptr,
    int64_t num_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= num_pairs) {
    return;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(counts_ptr + tile_keys_ptr[pair_idx]), 1ULL);
}

__global__ void tile_accumulate_kernel(
    const float* row_coords_ptr,
    const float* col_coords_ptr,
    const float* mu_ptr,
    const float* amp_ptr,
    const float* inv_cov_ptr,
    const float* det_cov_ptr,
    const int64_t* tile_ptr_ptr,
    const int64_t* per_tile_counts_ptr,
    const int64_t* gaussian_ids_ptr,
    float* delta_ptr,
    int64_t out_features,
    int64_t in_features,
    int64_t tile_out,
    int64_t tile_in,
    int64_t num_tile_cols,
    bool normalize,
    bool has_clamp_quad,
    double clamp_quad,
    int64_t num_tiles,
    int64_t max_blocks_per_tile) {
  __shared__ float sh_mu_row[kAccumulateGaussianChunk];
  __shared__ float sh_mu_col[kAccumulateGaussianChunk];
  __shared__ float sh_amp[kAccumulateGaussianChunk];
  __shared__ float sh_inv00[kAccumulateGaussianChunk];
  __shared__ float sh_inv01[kAccumulateGaussianChunk];
  __shared__ float sh_inv10[kAccumulateGaussianChunk];
  __shared__ float sh_inv11[kAccumulateGaussianChunk];
  __shared__ float sh_det[kAccumulateGaussianChunk];

  const int64_t work_block_id = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_id = work_block_id / max_blocks_per_tile;
  const int64_t tile_worker_id = work_block_id % max_blocks_per_tile;
  if (tile_id >= num_tiles) {
    return;
  }

  const int64_t start = tile_ptr_ptr[tile_id];
  const int64_t end = tile_ptr_ptr[tile_id + 1];
  if (end <= start) {
    return;
  }
  const int64_t candidate_count = per_tile_counts_ptr[tile_id];
  const int64_t active_blocks = min(
      max_blocks_per_tile,
      max(
          static_cast<int64_t>(1),
          (candidate_count + kAccumulateCandidatesPerWorkBlock - 1) / kAccumulateCandidatesPerWorkBlock));
  if (tile_worker_id >= active_blocks) {
    return;
  }

  const int64_t tile_row = tile_id / num_tile_cols;
  const int64_t tile_col = tile_id % num_tile_cols;
  const int64_t row_start = tile_row * tile_out;
  const int64_t row_end = min(row_start + tile_out, out_features);
  const int64_t col_start = tile_col * tile_in;
  const int64_t col_end = min(col_start + tile_in, in_features);
  const int64_t tile_rows = row_end - row_start;
  const int64_t tile_cols = col_end - col_start;
  const int64_t tile_elems = tile_rows * tile_cols;

  for (int64_t pos = tile_worker_id * blockDim.x + threadIdx.x;
       pos < tile_elems;
       pos += blockDim.x * active_blocks) {
    const int64_t local_row = pos / tile_cols;
    const int64_t local_col = pos % tile_cols;
    const int64_t global_row = row_start + local_row;
    const int64_t global_col = col_start + local_col;
    const float coord_row = row_coords_ptr[global_row];
    const float coord_col = col_coords_ptr[global_col];

    float value = 0.0f;
    for (int64_t chunk_start = start; chunk_start < end; chunk_start += kAccumulateGaussianChunk) {
      const int64_t chunk_count = min(
          static_cast<int64_t>(kAccumulateGaussianChunk),
          end - chunk_start);

      if (threadIdx.x < chunk_count) {
        const int64_t gaussian_id = gaussian_ids_ptr[chunk_start + threadIdx.x];
        sh_mu_row[threadIdx.x] = mu_ptr[gaussian_id * 2 + 0];
        sh_mu_col[threadIdx.x] = mu_ptr[gaussian_id * 2 + 1];
        sh_amp[threadIdx.x] = amp_ptr[gaussian_id];
        sh_inv00[threadIdx.x] = inv_cov_ptr[gaussian_id * 4 + 0];
        sh_inv01[threadIdx.x] = inv_cov_ptr[gaussian_id * 4 + 1];
        sh_inv10[threadIdx.x] = inv_cov_ptr[gaussian_id * 4 + 2];
        sh_inv11[threadIdx.x] = inv_cov_ptr[gaussian_id * 4 + 3];
        sh_det[threadIdx.x] = det_cov_ptr[gaussian_id];
      }
      __syncthreads();

      for (int64_t local_idx = 0; local_idx < chunk_count; ++local_idx) {
        const float dx = coord_row - sh_mu_row[local_idx];
        const float dy = coord_col - sh_mu_col[local_idx];

        float quad = dx * (sh_inv00[local_idx] * dx + sh_inv01[local_idx] * dy) +
                     dy * (sh_inv10[local_idx] * dx + sh_inv11[local_idx] * dy);
        quad = fmaxf(quad, 0.0f);
        if (has_clamp_quad) {
          quad = fminf(quad, static_cast<float>(clamp_quad));
        }

        float basis = expf(-0.5f * quad);
        if (normalize) {
          const float det_cov = fmaxf(sh_det[local_idx], static_cast<float>(kCovarianceFloor));
          basis *= 1.0f / (static_cast<float>(kTwoPi) * sqrtf(det_cov));
        }
        value += sh_amp[local_idx] * basis;
      }
      __syncthreads();
    }

    delta_ptr[global_row * in_features + global_col] = value;
  }
}

}  // namespace

GeometryState preprocess_geometry(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& chol_raw,
    const ForwardConfig& config) {
  auto float_opts = mu.options().dtype(torch::kFloat32);
  auto index_opts = mu.options().dtype(torch::kInt64);
  const auto out_features = row_coords.size(0);
  const auto in_features = col_coords.size(0);
  const auto num_tile_rows = compute_num_tile_rows(out_features, config.tile_out);
  const auto num_tile_cols = compute_num_tile_cols(in_features, config.tile_in);
  const auto k = mu.size(0);
  auto guard = at::cuda::CUDAGuard(mu.device());
  auto mu_f = mu.to(torch::kFloat32).contiguous();
  auto chol_raw_f = chol_raw.to(torch::kFloat32).contiguous();
  auto inv_cov = torch::empty({k, 2, 2}, float_opts);
  auto det_cov = torch::empty({k}, float_opts);
  auto tile_r0 = torch::empty({k}, index_opts);
  auto tile_r1 = torch::empty({k}, index_opts);
  auto tile_c0 = torch::empty({k}, index_opts);
  auto tile_c1 = torch::empty({k}, index_opts);
  auto tiles_touched = torch::empty({k}, index_opts);

  const int blocks = static_cast<int>((k + kPreprocessThreads - 1) / kPreprocessThreads);
  preprocess_geometry_kernel<<<blocks, kPreprocessThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
      mu_f.data_ptr<float>(),
      chol_raw_f.data_ptr<float>(),
      inv_cov.data_ptr<float>(),
      det_cov.data_ptr<float>(),
      tile_r0.data_ptr<int64_t>(),
      tile_r1.data_ptr<int64_t>(),
      tile_c0.data_ptr<int64_t>(),
      tile_c1.data_ptr<int64_t>(),
      tiles_touched.data_ptr<int64_t>(),
      out_features,
      in_features,
      config.tile_out,
      config.tile_in,
      num_tile_rows,
      num_tile_cols,
      config.sigma_multiplier,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return GeometryState{
      inv_cov,
      det_cov,
      tile_r0,
      tile_r1,
      tile_c0,
      tile_c1,
      tiles_touched,
  };
}

BinningState build_binning_state(
    const GeometryState& geometry,
    int64_t num_tile_rows,
    int64_t num_tile_cols) {
  auto index_opts = geometry.tile_r0.options().dtype(torch::kInt64);
  const auto device = geometry.tile_r0.device();
  const auto num_gaussians = geometry.tile_r0.size(0);
  const auto num_tiles = num_tile_rows * num_tile_cols;
  auto guard = at::cuda::CUDAGuard(device);

  auto tiles_touched_i64 = geometry.tiles_touched.to(torch::kInt64).contiguous();
  auto [point_offsets_i64, point_offsets_scan_workspace] = cub_exclusive_sum_int64(tiles_touched_i64);
  auto tile_r0_i64 = geometry.tile_r0.to(torch::kInt64).contiguous();
  auto tile_r1_i64 = geometry.tile_r1.to(torch::kInt64).contiguous();
  auto tile_c0_i64 = geometry.tile_c0.to(torch::kInt64).contiguous();
  auto tile_c1_i64 = geometry.tile_c1.to(torch::kInt64).contiguous();
  const auto total_pairs = read_total_pairs_from_scan(point_offsets_i64, tiles_touched_i64);

  if (total_pairs == 0) {
    auto empty = torch::empty({0}, index_opts.device(device));
    auto tile_ptr = torch::zeros({num_tiles + 1}, index_opts.device(device));
    auto per_tile_counts = torch::zeros({num_tiles}, index_opts.device(device));
    return BinningState{
        point_offsets_i64,
        empty,
        empty,
        empty,
        per_tile_counts,
        empty,
        empty,
        empty,
        tile_ptr,
        0,
        0,
    };
  }

  auto tile_keys_unsorted = torch::empty({total_pairs}, index_opts.device(device));
  auto gaussian_ids_unsorted = torch::empty({total_pairs}, index_opts.device(device));
  const int emit_blocks =
      static_cast<int>((num_gaussians + kBinningThreads - 1) / kBinningThreads);
  emit_tile_pairs_kernel<<<emit_blocks, kBinningThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
      point_offsets_i64.data_ptr<int64_t>(),
      tile_r0_i64.data_ptr<int64_t>(),
      tile_r1_i64.data_ptr<int64_t>(),
      tile_c0_i64.data_ptr<int64_t>(),
      tile_c1_i64.data_ptr<int64_t>(),
      tile_keys_unsorted.data_ptr<int64_t>(),
      gaussian_ids_unsorted.data_ptr<int64_t>(),
      num_tile_cols,
      num_gaussians);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto [tile_keys_sorted, gaussian_ids_sorted, sort_workspace] =
      cub_sort_pairs_int64(tile_keys_unsorted, gaussian_ids_unsorted);
  auto counts = torch::zeros({num_tiles}, index_opts.device(device));
  const int range_blocks =
      static_cast<int>((total_pairs + kBinningThreads - 1) / kBinningThreads);
  accumulate_tile_counts_kernel<<<range_blocks, kBinningThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
      tile_keys_sorted.data_ptr<int64_t>(),
      counts.data_ptr<int64_t>(),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto tile_ptr_counts = torch::zeros({num_tiles + 1}, index_opts.device(device));
  tile_ptr_counts.slice(0, 0, num_tiles).copy_(counts);
  auto [tile_ptr, tile_ptr_scan_workspace] = cub_exclusive_sum_int64(tile_ptr_counts);
  auto scan_workspace = tile_ptr_scan_workspace.numel() >= point_offsets_scan_workspace.numel()
      ? tile_ptr_scan_workspace
      : point_offsets_scan_workspace;

  return BinningState{
      point_offsets_i64,
      tile_keys_unsorted,
      tile_keys_sorted,
      gaussian_ids_unsorted,
      counts,
      scan_workspace,
      sort_workspace,
      gaussian_ids_sorted,
      tile_ptr,
      total_pairs,
      total_pairs,
  };
}

torch::Tensor forward_accumulate_tiles(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& amp,
    const GeometryState& geometry,
    const BinningState& binning,
    const ForwardConfig& config) {
  const auto out_features = row_coords.size(0);
  const auto in_features = col_coords.size(0);
  const auto num_tile_rows = compute_num_tile_rows(out_features, config.tile_out);
  const auto num_tile_cols = compute_num_tile_cols(in_features, config.tile_in);
  const auto num_tiles = num_tile_rows * num_tile_cols;
  const int accumulate_threads = choose_accumulate_threads(config.tile_out, config.tile_in);
  const int64_t launched_tile_work_blocks = num_tiles * kAccumulateMaxBlocksPerTile;
  auto guard = at::cuda::CUDAGuard(row_coords.device());

  auto delta = torch::zeros({out_features * in_features}, row_coords.options().dtype(torch::kFloat32));
  auto row_coords_f = row_coords.to(torch::kFloat32).contiguous();
  auto col_coords_f = col_coords.to(torch::kFloat32).contiguous();
  auto mu_f = mu.to(torch::kFloat32).contiguous();
  auto amp_f = amp.to(torch::kFloat32).contiguous().view({-1});
  auto inv_cov_f = geometry.inv_cov.contiguous().view({-1});
  auto det_cov_f = geometry.det_cov.contiguous();
  auto tile_ptr_i64 = binning.tile_ptr.contiguous().to(torch::kInt64);
  auto per_tile_counts_i64 = binning.per_tile_counts.contiguous().to(torch::kInt64);
  auto gaussian_ids_i64 = binning.gaussian_ids_sorted.contiguous().to(torch::kInt64);

  tile_accumulate_kernel<<<
      static_cast<unsigned int>(launched_tile_work_blocks),
      accumulate_threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
      row_coords_f.data_ptr<float>(),
      col_coords_f.data_ptr<float>(),
      mu_f.data_ptr<float>(),
      amp_f.data_ptr<float>(),
      inv_cov_f.data_ptr<float>(),
      det_cov_f.data_ptr<float>(),
      tile_ptr_i64.data_ptr<int64_t>(),
      per_tile_counts_i64.data_ptr<int64_t>(),
      gaussian_ids_i64.data_ptr<int64_t>(),
      delta.data_ptr<float>(),
      out_features,
      in_features,
      config.tile_out,
      config.tile_in,
      num_tile_cols,
      config.normalize,
      config.has_clamp_quad,
      config.clamp_quad,
      num_tiles,
      kAccumulateMaxBlocksPerTile);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return delta.contiguous();
}

}  // namespace gaussian_peft::cuda_field
