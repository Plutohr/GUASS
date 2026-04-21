#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace gaussian_peft::cuda_field {

struct ForwardConfig {
  int64_t tile_out;
  int64_t tile_in;
  double sigma_multiplier;
  bool normalize;
  bool has_clamp_quad;
  double clamp_quad;
};

struct SavedTensorBundle {
  torch::Tensor row_coords;           // [out_features], float32, CUDA, contiguous
  torch::Tensor col_coords;           // [in_features], float32, CUDA, contiguous
  torch::Tensor mu;                   // [K, 2], float32, CUDA, contiguous
  torch::Tensor chol_raw;             // [K, 3], float32, CUDA, contiguous
  torch::Tensor amp;                  // [K, 1], float32, CUDA, contiguous
  torch::Tensor tile_ptr;             // [num_tiles + 1], int32/int64, CUDA, contiguous
  torch::Tensor gaussian_ids_sorted;  // [num_pairs], int32/int64, CUDA, contiguous
};

struct GeometryState {
  torch::Tensor inv_cov;      // [K, 2, 2], optional rebuild target
  torch::Tensor det_cov;      // [K], optional rebuild target
  torch::Tensor tile_r0;      // [K], optional rebuild target
  torch::Tensor tile_r1;      // [K], optional rebuild target
  torch::Tensor tile_c0;      // [K], optional rebuild target
  torch::Tensor tile_c1;      // [K], optional rebuild target
  torch::Tensor tiles_touched;// [K], optional rebuild target
};

struct BinningState {
  torch::Tensor point_offsets;        // [K], forward workspace only
  torch::Tensor tile_keys_unsorted;   // [num_pairs], forward workspace only
  torch::Tensor tile_keys_sorted;     // [num_pairs], forward workspace only
  torch::Tensor gaussian_ids_unsorted;// [num_pairs], forward workspace only
  torch::Tensor per_tile_counts;      // [num_tiles], workspace + launch planning
  torch::Tensor scan_workspace;       // [bytes], forward workspace only
  torch::Tensor sort_workspace;       // [bytes], forward workspace only
  torch::Tensor gaussian_ids_sorted;  // [num_pairs], persistent output
  torch::Tensor tile_ptr;             // [num_tiles + 1], persistent output
  int64_t total_pairs_host;           // orchestration-owned host scalar
  int64_t pair_capacity;              // current allocated pair capacity
};

struct TileState {
  int64_t num_tile_rows;
  int64_t num_tile_cols;
  int64_t num_tiles;
  int64_t max_blocks_per_tile;
  int64_t launched_tile_work_blocks;
};

struct ForwardOrchestrationState {
  GeometryState geometry;
  BinningState binning;
  TileState tile;
  SavedTensorBundle saved;
};

struct BackwardIntermediate {
  torch::Tensor d_amp;       // [K, 1]
  torch::Tensor d_mu;        // [K, 2]
  torch::Tensor d_inv_cov;   // [K, 2, 2]
  torch::Tensor d_det_cov;   // [K]
  torch::Tensor d_cov;       // [K, 2, 2]
  torch::Tensor d_chol;      // [K, 2, 2]
  torch::Tensor d_chol_raw;  // [K, 3]
};

struct PairGradientBuffer {
  torch::Tensor pair_gaussian_id;   // [num_pairs]
  torch::Tensor pair_tile_id;       // [num_pairs]
  torch::Tensor partial_amp;        // [num_pairs]
  torch::Tensor partial_mu;         // [num_pairs, 2]
  torch::Tensor partial_inv_cov;    // [num_pairs, 2, 2]
  torch::Tensor partial_det_cov;    // [num_pairs]
};

GeometryState preprocess_geometry(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& chol_raw,
    const ForwardConfig& config);

BinningState build_binning_state(
    const GeometryState& geometry,
    int64_t num_tile_rows,
    int64_t num_tile_cols);

torch::Tensor forward_accumulate_tiles(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& amp,
    const GeometryState& geometry,
    const BinningState& binning,
    const ForwardConfig& config);

std::vector<torch::Tensor> backward_reference_autograd(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config);

std::vector<torch::Tensor> backward_pair_reduce_train(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config);

ForwardOrchestrationState build_forward_orchestration_state(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    const ForwardConfig& config);

// Frozen cross-language forward return protocol:
// (delta, tile_ptr, gaussian_ids_sorted)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gaussian_field_forward(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    const ForwardConfig& config);

// Frozen backward return protocol:
// (d_mu, d_chol_raw, d_amp)
std::vector<torch::Tensor> gaussian_field_backward(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config);

std::vector<torch::Tensor> gaussian_field_backward_reference(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config);

std::vector<torch::Tensor> gaussian_field_backward_train(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config);

torch::Tensor cell_average_diag_v1_forward(
    const torch::Tensor& mu_raw,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    int64_t out_features,
    int64_t in_features,
    double sigma_min);

std::vector<torch::Tensor> cell_average_diag_v1_backward(
    const torch::Tensor& grad_delta,
    const torch::Tensor& mu_raw,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    int64_t out_features,
    int64_t in_features,
    double sigma_min);

// Validation-only helpers.
void validate_forward_inputs(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp);

void validate_backward_inputs(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved);

void validate_saved_state_contract(const SavedTensorBundle& saved);

}  // namespace gaussian_peft::cuda_field
