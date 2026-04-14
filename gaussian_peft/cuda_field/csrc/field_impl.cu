#include "field.h"

#include <stdexcept>

#include "common.h"

namespace gaussian_peft::cuda_field {

namespace {
int64_t read_last_index_value(const torch::Tensor& tensor) {
  return tensor.index({-1}).item<int64_t>();
}

constexpr int64_t kAccumulateMaxBlocksPerTile = 4;

TileState build_tile_state(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const ForwardConfig& config) {
  const auto out_features = row_coords.size(0);
  const auto in_features = col_coords.size(0);
  const auto num_tile_rows = (out_features + config.tile_out - 1) / config.tile_out;
  const auto num_tile_cols = (in_features + config.tile_in - 1) / config.tile_in;
  return TileState{
      num_tile_rows,
      num_tile_cols,
      num_tile_rows * num_tile_cols,
      kAccumulateMaxBlocksPerTile,
      num_tile_rows * num_tile_cols * kAccumulateMaxBlocksPerTile,
  };
}

SavedTensorBundle pack_saved_tensor_bundle(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    const BinningState& binning) {
  return SavedTensorBundle{
      row_coords,
      col_coords,
      mu,
      chol_raw,
      amp,
      binning.tile_ptr,
      binning.gaussian_ids_sorted,
  };
}
}  // namespace

void validate_forward_inputs(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp) {
  check_float32_tensor(row_coords, "row_coords");
  check_float32_tensor(col_coords, "col_coords");
  check_float32_tensor(mu, "mu");
  check_float32_tensor(chol_raw, "chol_raw");
  check_float32_tensor(amp, "amp");

  TORCH_CHECK(row_coords.dim() == 1, "row_coords must have shape [out_features]");
  TORCH_CHECK(col_coords.dim() == 1, "col_coords must have shape [in_features]");
  TORCH_CHECK(mu.dim() == 2 && mu.size(1) == 2, "mu must have shape [K, 2]");
  TORCH_CHECK(chol_raw.dim() == 2 && chol_raw.size(1) == 3, "chol_raw must have shape [K, 3]");
  TORCH_CHECK(amp.dim() == 2 && amp.size(1) == 1, "amp must have shape [K, 1]");
  TORCH_CHECK(mu.size(0) == chol_raw.size(0), "mu and chol_raw must have the same K");
  TORCH_CHECK(mu.size(0) == amp.size(0), "mu and amp must have the same K");
}

void validate_backward_inputs(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved) {
  check_float32_tensor(grad_delta, "grad_delta");
  check_float32_tensor(saved.row_coords, "saved.row_coords");
  check_float32_tensor(saved.col_coords, "saved.col_coords");
  check_float32_tensor(saved.mu, "saved.mu");
  check_float32_tensor(saved.chol_raw, "saved.chol_raw");
  check_float32_tensor(saved.amp, "saved.amp");
  check_index_tensor(saved.tile_ptr, "saved.tile_ptr");
  check_index_tensor(saved.gaussian_ids_sorted, "saved.gaussian_ids_sorted");
}

void validate_saved_state_contract(const SavedTensorBundle& saved) {
  TORCH_CHECK(saved.tile_ptr.dim() == 1, "saved.tile_ptr must have shape [num_tiles + 1]");
  TORCH_CHECK(
      saved.gaussian_ids_sorted.dim() == 1,
      "saved.gaussian_ids_sorted must have shape [num_pairs]");
  TORCH_CHECK(saved.tile_ptr.numel() >= 1, "saved.tile_ptr must contain at least one element");
  TORCH_CHECK(
      read_last_index_value(saved.tile_ptr) == saved.gaussian_ids_sorted.numel(),
      "saved.tile_ptr[-1] must equal saved.gaussian_ids_sorted.numel()");
  auto tile_ptr_i64 = saved.tile_ptr.to(torch::kInt64);
  auto diffs = tile_ptr_i64.slice(0, 1) - tile_ptr_i64.slice(0, 0, -1);
  TORCH_CHECK(
      torch::all(diffs >= 0).item<bool>(),
      "saved.tile_ptr must be monotonic non-decreasing");
  if (saved.gaussian_ids_sorted.numel() > 0) {
    TORCH_CHECK(
        torch::all(saved.gaussian_ids_sorted >= 0).item<bool>(),
        "saved.gaussian_ids_sorted must be non-negative");
    TORCH_CHECK(
        torch::all(saved.gaussian_ids_sorted < saved.mu.size(0)).item<bool>(),
        "saved.gaussian_ids_sorted contains out-of-range gaussian ids");
  }
}

ForwardOrchestrationState build_forward_orchestration_state(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    const ForwardConfig& config) {
  auto tile = build_tile_state(row_coords, col_coords, config);
  auto geometry = preprocess_geometry(row_coords, col_coords, mu, chol_raw, config);
  auto binning = build_binning_state(geometry, tile.num_tile_rows, tile.num_tile_cols);
  auto saved = pack_saved_tensor_bundle(
      row_coords,
      col_coords,
      mu,
      chol_raw,
      amp,
      binning);
  return ForwardOrchestrationState{
      geometry,
      binning,
      tile,
      saved,
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gaussian_field_forward(
    const torch::Tensor& row_coords,
    const torch::Tensor& col_coords,
    const torch::Tensor& mu,
    const torch::Tensor& chol_raw,
    const torch::Tensor& amp,
    const ForwardConfig& config) {
  validate_forward_inputs(row_coords, col_coords, mu, chol_raw, amp);
  TORCH_CHECK(config.tile_out > 0, "tile_out must be positive");
  TORCH_CHECK(config.tile_in > 0, "tile_in must be positive");
  TORCH_CHECK(config.sigma_multiplier > 0.0, "sigma_multiplier must be positive");

  auto state = build_forward_orchestration_state(
      row_coords,
      col_coords,
      mu,
      chol_raw,
      amp,
      config);
  auto delta = forward_accumulate_tiles(
      row_coords,
      col_coords,
      mu,
      amp,
      state.geometry,
      state.binning,
      config);
  return {delta, state.saved.tile_ptr, state.saved.gaussian_ids_sorted};
}

std::vector<torch::Tensor> gaussian_field_backward(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config) {
  validate_backward_inputs(grad_delta, saved);
  validate_saved_state_contract(saved);
  TORCH_CHECK(config.tile_out > 0, "tile_out must be positive");
  TORCH_CHECK(config.tile_in > 0, "tile_in must be positive");
  TORCH_CHECK(config.sigma_multiplier > 0.0, "sigma_multiplier must be positive");

  TORCH_CHECK(
      grad_delta.numel() == saved.row_coords.size(0) * saved.col_coords.size(0),
      "grad_delta must match flattened delta shape");
  return backward_pair_reduce_train(grad_delta, saved, config);
}

std::vector<torch::Tensor> gaussian_field_backward_reference(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config) {
  validate_backward_inputs(grad_delta, saved);
  validate_saved_state_contract(saved);
  TORCH_CHECK(config.tile_out > 0, "tile_out must be positive");
  TORCH_CHECK(config.tile_in > 0, "tile_in must be positive");
  TORCH_CHECK(config.sigma_multiplier > 0.0, "sigma_multiplier must be positive");
  TORCH_CHECK(
      grad_delta.numel() == saved.row_coords.size(0) * saved.col_coords.size(0),
      "grad_delta must match flattened delta shape");
  return backward_reference_autograd(grad_delta, saved, config);
}

std::vector<torch::Tensor> gaussian_field_backward_train(
    const torch::Tensor& grad_delta,
    const SavedTensorBundle& saved,
    const ForwardConfig& config) {
  validate_backward_inputs(grad_delta, saved);
  validate_saved_state_contract(saved);
  TORCH_CHECK(config.tile_out > 0, "tile_out must be positive");
  TORCH_CHECK(config.tile_in > 0, "tile_in must be positive");
  TORCH_CHECK(config.sigma_multiplier > 0.0, "sigma_multiplier must be positive");

  TORCH_CHECK(
      grad_delta.numel() == saved.row_coords.size(0) * saved.col_coords.size(0),
      "grad_delta must match flattened delta shape");
  return backward_pair_reduce_train(grad_delta, saved, config);
}

}  // namespace gaussian_peft::cuda_field
