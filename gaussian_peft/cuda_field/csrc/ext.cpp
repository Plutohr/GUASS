#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "field.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "gaussian_field_forward",
      [](const torch::Tensor& row_coords,
         const torch::Tensor& col_coords,
         const torch::Tensor& mu,
         const torch::Tensor& chol_raw,
         const torch::Tensor& amp,
         int64_t tile_out,
         int64_t tile_in,
         double sigma_multiplier,
         bool normalize,
         const c10::optional<double>& clamp_quad) {
        gaussian_peft::cuda_field::ForwardConfig config{
            tile_out,
            tile_in,
            sigma_multiplier,
            normalize,
            clamp_quad.has_value(),
            clamp_quad.value_or(0.0),
        };
        py::gil_scoped_release no_gil;
        return gaussian_peft::cuda_field::gaussian_field_forward(
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            config);
      });
  m.def(
      "gaussian_field_backward",
      [](const torch::Tensor& grad_delta,
         const torch::Tensor& row_coords,
         const torch::Tensor& col_coords,
         const torch::Tensor& mu,
         const torch::Tensor& chol_raw,
         const torch::Tensor& amp,
         const torch::Tensor& tile_ptr,
         const torch::Tensor& gaussian_ids_sorted,
         int64_t tile_out,
         int64_t tile_in,
         double sigma_multiplier,
         bool normalize,
         const c10::optional<double>& clamp_quad) {
        gaussian_peft::cuda_field::SavedTensorBundle saved{
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            tile_ptr,
            gaussian_ids_sorted,
        };
        gaussian_peft::cuda_field::ForwardConfig config{
            tile_out,
            tile_in,
            sigma_multiplier,
            normalize,
            clamp_quad.has_value(),
            clamp_quad.value_or(0.0),
        };
        py::gil_scoped_release no_gil;
        return gaussian_peft::cuda_field::gaussian_field_backward(grad_delta, saved, config);
      });
  m.def(
      "gaussian_field_backward_reference",
      [](const torch::Tensor& grad_delta,
         const torch::Tensor& row_coords,
         const torch::Tensor& col_coords,
         const torch::Tensor& mu,
         const torch::Tensor& chol_raw,
         const torch::Tensor& amp,
         const torch::Tensor& tile_ptr,
         const torch::Tensor& gaussian_ids_sorted,
         int64_t tile_out,
         int64_t tile_in,
         double sigma_multiplier,
         bool normalize,
         const c10::optional<double>& clamp_quad) {
        gaussian_peft::cuda_field::SavedTensorBundle saved{
            row_coords,
            col_coords,
            mu,
            chol_raw,
            amp,
            tile_ptr,
            gaussian_ids_sorted,
        };
        gaussian_peft::cuda_field::ForwardConfig config{
            tile_out,
            tile_in,
            sigma_multiplier,
            normalize,
            clamp_quad.has_value(),
            clamp_quad.value_or(0.0),
        };
        py::gil_scoped_release no_gil;
        return gaussian_peft::cuda_field::gaussian_field_backward_reference(
            grad_delta,
            saved,
            config);
      });
}
