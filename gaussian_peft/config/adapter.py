from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class GaussianAdapterConfig:
    init_num_gaussians: int = 32
    init_method: str = "grid_overlap"
    coord_norm: str = "minus_one_to_one"
    covariance_type: str = "cholesky"
    use_amp_scaling: bool = True
    adapter_scale: float = 1.0
    chunk_size: int | None = None
    train_bias: bool = False
    merge_weights: bool = False
    normalize_gaussian: bool = False
    compute_dtype: torch.dtype = torch.float32
    eps: float = 1e-5
    init_amp_scale: float = 1e-4
    init_chol_scale_multiplier: float = 1.5
    min_cov_diag: float = 1e-4
    execution_mode: str = "cuda_field_train"
    tile_out: int = 32
    tile_in: int = 32
    sigma_multiplier: float = 3.0

    def validate(self) -> None:
        valid_init_methods = {"uniform_grid", "random_uniform", "grid_overlap"}
        valid_coord_norm = {"minus_one_to_one"}
        valid_covariance_types = {"cholesky"}
        valid_execution_modes = {
            "dense_reference",
            "tiled_reference_pytorch",
            "cuda_field_stage2_validation",
            "cuda_field_train",
        }

        if self.init_num_gaussians <= 0:
            raise ValueError("init_num_gaussians must be positive")
        if self.init_method not in valid_init_methods:
            raise ValueError(
                f"Unsupported init_method={self.init_method!r}. "
                f"Expected one of {sorted(valid_init_methods)}."
            )
        if self.coord_norm not in valid_coord_norm:
            raise ValueError(
                f"Unsupported coord_norm={self.coord_norm!r}. "
                f"Expected one of {sorted(valid_coord_norm)}."
            )
        if self.covariance_type not in valid_covariance_types:
            raise ValueError(
                f"Unsupported covariance_type={self.covariance_type!r}. "
                f"Expected one of {sorted(valid_covariance_types)}."
            )
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive when provided")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.min_cov_diag <= 0:
            raise ValueError("min_cov_diag must be positive")
        if self.init_amp_scale < 0:
            raise ValueError("init_amp_scale must be non-negative")
        if self.init_chol_scale_multiplier <= 0:
            raise ValueError("init_chol_scale_multiplier must be positive")
        if self.execution_mode not in valid_execution_modes:
            raise ValueError(
                f"Unsupported execution_mode={self.execution_mode!r}. "
                f"Expected one of {sorted(valid_execution_modes)}."
            )
        if self.tile_out <= 0 or self.tile_in <= 0:
            raise ValueError("tile_out and tile_in must be positive")
        if self.sigma_multiplier <= 0:
            raise ValueError("sigma_multiplier must be positive")
        if self.merge_weights:
            raise ValueError("merge_weights is disabled in the first implementation")
        self.execution_mode = normalize_execution_mode(self.execution_mode)


def normalize_execution_mode(execution_mode: str) -> str:
    aliases = {
        "dense": "dense_reference",
        "tiled": "tiled_reference_pytorch",
        "tiled_pytorch": "tiled_reference_pytorch",
        "tiled_triton_forward": "tiled_reference_pytorch",
        "cuda_field_stage3_custom": "cuda_field_train",
    }
    return aliases.get(execution_mode, execution_mode)
