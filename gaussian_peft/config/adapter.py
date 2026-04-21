from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

VALID_READOUT_SCHEMES = {"point_sample", "cell_average_diag_v1"}
VALID_COORD_NORM = {"minus_one_to_one"}
VALID_COVARIANCE_TYPES = {"cholesky"}
VALID_EXECUTION_BACKENDS = {
    "torch_reference",
    "cuda_field",
    "cuda_field_stage2_validation",
    "cuda_cellavg_diag_v1",
}
VALID_GAUSSIAN_NORMALIZATIONS = {"unnormalized_legacy", "normalized_density"}
VALID_COMPONENT_COUNT_NORMALIZATIONS = {"none", "sqrt_num_gaussians"}


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
    gaussian_normalization: str | None = None
    component_count_normalization: str = "none"
    mu_grad_multiplier: float = 1.0
    chol_grad_multiplier: float = 1.0
    domain_renorm: bool = False
    compute_dtype: torch.dtype = torch.float32
    eps: float = 1e-5
    sigma_min: float = 1e-3
    init_amp_scale: float = 1e-4
    init_chol_scale_multiplier: float = 1.5
    init_mu_jitter_scale: float = 0.0
    init_sigma_log_jitter_scale: float = 0.0
    min_cov_diag: float = 1e-4
    readout_scheme: str = "point_sample"
    execution_backend: str | None = None
    execution_mode: str | None = None
    tile_out: int = 32
    tile_in: int = 32
    sigma_multiplier: float = 3.0

    def validate(self) -> None:
        valid_init_methods = {
            "uniform_grid",
            "random_uniform",
            "grid_overlap",
            "weight_absmin_positions",
        }

        if self.init_num_gaussians <= 0:
            raise ValueError("init_num_gaussians must be positive")
        if self.init_method not in valid_init_methods:
            raise ValueError(
                f"Unsupported init_method={self.init_method!r}. "
                f"Expected one of {sorted(valid_init_methods)}."
            )
        if self.coord_norm not in VALID_COORD_NORM:
            raise ValueError(
                f"Unsupported coord_norm={self.coord_norm!r}. "
                f"Expected one of {sorted(VALID_COORD_NORM)}."
            )
        if self.covariance_type not in VALID_COVARIANCE_TYPES:
            raise ValueError(
                f"Unsupported covariance_type={self.covariance_type!r}. "
                f"Expected one of {sorted(VALID_COVARIANCE_TYPES)}."
            )
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive when provided")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive")
        if self.min_cov_diag <= 0:
            raise ValueError("min_cov_diag must be positive")
        if self.init_amp_scale < 0:
            raise ValueError("init_amp_scale must be non-negative")
        if self.init_mu_jitter_scale < 0:
            raise ValueError("init_mu_jitter_scale must be non-negative")
        if self.init_sigma_log_jitter_scale < 0:
            raise ValueError("init_sigma_log_jitter_scale must be non-negative")
        if self.mu_grad_multiplier <= 0:
            raise ValueError("mu_grad_multiplier must be positive")
        if self.chol_grad_multiplier <= 0:
            raise ValueError("chol_grad_multiplier must be positive")
        if self.init_chol_scale_multiplier <= 0:
            raise ValueError("init_chol_scale_multiplier must be positive")
        if self.tile_out <= 0 or self.tile_in <= 0:
            raise ValueError("tile_out and tile_in must be positive")
        if self.sigma_multiplier <= 0:
            raise ValueError("sigma_multiplier must be positive")
        if self.merge_weights:
            raise ValueError("merge_weights is disabled in the first implementation")

        self.readout_scheme = normalize_readout_scheme(self.readout_scheme)
        self.execution_backend = resolve_execution_backend(
            readout_scheme=self.readout_scheme,
            execution_backend=self.execution_backend,
            execution_mode=self.execution_mode,
        )
        self.gaussian_normalization = resolve_gaussian_normalization(
            gaussian_normalization=self.gaussian_normalization,
            normalize_gaussian=self.normalize_gaussian,
        )
        self.component_count_normalization = normalize_component_count_normalization(
            self.component_count_normalization
        )
        self.execution_mode = self.execution_backend
        self.normalize_gaussian = self.gaussian_normalization == "normalized_density"

        if self.readout_scheme == "cell_average_diag_v1":
            if self.execution_backend not in {"torch_reference", "cuda_cellavg_diag_v1"}:
                raise ValueError(
                    "readout_scheme='cell_average_diag_v1' only supports "
                    "execution_backend in {'torch_reference', 'cuda_cellavg_diag_v1'}."
                )
            if self.gaussian_normalization != "normalized_density":
                raise ValueError(
                    "readout_scheme='cell_average_diag_v1' requires "
                    "gaussian_normalization='normalized_density'."
                )
            if self.domain_renorm:
                raise ValueError(
                    "readout_scheme='cell_average_diag_v1' does not support "
                    "domain_renorm=True in V1."
                )

    def uses_cell_average_v1(self) -> bool:
        return self.readout_scheme == "cell_average_diag_v1"

    def checkpoint_semantics(self) -> dict[str, Any]:
        return {
            "readout_scheme": self.readout_scheme,
            "coord_domain": self.coord_norm,
            "mu_parameterization": "tanh",
            "cov_parameterization": (
                "diag_from_chol_slots_v1" if self.uses_cell_average_v1() else "cholesky"
            ),
            "gaussian_normalization": self.gaussian_normalization,
            "component_count_normalization": self.component_count_normalization,
            "domain_renorm": self.domain_renorm,
            "execution_backend": self.execution_backend,
        }


def normalize_readout_scheme(readout_scheme: str) -> str:
    aliases = {
        "point": "point_sample",
        "point_sample": "point_sample",
        "cell_average": "cell_average_diag_v1",
        "cell_average_diag": "cell_average_diag_v1",
        "cell_average_diag_v1": "cell_average_diag_v1",
    }
    normalized = aliases.get(readout_scheme, readout_scheme)
    if normalized not in VALID_READOUT_SCHEMES:
        raise ValueError(
            f"Unsupported readout_scheme={readout_scheme!r}. "
            f"Expected one of {sorted(VALID_READOUT_SCHEMES)}."
        )
    return normalized


def normalize_execution_backend(execution_backend: str | None) -> str | None:
    if execution_backend is None:
        return None
    aliases = {
        "dense": "torch_reference",
        "dense_reference": "torch_reference",
        "tiled": "torch_reference",
        "tiled_pytorch": "torch_reference",
        "tiled_reference_pytorch": "torch_reference",
        "tiled_triton_forward": "torch_reference",
        "torch": "torch_reference",
        "torch_reference": "torch_reference",
        "cuda_cellavg": "cuda_cellavg_diag_v1",
        "cuda_cellavg_diag": "cuda_cellavg_diag_v1",
        "cuda_cellavg_diag_v1": "cuda_cellavg_diag_v1",
        "cuda_field_train": "cuda_field",
        "cuda_field_stage3_custom": "cuda_field",
        "cuda_field": "cuda_field",
        "cuda_field_stage2_validation": "cuda_field_stage2_validation",
    }
    normalized = aliases.get(execution_backend, execution_backend)
    if normalized not in VALID_EXECUTION_BACKENDS:
        raise ValueError(
            f"Unsupported execution_backend={execution_backend!r}. "
            f"Expected one of {sorted(VALID_EXECUTION_BACKENDS)}."
        )
    return normalized


def resolve_execution_backend(
    *,
    readout_scheme: str,
    execution_backend: str | None,
    execution_mode: str | None,
) -> str:
    normalized_backend = normalize_execution_backend(execution_backend)
    normalized_mode = normalize_execution_backend(execution_mode)
    if normalized_backend is not None and normalized_mode is not None and normalized_backend != normalized_mode:
        raise ValueError(
            f"execution_backend={execution_backend!r} conflicts with "
            f"legacy execution_mode={execution_mode!r}"
        )
    resolved = normalized_backend or normalized_mode
    if resolved is None:
        return "torch_reference" if readout_scheme == "cell_average_diag_v1" else "cuda_field"
    return resolved


def resolve_gaussian_normalization(
    *,
    gaussian_normalization: str | None,
    normalize_gaussian: bool,
) -> str:
    if gaussian_normalization is None:
        return "normalized_density" if normalize_gaussian else "unnormalized_legacy"
    aliases = {
        "normalized": "normalized_density",
        "normalized_density": "normalized_density",
        "legacy_unnormalized": "unnormalized_legacy",
        "unnormalized": "unnormalized_legacy",
        "unnormalized_legacy": "unnormalized_legacy",
    }
    normalized = aliases.get(gaussian_normalization, gaussian_normalization)
    if normalized not in VALID_GAUSSIAN_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported gaussian_normalization={gaussian_normalization!r}. "
            f"Expected one of {sorted(VALID_GAUSSIAN_NORMALIZATIONS)}."
        )
    return normalized


def normalize_component_count_normalization(component_count_normalization: str) -> str:
    aliases = {
        "none": "none",
        "sqrt_k": "sqrt_num_gaussians",
        "sqrt_gaussians": "sqrt_num_gaussians",
        "sqrt_num_gaussians": "sqrt_num_gaussians",
    }
    normalized = aliases.get(component_count_normalization, component_count_normalization)
    if normalized not in VALID_COMPONENT_COUNT_NORMALIZATIONS:
        raise ValueError(
            "Unsupported component_count_normalization="
            f"{component_count_normalization!r}. "
            f"Expected one of {sorted(VALID_COMPONENT_COUNT_NORMALIZATIONS)}."
        )
    return normalized


def normalize_execution_mode(execution_mode: str | None) -> str | None:
    return normalize_execution_backend(execution_mode)
