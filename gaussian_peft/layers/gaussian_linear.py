from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.cuda_field.runtime import (
    gaussian_field_cell_average_diag_v1_cuda,
    gaussian_field_cuda_field_stage2_validation,
    gaussian_field_train,
)
from gaussian_peft.initializers.grid import GridInitializer
from gaussian_peft.kernels.cell_average_diag import gaussian_field_cell_average_diag_v1
from gaussian_peft.kernels.coords import build_linear_axes, build_linear_coords, reshape_delta_to_weight
from gaussian_peft.kernels.covariance import (
    activate_cholesky,
    activate_diag_sigma_from_chol_slots,
    zero_inactive_chol_slot_,
)
from gaussian_peft.kernels.gaussian_field import gaussian_field
from gaussian_peft.layers.base import GaussianAdapterBase


class GaussianLinear(nn.Linear, GaussianAdapterBase):
    """Linear layer with a Gaussian field adapter over Delta W."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        adapter_config: GaussianAdapterConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        GaussianAdapterBase.__init__(self, adapter_config=adapter_config)
        self.weight.requires_grad = False
        if self.bias is not None and not self.adapter_config.train_bias:
            self.bias.requires_grad = False

        row_coords, col_coords = self._build_axes()
        self.register_buffer("row_coords", row_coords, persistent=False)
        self.register_buffer("col_coords", col_coords, persistent=False)
        self.last_forward_metadata: dict[str, Tensor | float | str] = {}
        self._gradient_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.reset_gaussian_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        adapter_config: GaussianAdapterConfig,
    ) -> "GaussianLinear":
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            adapter_config=adapter_config,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias)
        if adapter_config.init_method == "weight_absmin_positions":
            layer.reset_gaussian_parameters()
        return layer

    @property
    def num_gaussians(self) -> int:
        return int(self.mu_raw.shape[0])

    def reset_gaussian_parameters(self) -> None:
        param_dtype = self._parameter_dtype()
        if self.adapter_config.init_method == "grid_overlap":
            init_state = GridInitializer.initialize(
                num_gaussians=self.adapter_config.init_num_gaussians,
                device=self.weight.device,
                dtype=param_dtype,
                eps=self.adapter_config.eps,
                amp_scale=self.adapter_config.init_amp_scale,
                min_cov_diag=self.adapter_config.min_cov_diag,
                coord_range=(-1.0, 1.0),
                chol_scale_multiplier=self.adapter_config.init_chol_scale_multiplier,
            )
            mu_raw = self._inverse_mu_activation(init_state.mu_model)
            chol_raw = init_state.chol_raw
            amp = init_state.amp
        else:
            mu_raw = self._init_mu()
            chol_raw = self._init_chol_raw()
            amp = self._init_amp()
        if self.adapter_config.uses_cell_average_v1():
            # V1 overrides the Gaussian shape semantics, so keep the grid centers
            # but rebuild both sigma and amplitude from the V1-specific initializer.
            chol_raw = self._init_chol_raw()
            amp = self._init_amp()
        mu_raw, chol_raw = self._apply_initial_parameter_jitter(mu_raw=mu_raw, chol_raw=chol_raw)
        self.mu_raw = nn.Parameter(mu_raw)
        self.chol_raw = nn.Parameter(chol_raw)
        self.amp = nn.Parameter(amp)
        self._refresh_parameter_gradient_hooks()
        self.enforce_parameter_constraints_()

    def _init_mu(self) -> Tensor:
        k = self.adapter_config.init_num_gaussians
        device = self.weight.device
        dtype = self._parameter_dtype()
        if self.adapter_config.init_method == "uniform_grid":
            side = math.ceil(math.sqrt(k))
            x = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
            y = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
            grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
            mu = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)[:k]
        elif self.adapter_config.init_method == "random_uniform":
            mu = torch.empty(k, 2, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        elif self.adapter_config.init_method == "weight_absmin_positions":
            mu = self._init_mu_from_weight_absmin().to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported init_method: {self.adapter_config.init_method!r}")
        return self._inverse_mu_activation(mu)

    def _init_mu_from_weight_absmin(self) -> Tensor:
        k = self.adapter_config.init_num_gaussians
        flat_abs_weight = self.weight.detach().abs().reshape(-1)
        num_positions = int(flat_abs_weight.numel())
        if num_positions <= 0:
            raise ValueError("weight_absmin_positions requires a non-empty base weight tensor")

        if k <= num_positions:
            selected_indices = torch.topk(
                flat_abs_weight,
                k,
                largest=False,
                sorted=True,
            ).indices
        else:
            sorted_indices = torch.argsort(flat_abs_weight, stable=True)
            repeats = math.ceil(k / num_positions)
            selected_indices = sorted_indices.repeat(repeats)[:k]

        row_indices = torch.div(selected_indices, self.in_features, rounding_mode="floor")
        col_indices = torch.remainder(selected_indices, self.in_features)
        mu = torch.stack(
            (
                self.row_coords[row_indices],
                self.col_coords[col_indices],
            ),
            dim=-1,
        )
        return mu

    def _init_chol_raw(self) -> Tensor:
        k = self.adapter_config.init_num_gaussians
        param_dtype = self._parameter_dtype()
        chol_raw = torch.zeros(k, 3, device=self.weight.device, dtype=param_dtype)
        if self.adapter_config.uses_cell_average_v1():
            sigma_x, sigma_y = self._initial_v1_sigma()
            target_sigma = torch.tensor(
                [sigma_x, sigma_y],
                device=self.weight.device,
                dtype=param_dtype,
            ).expand(k, 2)
            raw_target = (target_sigma - self.adapter_config.sigma_min).clamp_min(self.adapter_config.eps)
            chol_raw[:, 0] = self._inverse_softplus(raw_target[:, 0])
            chol_raw[:, 2] = self._inverse_softplus(raw_target[:, 1])
            return chol_raw
        target_scale = max(
            2.0 / math.sqrt(max(k, 1)) * self.adapter_config.init_chol_scale_multiplier,
            self.adapter_config.min_cov_diag ** 0.5,
        )
        target_diag = torch.full((k, 2), target_scale, device=self.weight.device, dtype=param_dtype)
        chol_raw[:, 0] = self._inverse_softplus(target_diag[:, 0] - self.adapter_config.eps)
        chol_raw[:, 2] = self._inverse_softplus(target_diag[:, 1] - self.adapter_config.eps)
        return chol_raw

    def _init_amp(self) -> Tensor:
        k = self.adapter_config.init_num_gaussians
        amp = torch.zeros(k, 1, device=self.weight.device, dtype=self._parameter_dtype())
        scale = self.adapter_config.init_amp_scale
        if scale > 0:
            amp.normal_(mean=0.0, std=scale)
        return amp

    def _build_coords(self) -> Tensor:
        coords = build_linear_coords(
            out_features=self.out_features,
            in_features=self.in_features,
            device=self.weight.device,
            dtype=torch.float32,
        )
        return coords

    def _build_axes(self) -> tuple[Tensor, Tensor]:
        return build_linear_axes(
            out_features=self.out_features,
            in_features=self.in_features,
            device=self.weight.device,
            dtype=torch.float32,
        )

    def materialize_mu(self) -> Tensor:
        return torch.tanh(self.mu_raw)

    def materialize_cholesky(self) -> Tensor:
        return activate_cholesky(self.chol_raw, eps=self.adapter_config.eps)

    def materialize_sigma(self) -> tuple[Tensor, Tensor]:
        return activate_diag_sigma_from_chol_slots(
            self.chol_raw,
            sigma_min=self.adapter_config.sigma_min,
        )

    def _build_dense_reference_coords(self) -> Tensor:
        return self._build_coords()

    def _component_count_scale_factor(self) -> float:
        mode = self.adapter_config.component_count_normalization
        if mode == "none":
            return 1.0
        if mode == "sqrt_num_gaussians":
            return math.sqrt(max(self.num_gaussians, 1))
        raise ValueError(f"Unsupported component_count_normalization: {mode!r}")

    def compute_delta_weight(self) -> Tensor:
        autocast_context = _gaussian_compute_autocast_context(self.weight.device.type)
        with autocast_context:
            if self.adapter_config.readout_scheme == "cell_average_diag_v1":
                if self.adapter_config.execution_backend == "torch_reference":
                    result = gaussian_field_cell_average_diag_v1(
                        mu_raw=self.mu_raw,
                        chol_raw=self.chol_raw,
                        amp=self.amp,
                        out_features=self.out_features,
                        in_features=self.in_features,
                        sigma_min=self.adapter_config.sigma_min,
                        chunk_size=self.adapter_config.chunk_size,
                        compute_dtype=self.adapter_config.compute_dtype,
                        return_metadata=True,
                    )
                elif self.adapter_config.execution_backend == "cuda_cellavg_diag_v1":
                    delta = gaussian_field_cell_average_diag_v1_cuda(
                        mu_raw=self.mu_raw,
                        chol_raw=self.chol_raw,
                        amp=self.amp,
                        out_features=self.out_features,
                        in_features=self.in_features,
                        sigma_min=self.adapter_config.sigma_min,
                    )
                    sigma_x, sigma_y = self.materialize_sigma()
                    empty = torch.empty(0, device=self.weight.device, dtype=torch.int64)
                    result = (
                        delta,
                        {
                            "readout_scheme": "cell_average_diag_v1",
                            "execution_backend": "cuda_cellavg_diag_v1",
                            "execution_mode": "cell_average_diag_v1_cuda",
                            "accumulate_backend": "cuda_cellavg_diag_v1",
                            "tiles_touched": empty,
                            "tile_ptr": empty,
                            "gaussian_ids": empty,
                            "per_tile_counts": empty,
                            "num_tile_rows": 1.0,
                            "num_tile_cols": 1.0,
                            "sigma_x_mean": float(sigma_x.detach().mean().item()),
                            "sigma_x_min": float(sigma_x.detach().min().item()),
                            "sigma_x_max": float(sigma_x.detach().max().item()),
                            "sigma_y_mean": float(sigma_y.detach().mean().item()),
                            "sigma_y_min": float(sigma_y.detach().min().item()),
                            "sigma_y_max": float(sigma_y.detach().max().item()),
                        },
                    )
                else:
                    raise NotImplementedError(
                        "cell_average_diag_v1 currently only supports execution_backend in "
                        "{'torch_reference', 'cuda_cellavg_diag_v1'}"
                    )
            elif self.adapter_config.execution_backend == "cuda_field_stage2_validation":
                result = gaussian_field_cuda_field_stage2_validation(
                    row_coords=self.row_coords,
                    col_coords=self.col_coords,
                    mu=self.materialize_mu(),
                    chol_raw=self.chol_raw,
                    amp=self.amp,
                    tile_out=self.adapter_config.tile_out,
                    tile_in=self.adapter_config.tile_in,
                    sigma_multiplier=self.adapter_config.sigma_multiplier,
                    normalize=self.adapter_config.normalize_gaussian,
                    clamp_quad=80.0,
                )
            elif self.adapter_config.execution_backend == "cuda_field":
                delta = gaussian_field_train(
                    row_coords=self.row_coords,
                    col_coords=self.col_coords,
                    mu=self.materialize_mu(),
                    chol_raw=self.chol_raw,
                    amp=self.amp,
                    tile_out=self.adapter_config.tile_out,
                    tile_in=self.adapter_config.tile_in,
                    sigma_multiplier=self.adapter_config.sigma_multiplier,
                    normalize=self.adapter_config.normalize_gaussian,
                    clamp_quad=80.0,
                )
                result = (
                    delta,
                    {
                        "readout_scheme": "point_sample",
                        "execution_backend": "cuda_field",
                        "execution_mode": "cuda_field_train",
                        "accumulate_backend": "cuda_field_train",
                    },
                )
            else:
                result = gaussian_field(
                    coords=self._build_dense_reference_coords(),
                    mu=self.materialize_mu(),
                    chol=self.materialize_cholesky(),
                    amp=self.amp,
                    chunk_size=self.adapter_config.chunk_size,
                    normalize=self.adapter_config.normalize_gaussian,
                    compute_dtype=self.adapter_config.compute_dtype,
                    execution_mode="tiled_reference_pytorch",
                    row_coords=self.row_coords,
                    col_coords=self.col_coords,
                    tile_out=self.adapter_config.tile_out,
                    tile_in=self.adapter_config.tile_in,
                    sigma_multiplier=self.adapter_config.sigma_multiplier,
                    return_metadata=True,
                )
        delta, metadata = result
        delta_weight = reshape_delta_to_weight(delta, self.out_features, self.in_features)
        metadata = dict(metadata)
        raw_delta_w_norm = float(delta_weight.detach().float().norm().item())
        component_count_scale_factor = self._component_count_scale_factor()
        if component_count_scale_factor != 1.0:
            delta_weight = delta_weight / component_count_scale_factor
        metadata.setdefault("readout_scheme", self.adapter_config.readout_scheme)
        metadata.setdefault("execution_backend", self.adapter_config.execution_backend or "torch_reference")
        metadata["component_count_normalization"] = self.adapter_config.component_count_normalization
        metadata["component_count_scale_factor"] = float(component_count_scale_factor)
        metadata["delta_w_raw_norm"] = raw_delta_w_norm
        metadata["delta_w_norm"] = float(delta_weight.detach().float().norm().item())
        self.last_forward_metadata = metadata
        return delta_weight * self.adapter_config.adapter_scale

    def _delta_apply_row_chunk(self) -> int:
        base = max(1, int(self.adapter_config.tile_out))
        return min(self.out_features, max(base, 64))

    def _apply_delta_weight_chunked(self, input: Tensor, base_output: Tensor, delta_weight: Tensor) -> Tensor:
        row_chunk = self._delta_apply_row_chunk()
        output = base_output.clone()
        for row_start in range(0, self.out_features, row_chunk):
            row_end = min(row_start + row_chunk, self.out_features)
            delta_chunk = delta_weight[row_start:row_end, :]
            output[..., row_start:row_end].add_(F.linear(input, delta_chunk, None))
        return output

    def forward(self, input: Tensor) -> Tensor:
        delta_weight = self.compute_delta_weight().to(dtype=self.weight.dtype)
        base_output = F.linear(input, self.weight, self.bias)
        metadata = dict(self.last_forward_metadata)
        metadata["linear_composition"] = "base_plus_delta_row_chunk_accumulate"
        metadata["delta_apply_row_chunk"] = float(self._delta_apply_row_chunk())
        self.last_forward_metadata = metadata
        return self._apply_delta_weight_chunked(input, base_output, delta_weight)

    def get_gaussian_parameters(self) -> dict[str, nn.Parameter]:
        return {
            "mu_raw": self.mu_raw,
            "chol_raw": self.chol_raw,
            "amp": self.amp,
        }

    def load_adapter_state(self, state: dict[str, Tensor]) -> None:
        self.replace_gaussian_parameters_(
            mu_raw=state["mu_raw"],
            chol_raw=state["chol_raw"],
            amp=state["amp"],
        )

    def replace_gaussian_parameters_(
        self,
        *,
        mu_raw: Tensor | None = None,
        chol_raw: Tensor | None = None,
        amp: Tensor | None = None,
    ) -> None:
        param_dtype = self._parameter_dtype()
        if mu_raw is not None:
            self.mu_raw = nn.Parameter(mu_raw.to(device=self.weight.device, dtype=param_dtype))
        if chol_raw is not None:
            self.chol_raw = nn.Parameter(
                chol_raw.to(device=self.weight.device, dtype=param_dtype)
            )
        if amp is not None:
            self.amp = nn.Parameter(amp.to(device=self.weight.device, dtype=param_dtype))
        self._refresh_parameter_gradient_hooks()
        self.enforce_parameter_constraints_()

    def _refresh_parameter_gradient_hooks(self) -> None:
        for handle in self._gradient_hook_handles:
            handle.remove()
        self._gradient_hook_handles = []
        self._maybe_register_gradient_multiplier(self.mu_raw, self.adapter_config.mu_grad_multiplier)
        self._maybe_register_gradient_multiplier(self.chol_raw, self.adapter_config.chol_grad_multiplier)

    def _maybe_register_gradient_multiplier(self, param: nn.Parameter, scale: float) -> None:
        if scale == 1.0:
            return

        def _scale_grad(grad: Tensor) -> Tensor:
            return grad * scale

        self._gradient_hook_handles.append(param.register_hook(_scale_grad))

    def enforce_parameter_constraints_(self) -> None:
        if not self.adapter_config.uses_cell_average_v1():
            return
        with torch.no_grad():
            zero_inactive_chol_slot_(self.chol_raw)

    def project_effective_chol_tensor(self, tensor: Tensor | None) -> Tensor | None:
        if tensor is None or not self.adapter_config.uses_cell_average_v1():
            return tensor
        projected = tensor.clone()
        projected[..., 1] = 0
        return projected

    def _initial_v1_sigma(self) -> tuple[float, float]:
        sigma_x = max(
            self.adapter_config.init_chol_scale_multiplier * (2.0 / max(self.in_features, 1)),
            self.adapter_config.sigma_min + self.adapter_config.eps,
        )
        sigma_y = max(
            self.adapter_config.init_chol_scale_multiplier * (2.0 / max(self.out_features, 1)),
            self.adapter_config.sigma_min + self.adapter_config.eps,
        )
        return sigma_x, sigma_y

    def _parameter_dtype(self) -> torch.dtype:
        compute_dtype = self.adapter_config.compute_dtype
        if compute_dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return compute_dtype

    def _apply_initial_parameter_jitter(
        self,
        *,
        mu_raw: Tensor,
        chol_raw: Tensor,
    ) -> tuple[Tensor, Tensor]:
        jittered_mu_raw = mu_raw.clone()
        jittered_chol_raw = chol_raw.clone()

        if self.adapter_config.init_mu_jitter_scale > 0:
            mu_model = torch.tanh(jittered_mu_raw)
            mu_noise = torch.randn_like(mu_model) * self.adapter_config.init_mu_jitter_scale
            mu_model = (mu_model + mu_noise).clamp(min=-0.999999, max=0.999999)
            jittered_mu_raw = self._inverse_mu_activation(mu_model)

        if self.adapter_config.init_sigma_log_jitter_scale > 0:
            sigma_x, sigma_y = activate_diag_sigma_from_chol_slots(
                jittered_chol_raw,
                sigma_min=self.adapter_config.sigma_min,
            )
            sigma_noise_x = torch.randn_like(sigma_x) * self.adapter_config.init_sigma_log_jitter_scale
            sigma_noise_y = torch.randn_like(sigma_y) * self.adapter_config.init_sigma_log_jitter_scale
            min_sigma = self.adapter_config.sigma_min + self.adapter_config.eps
            sigma_x = (sigma_x * sigma_noise_x.exp()).clamp_min(min_sigma)
            sigma_y = (sigma_y * sigma_noise_y.exp()).clamp_min(min_sigma)
            sigma_x_raw = self._inverse_softplus(
                (sigma_x - self.adapter_config.sigma_min).clamp_min(self.adapter_config.eps)
            )
            sigma_y_raw = self._inverse_softplus(
                (sigma_y - self.adapter_config.sigma_min).clamp_min(self.adapter_config.eps)
            )
            jittered_chol_raw[:, 0] = sigma_x_raw
            jittered_chol_raw[:, 2] = sigma_y_raw
            if self.adapter_config.uses_cell_average_v1():
                jittered_chol_raw[:, 1] = 0

        return jittered_mu_raw, jittered_chol_raw

    @staticmethod
    def _inverse_mu_activation(mu: Tensor) -> Tensor:
        clipped = mu.clamp(min=-0.999999, max=0.999999)
        return torch.atanh(clipped)

    @staticmethod
    def _inverse_softplus(value: Tensor) -> Tensor:
        return torch.log(torch.expm1(value))


def _gaussian_compute_autocast_context(device_type: str):
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()
