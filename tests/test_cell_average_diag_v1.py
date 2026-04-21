from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import math
import torch

from gaussian_peft.checkpoints.io import save_adapter_checkpoint
from gaussian_peft.checkpoints.state_dict import validate_checkpoint_semantics
from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.config.training import TrainingConfig
from gaussian_peft.cuda_field.runtime import gaussian_field_cell_average_diag_v1_cuda
from gaussian_peft.kernels.cell_average_diag import gaussian_field_cell_average_diag_v1
from gaussian_peft.layers.gaussian_linear import GaussianLinear


class CellAverageDiagV1Tests(unittest.TestCase):
    def test_small_sigma_matches_point_sample_limit(self) -> None:
        dtype = torch.float32
        target_sigma = 0.05
        sigma_offset = 1e-6
        mu_raw = torch.zeros(1, 2, dtype=dtype)
        chol_raw = torch.zeros(1, 3, dtype=dtype)
        chol_raw[:, 0] = self._inverse_softplus(torch.tensor([target_sigma - sigma_offset], dtype=dtype))
        chol_raw[:, 2] = self._inverse_softplus(torch.tensor([target_sigma - sigma_offset], dtype=dtype))
        amp = torch.tensor([[0.7]], dtype=dtype)

        delta_v1 = gaussian_field_cell_average_diag_v1(
            mu_raw=mu_raw,
            chol_raw=chol_raw,
            amp=amp,
            out_features=256,
            in_features=256,
            sigma_min=sigma_offset,
            compute_dtype=torch.float32,
        )
        delta_point = self._cell_center_point_sample_reference(
            mu_x=0.0,
            mu_y=0.0,
            sigma_x=target_sigma,
            sigma_y=target_sigma,
            amp=float(amp.item()),
            out_features=256,
            in_features=256,
            dtype=dtype,
        )
        max_error = (delta_v1 - delta_point).abs().max().item()
        self.assertLess(max_error, 0.15)

    def test_v1_gradients_reach_mu_and_active_sigma_slots(self) -> None:
        torch.manual_seed(0)
        config = GaussianAdapterConfig(
            init_num_gaussians=4,
            init_method="random_uniform",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            sigma_min=1e-3,
        )
        config.validate()
        layer = GaussianLinear(
            in_features=12,
            out_features=10,
            bias=False,
            adapter_config=config,
        )
        with torch.no_grad():
            layer.mu_raw.copy_(
                torch.tensor(
                    [
                        [-0.8, -0.2],
                        [-0.3, 0.5],
                        [0.4, -0.6],
                        [0.7, 0.1],
                    ],
                    dtype=layer.mu_raw.dtype,
                )
            )
            layer.chol_raw.zero_()
            layer.chol_raw[:, 0] = self._inverse_softplus(torch.full((4,), 0.18, dtype=layer.chol_raw.dtype))
            layer.chol_raw[:, 2] = self._inverse_softplus(torch.full((4,), 0.24, dtype=layer.chol_raw.dtype))
            layer.amp.copy_(torch.tensor([[0.1], [-0.2], [0.15], [0.3]], dtype=layer.amp.dtype))
            layer.enforce_parameter_constraints_()

        loss = layer.compute_delta_weight().square().mean()
        loss.backward()

        self.assertIsNotNone(layer.mu_raw.grad)
        self.assertIsNotNone(layer.chol_raw.grad)
        self.assertGreater(float(layer.mu_raw.grad.detach().abs().sum().item()), 0.0)
        self.assertGreater(float(layer.chol_raw.grad[:, [0, 2]].detach().abs().sum().item()), 0.0)
        self.assertEqual(float(layer.chol_raw.grad[:, 1].detach().abs().sum().item()), 0.0)

    def test_boundary_forward_backward_are_finite(self) -> None:
        torch.manual_seed(0)
        config = GaussianAdapterConfig(
            init_num_gaussians=1,
            init_method="random_uniform",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            sigma_min=1e-3,
        )
        config.validate()
        layer = GaussianLinear(
            in_features=8,
            out_features=8,
            bias=False,
            adapter_config=config,
        )
        with torch.no_grad():
            layer.mu_raw[:, 0] = torch.atanh(torch.tensor([0.999], dtype=layer.mu_raw.dtype))
            layer.mu_raw[:, 1] = torch.atanh(torch.tensor([-0.999], dtype=layer.mu_raw.dtype))
            layer.amp.fill_(0.2)
        loss = layer.compute_delta_weight().sum()
        loss.backward()

        self.assertTrue(torch.isfinite(loss.detach()).all())
        self.assertTrue(torch.isfinite(layer.mu_raw.grad).all())
        self.assertTrue(torch.isfinite(layer.chol_raw.grad).all())

    def test_component_count_sqrt_normalization_scales_delta_weight(self) -> None:
        torch.manual_seed(0)
        base_config = GaussianAdapterConfig(
            init_num_gaussians=4,
            init_method="random_uniform",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            component_count_normalization="none",
            sigma_min=1e-3,
        )
        scaled_config = GaussianAdapterConfig(
            init_num_gaussians=4,
            init_method="random_uniform",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            component_count_normalization="sqrt_num_gaussians",
            sigma_min=1e-3,
        )
        base_config.validate()
        scaled_config.validate()

        base_layer = GaussianLinear(
            in_features=10,
            out_features=12,
            bias=False,
            adapter_config=base_config,
        )
        scaled_layer = GaussianLinear(
            in_features=10,
            out_features=12,
            bias=False,
            adapter_config=scaled_config,
        )
        with torch.no_grad():
            mu_raw = torch.tensor(
                [
                    [-0.8, -0.2],
                    [-0.3, 0.5],
                    [0.4, -0.6],
                    [0.7, 0.1],
                ],
                dtype=base_layer.mu_raw.dtype,
            )
            chol_diag_x = self._inverse_softplus(torch.full((4,), 0.18, dtype=base_layer.chol_raw.dtype))
            chol_diag_y = self._inverse_softplus(torch.full((4,), 0.24, dtype=base_layer.chol_raw.dtype))
            amp = torch.tensor([[0.1], [-0.2], [0.15], [0.3]], dtype=base_layer.amp.dtype)

            for layer in (base_layer, scaled_layer):
                layer.mu_raw.copy_(mu_raw)
                layer.chol_raw.zero_()
                layer.chol_raw[:, 0] = chol_diag_x
                layer.chol_raw[:, 2] = chol_diag_y
                layer.amp.copy_(amp)
                layer.enforce_parameter_constraints_()

        base_delta = base_layer.compute_delta_weight()
        scaled_delta = scaled_layer.compute_delta_weight()

        expected_scale = math.sqrt(base_layer.num_gaussians)
        torch.testing.assert_close(scaled_delta, base_delta / expected_scale, rtol=1e-5, atol=1e-6)
        self.assertEqual(
            float(scaled_layer.last_forward_metadata["component_count_scale_factor"]),
            expected_scale,
        )

    def test_gradient_multipliers_scale_parameter_grads(self) -> None:
        torch.manual_seed(0)
        base_config = GaussianAdapterConfig(
            init_num_gaussians=4,
            init_method="random_uniform",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            sigma_min=1e-3,
            mu_grad_multiplier=1.0,
            chol_grad_multiplier=1.0,
        )
        scaled_config = GaussianAdapterConfig(
            init_num_gaussians=4,
            init_method="random_uniform",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            sigma_min=1e-3,
            mu_grad_multiplier=10.0,
            chol_grad_multiplier=50.0,
        )
        base_config.validate()
        scaled_config.validate()

        base_layer = GaussianLinear(
            in_features=10,
            out_features=12,
            bias=False,
            adapter_config=base_config,
        )
        scaled_layer = GaussianLinear(
            in_features=10,
            out_features=12,
            bias=False,
            adapter_config=scaled_config,
        )
        with torch.no_grad():
            mu_raw = torch.tensor(
                [
                    [-0.8, -0.2],
                    [-0.3, 0.5],
                    [0.4, -0.6],
                    [0.7, 0.1],
                ],
                dtype=base_layer.mu_raw.dtype,
            )
            chol_diag_x = self._inverse_softplus(torch.full((4,), 0.18, dtype=base_layer.chol_raw.dtype))
            chol_diag_y = self._inverse_softplus(torch.full((4,), 0.24, dtype=base_layer.chol_raw.dtype))
            amp = torch.tensor([[0.1], [-0.2], [0.15], [0.3]], dtype=base_layer.amp.dtype)
            for layer in (base_layer, scaled_layer):
                layer.mu_raw.copy_(mu_raw)
                layer.chol_raw.zero_()
                layer.chol_raw[:, 0] = chol_diag_x
                layer.chol_raw[:, 2] = chol_diag_y
                layer.amp.copy_(amp)
                layer.enforce_parameter_constraints_()

        base_loss = base_layer.compute_delta_weight().square().mean()
        scaled_loss = scaled_layer.compute_delta_weight().square().mean()
        base_loss.backward()
        scaled_loss.backward()

        torch.testing.assert_close(scaled_layer.mu_raw.grad, base_layer.mu_raw.grad * 10.0, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            scaled_layer.chol_raw.grad[:, [0, 2]],
            base_layer.chol_raw.grad[:, [0, 2]] * 50.0,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_initial_jitter_perturbs_mu_and_sigma(self) -> None:
        torch.manual_seed(123)
        base_config = GaussianAdapterConfig(
            init_num_gaussians=16,
            init_method="grid_overlap",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            init_amp_scale=0.0,
            init_mu_jitter_scale=0.0,
            init_sigma_log_jitter_scale=0.0,
        )
        base_config.validate()
        base_layer = GaussianLinear(
            in_features=16,
            out_features=16,
            bias=False,
            adapter_config=base_config,
        )

        torch.manual_seed(123)
        jitter_config = GaussianAdapterConfig(
            init_num_gaussians=16,
            init_method="grid_overlap",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            init_amp_scale=0.0,
            init_mu_jitter_scale=3e-3,
            init_sigma_log_jitter_scale=0.1,
        )
        jitter_config.validate()
        jitter_layer = GaussianLinear(
            in_features=16,
            out_features=16,
            bias=False,
            adapter_config=jitter_config,
        )

        base_mu = base_layer.materialize_mu()
        jitter_mu = jitter_layer.materialize_mu()
        base_sigma_x, base_sigma_y = base_layer.materialize_sigma()
        jitter_sigma_x, jitter_sigma_y = jitter_layer.materialize_sigma()

        self.assertGreater(float((jitter_mu - base_mu).detach().abs().max().item()), 0.0)
        self.assertGreater(float((jitter_sigma_x - base_sigma_x).detach().abs().max().item()), 0.0)
        self.assertGreater(float((jitter_sigma_y - base_sigma_y).detach().abs().max().item()), 0.0)
        self.assertTrue(torch.all(jitter_mu.abs() < 1.0))
        self.assertTrue(torch.all(jitter_sigma_x > 0.0))
        self.assertTrue(torch.all(jitter_sigma_y > 0.0))

    def test_weight_absmin_init_uses_smallest_weight_positions(self) -> None:
        config = GaussianAdapterConfig(
            init_num_gaussians=2,
            init_method="weight_absmin_positions",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            init_amp_scale=0.0,
            init_mu_jitter_scale=0.0,
            init_sigma_log_jitter_scale=0.0,
        )
        config.validate()

        linear = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            linear.weight.copy_(
                torch.tensor(
                    [
                        [0.5, 0.01, -0.9],
                        [0.02, 0.3, 0.4],
                    ],
                    dtype=linear.weight.dtype,
                )
            )

        layer = GaussianLinear.from_linear(linear, config)
        expected_mu = torch.tensor(
            [
                [-1.0, 0.0],
                [1.0, -1.0],
            ],
            dtype=layer.mu_raw.dtype,
        )
        torch.testing.assert_close(layer.materialize_mu(), expected_mu, rtol=1e-6, atol=1e-6)

    def test_amp_warmup_lr_is_zero_until_boundary(self) -> None:
        config = TrainingConfig(
            lr_mu=1e-4,
            lr_chol=1e-4,
            lr_amp=1e-5,
            lr_amp_warmup_steps=3,
            lr_amp_warmup_value=0.0,
        )
        config.validate()
        self.assertEqual(config.resolve_lr_amp(1), 0.0)
        self.assertEqual(config.resolve_lr_amp(3), 0.0)
        self.assertEqual(config.resolve_lr_amp(4), 1e-5)

    def test_checkpoint_semantics_are_saved_and_required(self) -> None:
        config = GaussianAdapterConfig(
            init_num_gaussians=2,
            init_method="random_uniform",
            readout_scheme="cell_average_diag_v1",
            execution_backend="torch_reference",
            gaussian_normalization="normalized_density",
            component_count_normalization="sqrt_num_gaussians",
        )
        config.validate()
        layer = GaussianLinear(
            in_features=4,
            out_features=4,
            bias=False,
            adapter_config=config,
        )
        model = torch.nn.Sequential(layer)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "adapter.pt"
            save_adapter_checkpoint(
                str(checkpoint_path),
                model,
                target_modules=["0"],
                config=config,
                step=1,
            )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            metadata = payload["metadata"]
            validate_checkpoint_semantics(metadata, expected_config=config)

        with self.assertRaises(ValueError):
            validate_checkpoint_semantics({}, expected_config=config)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for V1 CUDA backend test")
    def test_cuda_backend_matches_torch_reference_forward_and_grads(self) -> None:
        torch.manual_seed(0)

        def make_inputs():
            mu_raw = torch.randn(3, 2, device="cuda", dtype=torch.float32, requires_grad=True)
            chol_raw = torch.zeros(3, 3, device="cuda", dtype=torch.float32, requires_grad=True)
            chol_raw.data[:, 0] = self._inverse_softplus(torch.full((3,), 0.18, device="cuda"))
            chol_raw.data[:, 2] = self._inverse_softplus(torch.full((3,), 0.24, device="cuda"))
            amp = torch.randn(3, 1, device="cuda", dtype=torch.float32)
            amp.mul_(0.1)
            amp.requires_grad_()
            return mu_raw, chol_raw, amp

        mu_ref, chol_ref, amp_ref = make_inputs()
        mu_cuda, chol_cuda, amp_cuda = [tensor.detach().clone().requires_grad_(True) for tensor in (mu_ref, chol_ref, amp_ref)]

        delta_ref = gaussian_field_cell_average_diag_v1(
            mu_raw=mu_ref,
            chol_raw=chol_ref,
            amp=amp_ref,
            out_features=11,
            in_features=9,
            sigma_min=1e-3,
            compute_dtype=torch.float32,
        )
        delta_cuda = gaussian_field_cell_average_diag_v1_cuda(
            mu_raw=mu_cuda,
            chol_raw=chol_cuda,
            amp=amp_cuda,
            out_features=11,
            in_features=9,
            sigma_min=1e-3,
        )
        torch.testing.assert_close(delta_cuda.detach().cpu(), delta_ref.detach().cpu(), rtol=2e-4, atol=2e-5)

        grad = torch.randn_like(delta_ref)
        ref_loss = (delta_ref * grad).sum()
        cuda_loss = (delta_cuda * grad).sum()
        ref_loss.backward()
        cuda_loss.backward()

        torch.testing.assert_close(mu_cuda.grad.detach().cpu(), mu_ref.grad.detach().cpu(), rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(chol_cuda.grad.detach().cpu(), chol_ref.grad.detach().cpu(), rtol=4e-4, atol=4e-5)
        torch.testing.assert_close(amp_cuda.grad.detach().cpu(), amp_ref.grad.detach().cpu(), rtol=3e-4, atol=3e-5)

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(value))

    @staticmethod
    def _cell_center_point_sample_reference(
        *,
        mu_x: float,
        mu_y: float,
        sigma_x: float,
        sigma_y: float,
        amp: float,
        out_features: int,
        in_features: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x_edges = torch.linspace(-1.0, 1.0, steps=in_features + 1, dtype=dtype)
        y_edges = torch.linspace(-1.0, 1.0, steps=out_features + 1, dtype=dtype)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        x_density = torch.exp(-0.5 * ((x_centers - mu_x) / sigma_x) ** 2) / (math.sqrt(2.0 * math.pi) * sigma_x)
        y_density = torch.exp(-0.5 * ((y_centers - mu_y) / sigma_y) ** 2) / (math.sqrt(2.0 * math.pi) * sigma_y)
        return amp * y_density[:, None] * x_density[None, :]


if __name__ == "__main__":
    unittest.main()
