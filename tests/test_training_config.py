from __future__ import annotations

import unittest

from gaussian_peft.config.training import TrainingConfig


class TrainingConfigTests(unittest.TestCase):
    def test_piecewise_learning_rate_resolution(self) -> None:
        config = TrainingConfig(
            lr_mu=5e-5,
            lr_chol=5e-5,
            lr_amp=5e-6,
            lr_amp_warmup_steps=200,
            lr_amp_warmup_value=1e-6,
            lr_mu_schedule_steps=[3000, 4000],
            lr_mu_schedule_values=[5e-5, 1e-5],
            lr_chol_schedule_steps=[3000, 4000],
            lr_chol_schedule_values=[5e-5, 1e-5],
            lr_amp_schedule_steps=[3000, 4000],
            lr_amp_schedule_values=[5e-6, 1e-6],
        )
        config.validate()

        self.assertEqual(config.resolve_lr_mu(1), 5e-5)
        self.assertEqual(config.resolve_lr_mu(3500), 1e-5)
        self.assertEqual(config.resolve_lr_chol(1000), 5e-5)
        self.assertEqual(config.resolve_lr_chol(3500), 1e-5)
        self.assertEqual(config.resolve_lr_amp(1), 1e-6)
        self.assertEqual(config.resolve_lr_amp(200), 1e-6)
        self.assertEqual(config.resolve_lr_amp(201), 5e-6)
        self.assertEqual(config.resolve_lr_amp(3500), 1e-6)

    def test_group_grad_clip_resolution(self) -> None:
        config = TrainingConfig(
            lr_mu=1e-4,
            lr_chol=1e-4,
            lr_amp=1e-5,
            max_grad_norm=None,
            max_grad_norm_amp=1.0,
            max_grad_norm_chol=5.0,
        )
        config.validate()

        self.assertTrue(config.uses_per_group_grad_clipping())
        self.assertEqual(config.resolve_max_grad_norm("gaussian_amp"), 1.0)
        self.assertEqual(config.resolve_max_grad_norm("gaussian_cov"), 5.0)
        self.assertIsNone(config.resolve_max_grad_norm("gaussian_mu"))
        self.assertIsNone(config.resolve_max_grad_norm("gaussian_other"))


if __name__ == "__main__":
    unittest.main()
