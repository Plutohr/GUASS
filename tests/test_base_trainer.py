from __future__ import annotations

import unittest

import torch
from torch import nn

from gaussian_peft.config.training import TrainingConfig
from gaussian_peft.trainers.base_trainer import BaseTrainer


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.amp = nn.Parameter(torch.zeros(2))
        self.mu = nn.Parameter(torch.zeros(2))


class _DummyTrainer(BaseTrainer):
    def compute_loss(self, batch):
        del batch
        return torch.zeros((), dtype=torch.float32)


class BaseTrainerGradClipTests(unittest.TestCase):
    def test_per_group_grad_clip_can_clip_amp_without_clipping_mu(self) -> None:
        model = _DummyModel()
        optimizer = torch.optim.AdamW(
            [
                {"name": "gaussian_amp", "params": [model.amp], "lr": 1e-3},
                {"name": "gaussian_mu", "params": [model.mu], "lr": 1e-3},
            ]
        )
        config = TrainingConfig(
            lr_mu=1e-4,
            lr_chol=1e-4,
            lr_amp=1e-5,
            max_grad_norm=None,
            max_grad_norm_amp=1.0,
        )
        trainer = _DummyTrainer(
            model=model,
            optimizer=optimizer,
            training_config=config,
            scheduler=None,
            scaler=None,
            checkpoint_dir=None,
            device="cpu",
            stats_tracker=None,
            densify_scheduler=None,
        )

        model.amp.grad = torch.tensor([3.0, 4.0])
        model.mu.grad = torch.tensor([6.0, 8.0])

        grad_norm = trainer.clip_gradients()

        self.assertEqual(grad_norm, 5.0)
        self.assertAlmostEqual(float(model.amp.grad.norm().item()), 1.0, places=6)
        self.assertAlmostEqual(float(model.mu.grad.norm().item()), 10.0, places=6)
        self.assertEqual(trainer.last_grad_clip_metrics["grad_norm_gaussian_amp"], 5.0)
        self.assertNotIn("grad_norm_gaussian_mu", trainer.last_grad_clip_metrics)


if __name__ == "__main__":
    unittest.main()
