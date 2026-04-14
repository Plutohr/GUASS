from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaussian_peft.config.adapter import GaussianAdapterConfig
from gaussian_peft.patchers.replace_linear import apply_gaussian_peft
from gaussian_peft.layers.gaussian_linear import GaussianLinear


class TinyAttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x)
        v = self.to_v(x)
        return q + v


def collect_gaussian_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, GaussianLinear):
            bundle = module.get_gaussian_parameters()
            params.extend(bundle.values())
    return params


def count_gaussian_modules(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, GaussianLinear))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-gaussians", type=int, default=8)
    parser.add_argument("--tile-out", type=int, default=32)
    parser.add_argument("--tile-in", type=int, default=32)
    parser.add_argument("--sigma-multiplier", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke training script.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float32

    print("smoke: build_model", flush=True)
    model = TinyAttentionBlock(args.dim).to(device=device, dtype=dtype)
    config = GaussianAdapterConfig(
        init_num_gaussians=args.num_gaussians,
        init_method="grid_overlap",
        execution_mode="cuda_field_train",
        tile_out=args.tile_out,
        tile_in=args.tile_in,
        sigma_multiplier=args.sigma_multiplier,
        compute_dtype=torch.float32,
        init_amp_scale=1e-7,
    )
    print("smoke: apply_gaussian_peft", flush=True)
    model, replaced = apply_gaussian_peft(
        model,
        target_modules=["to_q", "to_v"],
        adapter_config=config,
        freeze_base=True,
    )
    print("smoke: collect_params", flush=True)
    gaussian_params = collect_gaussian_params(model)
    if not gaussian_params:
        raise RuntimeError("No Gaussian parameters were found after patching.")

    print("smoke: build_optimizer", flush=True)
    optimizer = torch.optim.AdamW(gaussian_params, lr=args.lr, weight_decay=0.0)

    losses: list[float] = []
    grad_norms: list[float] = []
    step_times_ms: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.steps):
        print(f"smoke: step_{step}_start", flush=True)
        x = torch.randn(args.batch, args.seq_len, args.dim, device=device, dtype=dtype)
        target = torch.randn(args.batch, args.seq_len, args.dim, device=device, dtype=dtype)

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        print(f"smoke: step_{step}_forward", flush=True)
        out = model(x)
        loss = F.mse_loss(out, target)
        print(f"smoke: step_{step}_backward", flush=True)
        loss.backward()
        print(f"smoke: step_{step}_optimizer", flush=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        step_times_ms.append((time.perf_counter() - start) * 1000.0)

        total_grad_sq = 0.0
        for param in gaussian_params:
            if param.grad is not None:
                total_grad_sq += float(param.grad.detach().float().pow(2).sum().item())
        grad_norms.append(total_grad_sq ** 0.5)
        losses.append(float(loss.detach().item()))

    first_gaussian = next(module for module in model.modules() if isinstance(module, GaussianLinear))
    metadata = dict(first_gaussian.last_forward_metadata or {})
    report = {
        "execution_mode": config.execution_mode,
        "patched_modules": len(replaced),
        "gaussian_modules": count_gaussian_modules(model),
        "steps": args.steps,
        "losses": losses,
        "grad_norms": grad_norms,
        "step_times_ms": step_times_ms,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
        "first_module_metadata": metadata,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
