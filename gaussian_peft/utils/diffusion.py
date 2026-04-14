from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


def build_gaussian_peft_param_groups(
    gaussian_layers: Iterable[nn.Module],
    *,
    lr_amp: float = 1e-3,
    lr_mu: float = 1e-4,
    lr_cov: float = 1e-5,
    lr_other: float | None = None,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """
    为 Gaussian-PEFT 构建适合 Stable Diffusion 微调的 AdamW 参数组。

    分组原则：
    - amp / w:
      决定“改多大”，通常可以用更高学习率，让适配器更快学到有效幅值。
    - mu:
      决定“去哪里改”，需要有一定移动能力，但不能像 amp 那样过猛。
    - chol_raw:
      当前实现里同时编码尺度与形状，自然属于最保守的一组。
    - other:
      例如你允许训练 bias 时，会落到这一组；默认沿用 amp 的学习率。

    这个函数只负责参数分组，不直接实例化优化器，方便你接入 accelerate /
    diffusers / 自定义 trainer。
    """

    mu_params: list[nn.Parameter] = []
    cov_params: list[nn.Parameter] = []
    amp_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    for layer in gaussian_layers:
        for name, param in layer.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name == "amp":
                amp_params.append(param)
            elif name == "mu_raw":
                mu_params.append(param)
            elif name == "chol_raw":
                cov_params.append(param)
            else:
                other_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if amp_params:
        param_groups.append(
            {
                "name": "gaussian_amp",
                "params": amp_params,
                "lr": lr_amp,
                "weight_decay": weight_decay,
            }
        )
    if mu_params:
        param_groups.append(
            {
                "name": "gaussian_mu",
                "params": mu_params,
                "lr": lr_mu,
                "weight_decay": weight_decay,
            }
        )
    if cov_params:
        param_groups.append(
            {
                "name": "gaussian_cov",
                "params": cov_params,
                "lr": lr_cov,
                "weight_decay": weight_decay,
            }
        )
    if other_params:
        param_groups.append(
            {
                "name": "gaussian_other",
                "params": other_params,
                "lr": lr_amp if lr_other is None else lr_other,
                "weight_decay": weight_decay,
            }
        )
    return param_groups


def build_gaussian_peft_adamw(
    gaussian_layers: Iterable[nn.Module],
    *,
    lr_amp: float = 1e-3,
    lr_mu: float = 1e-4,
    lr_cov: float = 1e-5,
    lr_other: float | None = None,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    基于上面的参数分组，直接创建 AdamW。

    这版默认值就是一个“可直接拿去接 diffusers 微调”的起点：
    - amp: 1e-3
    - mu: 1e-4
    - cov: 1e-5
    """

    param_groups = build_gaussian_peft_param_groups(
        gaussian_layers,
        lr_amp=lr_amp,
        lr_mu=lr_mu,
        lr_cov=lr_cov,
        lr_other=lr_other,
        weight_decay=weight_decay,
    )
    return torch.optim.AdamW(param_groups, betas=betas, eps=eps)


def default_sd_densify_config() -> dict[str, Any]:
    """
    Stable Diffusion 微调阶段的演化超参数建议。

    说明：
    - densify_interval:
      不建议太频繁，100~200 步是一个比较务实的起点。
    - grad_threshold:
      给一个保守静态阈值，适合作为 fallback。
    - prune_opacity_threshold:
      对应幅值绝对值的下限，过小就说明这个高斯长期几乎不起作用。
    - dynamic_threshold:
      如果启用，就用最近 N 步的历史梯度均值乘一个系数，得到动态阈值。
    """

    return {
        "densify_interval": 200,
        "grad_threshold": 5e-4,
        "prune_opacity_threshold": 1e-5,
        "dynamic_threshold": {
            "enabled": True,
            "window_size": 100,
            "scale": 1.25,
            "min_threshold": 1e-5,
        },
    }


@dataclass(slots=True)
class RollingGradThreshold:
    """
    基于最近若干步位置梯度统计，动态估计 densify 的触发阈值。

    用法：
    1. 每个训练 step 的 backward 之后，把当前所有 Gaussian 层的 mu 梯度范数喂进来；
    2. 调用 current_threshold() 得到一个平滑的、保守的 densify 阈值。
    """

    window_size: int = 100
    scale: float = 1.25
    min_threshold: float = 1e-5
    _history: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.min_threshold <= 0:
            raise ValueError("min_threshold must be positive")
        self._history: deque[float] = deque(maxlen=self.window_size)

    def update_from_layers(self, gaussian_layers: Iterable[nn.Module]) -> float:
        """
        从当前各层的 mu 梯度里提取一个标量统计，并加入滑动窗口。

        这里采用“所有高斯层 mu 梯度范数的均值”，而不是最大值：
        - 均值更稳，不容易被单个异常层拉爆；
        - 对大模型微调更适合作为全局 densify 门槛。
        """

        values: list[float] = []
        for layer in gaussian_layers:
            mu_grad = getattr(layer, "mu_raw", None)
            if mu_grad is None or mu_grad.grad is None:
                continue
            value = float(mu_grad.grad.detach().norm().item())
            if not torch.isfinite(torch.tensor(value)):
                continue
            values.append(value)

        if values:
            self._history.append(sum(values) / len(values))
        return self.current_threshold()

    def current_threshold(self, fallback: float | None = None) -> float:
        if not self._history:
            return self.min_threshold if fallback is None else max(self.min_threshold, fallback)
        mean_value = sum(self._history) / len(self._history)
        return max(mean_value * self.scale, self.min_threshold)


def collect_gaussian_layers(model: nn.Module) -> list[nn.Module]:
    """
    从任意模型中抽取 Gaussian-PEFT 层。

    不强依赖具体类型导入，而是通过核心参数名识别，
    这样接第三方模型 patch 后更稳。
    """

    layers: list[nn.Module] = []
    for module in model.modules():
        if all(hasattr(module, name) for name in ("mu_raw", "chol_raw", "amp")):
            layers.append(module)
    return layers


def compute_prune_mask_from_amp(amp: Tensor, threshold: float) -> Tensor:
    """
    按幅值绝对值做最简单的剪枝判定。

    这不是完整 prune 控制器，只是一个对接训练循环时很常见的“先验过滤器”。
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    score = amp.detach().abs().reshape(amp.shape[0], -1).mean(dim=1)
    return score >= threshold
