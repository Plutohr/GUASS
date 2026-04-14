from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class GridInitState:
    """
    网格初始化结果。

    - mu_unit:      [0, 1] x [0, 1] 连续平面上的 cell-center 网格中心
    - mu_model:     映射到当前模型内部坐标系后的中心点
    - chol_raw:     对应 Cholesky 参数的 raw 表示，适合直接放进现有 GaussianLinear
    - amp:          幅值初始化，默认接近 0，避免微调初期破坏底座输出
    - grid_shape:   真实采用的二维网格尺寸
    - sigma:        根据网格间距自动估计出的各向同性高斯标准差
    """

    mu_unit: Tensor
    mu_model: Tensor
    chol_raw: Tensor
    amp: Tensor
    grid_shape: tuple[int, int]
    sigma: Tensor


class GridInitializer:
    """
    针对大矩阵适配器的高级网格初始化器。

    设计目标：
    1. 不用随机中心点，而是把高斯核中心均匀铺满整个二维权重平面；
    2. 根据网格间距自动估计初始尺度，让相邻高斯一开始就有平滑交叠；
    3. 幅值置零或极小噪声，使初始 Delta W 接近 0，降低微调初期扰动。
    """

    DEFAULT_NUM_GAUSSIANS = 1000

    @classmethod
    def default_num_gaussians(cls) -> int:
        """
        推荐的默认高斯数量。

        这里直接固定成 1000，目的是更贴近 Stable Diffusion
        Cross-Attention 这类大矩阵适配时的参数规模。
        """

        return cls.DEFAULT_NUM_GAUSSIANS

    @classmethod
    def initialize(
        cls,
        *,
        num_gaussians: int | None,
        device: torch.device | str,
        dtype: torch.dtype,
        eps: float,
        amp_scale: float = 1e-5,
        min_cov_diag: float = 1e-4,
        coord_range: tuple[float, float] = (-1.0, 1.0),
        chol_scale_multiplier: float = 1.5,
    ) -> GridInitState:
        """
        生成一组适合 GaussianLinear 的网格初始化参数。

        参数说明：
        - num_gaussians:
          目标高斯核数量；如果不给，就使用推荐默认值 1000。
        - coord_range:
          当前模型内部坐标范围。现有 GaussianLinear 使用 [-1, 1]。
          但为了让初始化逻辑更直观，这里仍先在 [0, 1] 平面采样，再映射过去。
        - chol_scale_multiplier:
          K 自适应初始化的尺度倍率。默认采用 1.5，
          对应经验公式 sigma ~= coord_span / sqrt(K) * 1.5。
        """

        k = int(num_gaussians or cls.default_num_gaussians())
        if k <= 0:
            raise ValueError("num_gaussians must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_cov_diag <= 0:
            raise ValueError("min_cov_diag must be positive")
        if amp_scale < 0:
            raise ValueError("amp_scale must be non-negative")
        if chol_scale_multiplier <= 0:
            raise ValueError("chol_scale_multiplier must be positive")

        grid_h, grid_w = cls._choose_grid_shape(k)
        mu_unit = cls._build_unit_grid(
            num_gaussians=k,
            grid_h=grid_h,
            grid_w=grid_w,
            device=device,
            dtype=dtype,
        )

        # 先在 [0, 1] 中均匀放点，再映射到当前模型的内部坐标域。
        coord_min, coord_max = coord_range
        coord_span = coord_max - coord_min
        mu_model = mu_unit * coord_span + coord_min

        sigma = cls._estimate_sigma(
            num_gaussians=k,
            coord_span=coord_span,
            chol_scale_multiplier=chol_scale_multiplier,
            min_cov_diag=min_cov_diag,
            device=device,
            dtype=dtype,
        )
        chol_raw = cls._build_isotropic_cholesky_raw(
            num_gaussians=k,
            sigma=sigma,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        amp = cls._build_amplitude(
            num_gaussians=k,
            amp_scale=amp_scale,
            device=device,
            dtype=dtype,
        )
        return GridInitState(
            mu_unit=mu_unit,
            mu_model=mu_model,
            chol_raw=chol_raw,
            amp=amp,
            grid_shape=(grid_h, grid_w),
            sigma=sigma,
        )

    @staticmethod
    def _choose_grid_shape(num_gaussians: int) -> tuple[int, int]:
        """
        选一个尽量接近正方形的二维网格。

        例如：
        - 1000 -> 32 x 32 后截断到 1000 个点
        - 768 -> 28 x 28 后截断到 768 个点
        """

        grid_h = math.ceil(math.sqrt(num_gaussians))
        grid_w = math.ceil(num_gaussians / grid_h)
        return grid_h, grid_w

    @staticmethod
    def _build_unit_grid(
        *,
        num_gaussians: int,
        grid_h: int,
        grid_w: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        在 [0, 1] x [0, 1] 上生成均匀网格中心点。

        这里采用 cell-center 采样，不直接落在边界 0/1 上。
        这样映射到 [-1, 1] 后不会正好落在 tanh 的饱和边缘，
        能减轻边缘高斯中心位置的初始梯度压缩。
        """

        x = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / grid_h
        y = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / grid_w
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)[:num_gaussians]

    @staticmethod
    def _estimate_sigma(
        *,
        num_gaussians: int,
        coord_span: float,
        chol_scale_multiplier: float,
        min_cov_diag: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        根据 K 自动估算一个各向同性 sigma。

        核心直觉：
        - 整个二维坐标域边长是 coord_span；
        - K 个高斯均匀铺开时，平均线性间距约为 coord_span / sqrt(K)；
        - 再乘一个经验倍率，让相邻高斯一开始具有适度重叠。
        """
        sigma_value = max(
            coord_span / math.sqrt(max(num_gaussians, 1)) * chol_scale_multiplier,
            math.sqrt(min_cov_diag),
        )
        return torch.tensor(sigma_value, device=device, dtype=dtype)

    @staticmethod
    def _build_isotropic_cholesky_raw(
        *,
        num_gaussians: int,
        sigma: Tensor,
        eps: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        生成各向同性协方差对应的 Cholesky raw 参数。

        当前实现使用 2x2 下三角 Cholesky：
        [ l11   0 ]
        [ l21  l22 ]

        这里初始化成对角形式，l11 = l22 = sigma，l21 = 0。
        """

        chol_raw = torch.zeros(num_gaussians, 3, device=device, dtype=dtype)
        sigma_clamped = sigma.clamp_min(eps)
        raw_diag = GridInitializer._inverse_softplus(sigma_clamped - eps)
        chol_raw[:, 0] = raw_diag
        chol_raw[:, 2] = raw_diag
        return chol_raw

    @staticmethod
    def _build_amplitude(
        *,
        num_gaussians: int,
        amp_scale: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        幅值初始化。

        - 当 amp_scale = 0 时，完全零初始化，最稳；
        - 当 amp_scale 很小时，给一个极小高斯噪声，帮助打破完全对称。
        """

        amp = torch.zeros(num_gaussians, 1, device=device, dtype=dtype)
        if amp_scale > 0:
            amp.normal_(mean=0.0, std=amp_scale)
        return amp

    @staticmethod
    def _inverse_softplus(value: Tensor) -> Tensor:
        return torch.log(torch.expm1(value))
