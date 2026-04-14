from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn

from gaussian_peft.config.adapter import GaussianAdapterConfig


class GaussianAdapterBase(ABC):
    """Shared adapter surface for Gaussian-based layers."""

    def __init__(self, adapter_config: GaussianAdapterConfig) -> None:
        adapter_config.validate()
        self.adapter_config = adapter_config
        self.merged = False

    def get_adapter_config(self) -> GaussianAdapterConfig:
        return self.adapter_config

    @property
    @abstractmethod
    def num_gaussians(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_gaussian_parameters(self) -> dict[str, nn.Parameter]:
        raise NotImplementedError

    def get_adapter_state(self) -> dict[str, Tensor]:
        return {
            name: param.detach().clone()
            for name, param in self.get_gaussian_parameters().items()
        }

    @abstractmethod
    def load_adapter_state(self, state: dict[str, Tensor]) -> None:
        raise NotImplementedError

    @abstractmethod
    def replace_gaussian_parameters_(self, **kwargs: Tensor) -> None:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return (
            f"num_gaussians={self.num_gaussians}, "
            f"covariance_type={self.adapter_config.covariance_type}, "
            f"chunk_size={self.adapter_config.chunk_size}"
        )

    def get_serialization_metadata(self) -> dict[str, Any]:
        return {
            "num_gaussians": self.num_gaussians,
            "covariance_type": self.adapter_config.covariance_type,
            "coord_norm": self.adapter_config.coord_norm,
        }
