from typing import Callable, Tuple

import torch
from minisgl.utils import device as accel

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm: Callable | None = None
        if accel.is_cuda():
            try:
                from flashinfer import rmsnorm
            except ImportError:
                pass
            else:
                self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rmsnorm is not None:
            return self.rmsnorm(x, self.weight, self.eps)
        return (
            x
            * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
            * self.weight
        )

    def forward_inplace(self, x: torch.Tensor) -> None:
        if self.rmsnorm is not None:
            self.rmsnorm(x, self.weight, self.eps, out=x)
        else:
            x.copy_(self.forward(x))


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm: Callable | None = None
        self.fused_add_rmsnorm: Callable | None = None
        if accel.is_cuda():
            try:
                from flashinfer import fused_add_rmsnorm, rmsnorm
            except ImportError:
                pass
            else:
                self.rmsnorm = rmsnorm
                self.fused_add_rmsnorm = fused_add_rmsnorm

    def _native_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
            * self.weight
        )

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            if self.rmsnorm is not None:
                return self.rmsnorm(x, self.weight, self.eps), x
            return self._native_rmsnorm(x), x
        if self.fused_add_rmsnorm is not None:
            self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
        residual = residual + x
        return self._native_rmsnorm(residual), residual
