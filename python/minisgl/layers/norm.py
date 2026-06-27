import functools
import importlib
import importlib.util
from typing import Callable, Tuple

import torch
from minisgl.utils import device as accel

from .base import BaseOP


@functools.cache
def _get_torch_npu_op(name: str):
    if not accel.is_npu() or importlib.util.find_spec("torch_npu") is None:
        return None
    return getattr(importlib.import_module("torch_npu"), name, None)


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm: Callable | None = None
        self.npu_rms_norm: Callable | None = _get_torch_npu_op("npu_rms_norm")
        if accel.is_cuda():
            try:
                from flashinfer import rmsnorm
            except ImportError:
                pass
            else:
                self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.npu_rms_norm is not None:
            return self.npu_rms_norm(x, self.weight, self.eps)[0]
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
        self.npu_rms_norm: Callable | None = _get_torch_npu_op("npu_rms_norm")
        self.npu_add_rms_norm: Callable | None = _get_torch_npu_op("npu_add_rms_norm")
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
            if self.npu_rms_norm is not None:
                return self.npu_rms_norm(x, self.weight, self.eps)[0], x
            if self.rmsnorm is not None:
                return self.rmsnorm(x, self.weight, self.eps), x
            return self._native_rmsnorm(x), x
        if self.npu_add_rms_norm is not None:
            out, _, residual_out = self.npu_add_rms_norm(residual, x, self.weight, self.eps)
            return out, residual_out
        if self.fused_add_rmsnorm is not None:
            self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
        residual = residual + x
        return self._native_rmsnorm(residual), residual
