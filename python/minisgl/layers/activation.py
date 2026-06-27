from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from minisgl.utils import device as accel

if TYPE_CHECKING:
    import torch


def _split_gate_up(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.chunk(2, dim=-1)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if accel.is_cuda():
        try:
            from flashinfer import silu_and_mul as flashinfer_silu_and_mul
        except ImportError:
            pass
        else:
            return flashinfer_silu_and_mul(x, out=out)

    gate, up = _split_gate_up(x)
    result = F.silu(gate) * up
    if out is not None:
        out.copy_(result)
        return out
    return result


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if accel.is_cuda():
        try:
            from flashinfer import gelu_and_mul as flashinfer_gelu_and_mul
        except ImportError:
            pass
        else:
            return flashinfer_gelu_and_mul(x, out=out)

    gate, up = _split_gate_up(x)
    result = F.gelu(gate) * up
    if out is not None:
        out.copy_(result)
        return out
    return result


__all__ = ["silu_and_mul", "gelu_and_mul"]
