from __future__ import annotations

import functools
import importlib
import importlib.util
from typing import TYPE_CHECKING

import torch.nn.functional as F
from minisgl.utils import device as accel

if TYPE_CHECKING:
    import torch


@functools.cache
def _get_torch_npu_op(name: str):
    if not accel.is_npu() or importlib.util.find_spec("torch_npu") is None:
        return None
    return getattr(importlib.import_module("torch_npu"), name, None)


def _split_gate_up(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.chunk(2, dim=-1)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    npu_swiglu = _get_torch_npu_op("npu_swiglu")
    if npu_swiglu is not None:
        result = npu_swiglu(x, dim=-1)
        if out is not None:
            out.copy_(result)
            return out
        return result

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
