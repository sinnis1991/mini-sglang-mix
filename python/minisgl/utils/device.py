from __future__ import annotations

"""Small accelerator runtime wrapper for CUDA and Ascend NPU backends."""

from contextlib import nullcontext
from typing import Any

import torch


def _load_npu_backend() -> Any | None:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return None
    return getattr(torch, "npu", None)


def _cuda_runtime_available() -> bool:
    return hasattr(torch._C, "_cuda_setDevice") and torch.cuda.is_available()


_ACCELERATOR = torch.cuda if _cuda_runtime_available() else _load_npu_backend()
_DEVICE_TYPE = (
    "cuda" if _ACCELERATOR is torch.cuda else ("npu" if _ACCELERATOR is not None else "cpu")
)


def accelerator() -> Any:
    if _ACCELERATOR is None:
        raise RuntimeError(
            "No supported accelerator found. Expected CUDA or Ascend NPU (torch_npu)."
        )
    return _ACCELERATOR


def device_type() -> str:
    return _DEVICE_TYPE


def is_cuda() -> bool:
    return _DEVICE_TYPE == "cuda"


def is_npu() -> bool:
    return _DEVICE_TYPE == "npu"


def make_device(index: int | None = None) -> torch.device:
    if index is None:
        return torch.device(_DEVICE_TYPE)
    return torch.device(f"{_DEVICE_TYPE}:{index}")


def set_device(device: torch.device) -> None:
    accelerator().set_device(device)


def Stream(*args, **kwargs):  # noqa: N802 - mirrors torch.cuda.Stream
    return accelerator().Stream(*args, **kwargs)


def Event(*args, **kwargs):  # noqa: N802 - mirrors torch.cuda.Event
    return accelerator().Event(*args, **kwargs)


def stream(stream_obj):
    return accelerator().stream(stream_obj)


def current_stream():
    return accelerator().current_stream()


def set_stream(stream_obj) -> None:
    accelerator().set_stream(stream_obj)


def synchronize(device: torch.device | None = None) -> None:
    accelerator().synchronize(device)


def empty_cache() -> None:
    accelerator().empty_cache()


def reset_peak_memory_stats(device: torch.device | None = None) -> None:
    reset = getattr(accelerator(), "reset_peak_memory_stats", None)
    if reset is not None:
        reset(device)


def mem_get_info(device: torch.device | None = None) -> tuple[int, int]:
    get_info = getattr(accelerator(), "mem_get_info", None)
    if get_info is None:
        raise RuntimeError(f"Memory query is not supported on {_DEVICE_TYPE}")
    return get_info(device)


def nvtx_range(_name: str):
    if is_npu():
        return nullcontext()
    import torch.cuda.nvtx as nvtx

    return nvtx.range(_name)
