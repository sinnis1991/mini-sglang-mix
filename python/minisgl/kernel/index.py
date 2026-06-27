from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.utils import device as accel

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    if not accel.is_cuda():
        return _native_indexing(weights, indices, output=output, vocab_range=vocab_range)

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    try:
        module = _jit_index_module(element_size, num_splits=num_splits)
    except ImportError:
        return _native_indexing(weights, indices, output=output, vocab_range=vocab_range)
    module.launch(weights, indices, output, vocab_range)
    return output


def _native_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor,
    vocab_range: Tuple[int, int] | None = None,
) -> torch.Tensor:
    if vocab_range is None:
        output.copy_(F.embedding(indices.to(torch.long), weights))
        return output

    start, length = vocab_range
    local_indices = indices.to(torch.long) - start
    mask = (local_indices >= 0) & (local_indices < length)
    safe_indices = local_indices.clamp(min=0, max=max(length - 1, 0))
    gathered = F.embedding(safe_indices, weights)
    output.copy_(torch.where(mask.unsqueeze(-1), gathered, torch.zeros_like(gathered)))
    return output
