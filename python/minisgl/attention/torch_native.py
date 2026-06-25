from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class TorchNativeMetadata(BaseAttnMetadata):
    seqlens_q: List[int]
    seqlens_k: List[int]
    cu_seqlens_q: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TorchNativeBackend(BaseAttnBackend):
    """Portable PyTorch attention backend used for non-CUDA accelerators.

    This backend favors compatibility over performance. It avoids FlashInfer,
    FlashAttention, TRT-LLM and custom CUDA kernels so Ascend NPU deployments can
    get through model execution with standard PyTorch/torch_npu operators.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cu_seqlens_q = torch.tensor([0] + seqlens_q, dtype=torch.int32, device=self.device).cumsum_(
            0
        )
        batch.attn_metadata = TorchNativeMetadata(
            seqlens_q=seqlens_q,
            seqlens_k=seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchNativeMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_cache = self.kvcache.k_cache(layer_id).view(-1, self.kv_head_local, self.head_dim)
        v_cache = self.kvcache.v_cache(layer_id).view(-1, self.kv_head_local, self.head_dim)
        page_table = get_global_ctx().page_table
        outputs: list[torch.Tensor] = []
        q_offset = 0
        group_size = self.qo_head_local // self.kv_head_local

        for req, q_len, k_len in zip(batch.padded_reqs, metadata.seqlens_q, metadata.seqlens_k):
            q_req = q[q_offset : q_offset + q_len]
            q_offset += q_len
            if q_len == 0:
                continue

            indices = page_table[req.table_idx, :k_len].to(torch.long)
            k_req = k_cache.index_select(0, indices)
            v_req = v_cache.index_select(0, indices)
            if group_size != 1:
                k_req = k_req.repeat_interleave(group_size, dim=1)
                v_req = v_req.repeat_interleave(group_size, dim=1)

            # [heads, query_len, key_len]
            scores = torch.einsum("qhd,khd->hqk", q_req, k_req) * self.scale
            if batch.is_prefill:
                cached_len = k_len - q_len
                q_pos = torch.arange(q_len, device=q.device).unsqueeze(1)
                k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
                causal_mask = k_pos <= (cached_len + q_pos)
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            probs = torch.softmax(scores.float(), dim=-1).to(v_req.dtype)
            out = torch.einsum("hqk,khd->qhd", probs, v_req)
            outputs.append(out)

        if not outputs:
            return q.new_empty((0, self.qo_head_local, self.head_dim))
        return torch.cat(outputs, dim=0)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        return None

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.prepare_metadata(batch)
