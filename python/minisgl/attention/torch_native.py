from __future__ import annotations

import importlib
import importlib.util
import math
from dataclasses import dataclass
from typing import Callable, List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseAttnBackend, BaseAttnMetadata


def _get_npu_fused_infer_attention() -> Callable | None:
    if importlib.util.find_spec("torch_npu") is None:
        return None
    torch_npu = importlib.import_module("torch_npu")
    npu_fia = getattr(torch_npu, "npu_fused_infer_attention_score", None)
    if npu_fia is not None:
        return npu_fia
    return getattr(getattr(torch.ops, "npu", None), "npu_fused_infer_attention_score_v2", None)


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
        self.npu_fused_infer_attention = _get_npu_fused_infer_attention()
        self.npu_fused_infer_attention_disabled = False

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

        if (
            self.npu_fused_infer_attention is not None
            and not self.npu_fused_infer_attention_disabled
        ):
            try:
                return self._forward_npu_fused_attention(q, layer_id, batch, metadata)
            except Exception:
                self.npu_fused_infer_attention_disabled = True

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

    def _forward_npu_fused_attention(
        self,
        q: torch.Tensor,
        layer_id: int,
        batch: Batch,
        metadata: TorchNativeMetadata,
    ) -> torch.Tensor:
        assert self.npu_fused_infer_attention is not None
        bs = len(batch.padded_reqs)
        max_q_len = max(metadata.seqlens_q, default=0)
        max_k_len = max(metadata.seqlens_k, default=0)
        if bs == 0 or max_q_len == 0:
            return q.new_empty((0, self.qo_head_local, self.head_dim))

        k_cache = self.kvcache.k_cache(layer_id).view(-1, self.kv_head_local, self.head_dim)
        v_cache = self.kvcache.v_cache(layer_id).view(-1, self.kv_head_local, self.head_dim)
        page_table = get_global_ctx().page_table

        q_padded = q.new_zeros((bs, max_q_len, self.qo_head_local, self.head_dim))
        k_padded = q.new_zeros((bs, max_k_len, self.kv_head_local, self.head_dim))
        v_padded = q.new_zeros((bs, max_k_len, self.kv_head_local, self.head_dim))
        attn_mask = None
        if batch.is_prefill:
            attn_mask = torch.ones((bs, 1, max_q_len, max_k_len), dtype=torch.bool, device=q.device)

        q_offset = 0
        for b, (req, q_len, k_len) in enumerate(
            zip(batch.padded_reqs, metadata.seqlens_q, metadata.seqlens_k)
        ):
            if q_len > 0:
                q_padded[b, :q_len].copy_(q[q_offset : q_offset + q_len])
            q_offset += q_len

            indices = page_table[req.table_idx, :k_len].to(torch.long)
            k_padded[b, :k_len].copy_(k_cache.index_select(0, indices))
            v_padded[b, :k_len].copy_(v_cache.index_select(0, indices))

            if attn_mask is not None:
                cached_len = k_len - q_len
                q_pos = torch.arange(q_len, device=q.device).unsqueeze(1)
                k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
                attn_mask[b, 0, :q_len, :k_len] = k_pos > (cached_len + q_pos)

        kwargs = {
            "num_heads": self.qo_head_local,
            "num_key_value_heads": self.kv_head_local,
            "input_layout": "BSND",
            "atten_mask": attn_mask,
            "actual_seq_lengths": metadata.seqlens_q,
            "actual_seq_lengths_kv": metadata.seqlens_k,
        }
        try:
            attn_output, _ = self.npu_fused_infer_attention(
                q_padded,
                k_padded,
                v_padded,
                scale=self.scale,
                **kwargs,
            )
        except TypeError:
            attn_output, _ = self.npu_fused_infer_attention(
                q_padded,
                k_padded,
                v_padded,
                scale_value=self.scale,
                **kwargs,
            )

        outputs = [
            attn_output[b, :q_len] for b, q_len in enumerate(metadata.seqlens_q) if q_len > 0
        ]
        if not outputs:
            return q.new_empty((0, self.qo_head_local, self.head_dim))
        return torch.cat(outputs, dim=0)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        return None

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.prepare_metadata(batch)
