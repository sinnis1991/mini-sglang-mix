from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Set

import torch
from minisgl.core import Req
from minisgl.message import DetokenizeMsg
from minisgl.utils import device as accel

if TYPE_CHECKING:
    from .scheduler import Scheduler


@dataclass
class SampleManager:
    """Tracks requests whose logits are ready and samples them before model forward."""

    pending_reqs: Set[Req] = field(default_factory=set)

    def add_reqs(self, reqs: Iterable[Req]) -> None:
        for req in reqs:
            if req.logits is not None:
                self.pending_reqs.add(req)

    def remove_req(self, req: Req) -> None:
        self.pending_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.pending_reqs:
            if req.uid == uid:
                self.pending_reqs.remove(req)
                return req
        return None

    def sample_first(self, scheduler: Scheduler) -> None:
        if not self.runnable:
            return

        reqs = sorted(self.pending_reqs, key=lambda req: req.uid)
        self.pending_reqs.clear()
        req_logits = [req.logits for req in reqs]
        assert all(logit is not None for logit in req_logits)
        logits = torch.cat(req_logits, dim=0)
        for req in reqs:
            req.logits = None

        sample_args = scheduler.engine.sampler.prepare_reqs(reqs)
        next_tokens_gpu = scheduler.engine.sampler.sample(logits, sample_args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done = accel.Event()
        copy_done.record(accel.current_stream())
        copy_done.synchronize()

        mapping = torch.tensor(
            [req.table_idx for req in reqs], dtype=torch.int64, device=scheduler.device
        )
        seq_lens = torch.tensor(
            [req.device_len - 1 for req in reqs], dtype=torch.int64, device=scheduler.device
        )
        scheduler.token_pool[(mapping, seq_lens)] = next_tokens_gpu

        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with scheduler.cache_manager.lazy_free_region():
            for i, req in enumerate(reqs):
                next_token_tensor = next_tokens_cpu[i]
                req.append_host(next_token_tensor.unsqueeze(0))
                next_token = int(next_token_tensor.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == scheduler.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if finished and req not in scheduler.finished_reqs:
                    scheduler.decode_manager.remove_req(req)
                    scheduler._free_req_resources(req)
                    new_finished_reqs.add(req)
                else:
                    scheduler.decode_manager.add_req(req)
                    scheduler.cache_manager.cache_req(req, finished=False)

        scheduler.finished_reqs = new_finished_reqs
        scheduler.send_result(reply)

    @property
    def runnable(self) -> bool:
        return len(self.pending_reqs) > 0
