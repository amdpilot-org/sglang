"""Exercise the reported MLP-sync collective contract on localhost.

This is transport/metadata evidence only.  It does not reproduce GLM-5.2,
H100/NCCL, or the reported two-node scheduler ordering.
"""

import argparse
import os
import socket
from multiprocessing import get_context
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    active = torch.ones(world_size, dtype=torch.int64)
    original_get_tp_group = dp_attn.get_tp_group
    dp_attn.get_tp_group = lambda: SimpleNamespace(active_ranks_cpu=active)
    try:
        info = dp_attn.MLPSyncBatchInfo(
            dp_size=2,
            tp_size=world_size // 2,
            cp_size=1,
            num_tokens=64 if rank < world_size // 2 else 0,
            num_tokens_for_logprob=1 if rank < world_size // 2 else 0,
            can_run_decode_cuda_graph=rank != 0,
            can_run_prefill_cuda_graph=False,
            is_extend_in_batch=rank < world_size // 2,
            local_can_run_tbo=True,
            local_forward_mode=(
                ForwardMode.EXTEND.value
                if rank < world_size // 2
                else ForwardMode.IDLE.value
            ),
        )
        info.all_gather(device="cpu", group=torch.distributed.group.WORLD)
        result_queue.put(
            (
                rank,
                info.global_num_tokens,
                info.global_num_tokens_for_logprob,
                info.can_run_decode_cuda_graph,
                info.is_extend_in_batch,
            )
        )
    finally:
        dp_attn.get_tp_group = original_get_tp_group
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=16)
    args = parser.parse_args()
    if args.world_size < 2 or args.world_size % 2:
        parser.error("world size must be an even integer >= 2")

    ctx = get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    processes = [
        ctx.Process(target=_worker, args=(rank, args.world_size, port, queue))
        for rank in range(args.world_size)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=60) for _ in processes]
    for process in processes:
        process.join(timeout=60)
        if process.exitcode != 0:
            raise SystemExit(f"rank process failed: pid={process.pid} exit={process.exitcode}")

    expected_tokens = [64, 0]
    expected_logprob_tokens = [1, 0]
    for rank, tokens, logprob_tokens, can_decode_graph, has_extend in results:
        assert tokens == expected_tokens, (rank, tokens)
        assert logprob_tokens == expected_logprob_tokens, (rank, logprob_tokens)
        assert can_decode_graph is False, (rank, can_decode_graph)
        assert has_extend is True, (rank, has_extend)
    print(f"PASS: {args.world_size} Gloo ranks completed heterogeneous MLP metadata sync")


if __name__ == "__main__":
    main()
