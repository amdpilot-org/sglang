import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)


class FakeProfiler:
    def __init__(self, rank):
        self.rank = rank

    def export_chrome_trace(self, path):
        if self.rank == 0:
            time.sleep(1.0)
        Path(path).write_text("trace")


def worker(rank, init_file, output_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    manager = SchedulerProfilerManager.__new__(SchedulerProfilerManager)
    manager.dp_tp_cpu_group = dist.group.WORLD
    manager.merge_profiles = False
    manager.ps = SimpleNamespace(
        tp_rank=rank,
        dp_size=1,
        dp_rank=0,
        pp_size=1,
        pp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
    )
    thread = threading.Thread(
        target=manager._export_torch_trace,
        args=(FakeProfiler(rank), str(Path(output_dir) / f"rank-{rank}.json")),
        kwargs={
            "merge_profiles": False,
            "output_dir": Path(output_dir),
            "profile_id": "race",
            "stage_suffix": "",
            "expected_exports": 2,
        },
    )
    thread.start()
    if rank == 1:
        time.sleep(0.5)
    value = torch.tensor([rank + 1.0])
    dist.all_reduce(value)
    thread.join()
    print(f"rank={rank} value={value.item()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        mp.spawn(
            worker,
            args=(os.path.join(temp_dir, "init"), temp_dir),
            nprocs=2,
            join=True,
        )
