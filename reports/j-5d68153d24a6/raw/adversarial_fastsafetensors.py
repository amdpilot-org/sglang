import os
import tempfile
from multiprocessing import Manager
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.srt.model_loader.weight_utils import fastsafetensors_weights_iterator


def worker(rank, world_size, init_file, scenario, events, results):
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )

    class FakeBuffer:
        key_to_rank_lidx = {"weight": 0}

        def get_tensor(self, key):
            events.append((scenario, rank, "broadcast"))
            value = torch.tensor([37 if rank == 0 else -1], dtype=torch.int64)
            torch.distributed.broadcast(value, src=0)
            return value

    class FakeLoader:
        def __init__(self, group, device, nogds):
            self.nogds = nogds
            events.append((scenario, rank, "construct", nogds, str(device)))
            if scenario == "gds_constructor" and rank == 1 and not nogds:
                raise RuntimeError("is_gds_supported(0) failed")

        def add_filenames(self, rank_file_map):
            events.append((scenario, rank, "add", self.nogds))

        def copy_files_to_device(self):
            events.append((scenario, rank, "copy", self.nogds))
            if scenario == "non_gds" and rank == 1 and not self.nogds:
                raise RuntimeError("checkpoint metadata is corrupt")
            if scenario == "fallback_failure" and rank == 1 and self.nogds:
                raise RuntimeError("pread fallback failed")
            if scenario == "fallback_failure" and rank == 1 and not self.nogds:
                raise RuntimeError("is_gds_supported(0) failed")
            return FakeBuffer()

        def close(self):
            events.append((scenario, rank, "close", self.nogds))

    try:
        with (
            patch(
                "sglang.srt.model_loader.weight_utils.SafeTensorsFileLoader",
                FakeLoader,
            ),
            patch("torch.cuda.current_device", return_value=0),
        ):
            loaded = list(fastsafetensors_weights_iterator(["model.safetensors"]))
        results.append((scenario, rank, "ok", loaded[0][1].item()))
    except Exception as exc:
        results.append((scenario, rank, "error", type(exc).__name__, str(exc)))
    finally:
        torch.distributed.destroy_process_group()


def run_scenario(scenario, events, results):
    with tempfile.TemporaryDirectory() as tmpdir:
        mp.spawn(
            worker,
            args=(2, os.path.join(tmpdir, "pg"), scenario, events, results),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    with Manager() as manager:
        events = manager.list()
        results = manager.list()
        for scenario in ("gds_constructor", "non_gds", "fallback_failure"):
            run_scenario(scenario, events, results)

        actual = sorted(results)
        for item in actual:
            print("RESULT", item)
        for item in sorted(events):
            print("EVENT", item)

        assert ("gds_constructor", 0, "ok", 37) in actual
        assert ("gds_constructor", 1, "ok", 37) in actual
        assert sum(e[:3] == ("gds_constructor", 0, "broadcast") for e in events) == 1
        assert sum(e[:3] == ("gds_constructor", 1, "broadcast") for e in events) == 1

        non_gds = [x for x in actual if x[0] == "non_gds"]
        assert all(x[2] == "error" for x in non_gds), non_gds
        assert not any(e[:3] == ("non_gds", 0, "broadcast") for e in events)
        assert not any(e[:3] == ("non_gds", 1, "broadcast") for e in events)

        fallback_failure = [x for x in actual if x[0] == "fallback_failure"]
        assert all(x[2] == "error" for x in fallback_failure), fallback_failure
        assert not any(e[:3] == ("fallback_failure", 0, "broadcast") for e in events)
        assert not any(e[:3] == ("fallback_failure", 1, "broadcast") for e in events)

        print("ADVERSARIAL_ASSERTIONS_PASSED")
