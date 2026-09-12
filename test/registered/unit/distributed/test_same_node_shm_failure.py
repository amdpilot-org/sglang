import datetime
import socket
import unittest
from multiprocessing import get_context
from unittest.mock import patch

import torch

from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_same_node_probe(rank, world_size, port, fail_creation, results):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3),
    )
    try:
        if rank == 0 and fail_creation:
            with patch(
                "sglang.srt.distributed.parallel_state.shared_memory.SharedMemory",
                side_effect=OSError(63, "File name too long"),
            ):
                results[rank] = in_the_same_node_as(
                    torch.distributed.group.WORLD, source_rank=0
                )
        else:
            results[rank] = in_the_same_node_as(
                torch.distributed.group.WORLD, source_rank=0
            )
    except Exception as error:
        results[rank] = f"{type(error).__name__}: {error}"
    finally:
        torch.distributed.destroy_process_group()


class TestSameNodeShmFailure(unittest.TestCase):
    def _run_probe(self, fail_creation):
        ctx = get_context("spawn")
        with ctx.Manager() as manager:
            results = manager.dict()
            port = _free_port()
            processes = [
                ctx.Process(
                    target=_run_same_node_probe,
                    args=(rank, 2, port, fail_creation, results),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()

            return dict(results)

    def test_source_shm_creation_failure_is_collective(self):
        self.assertEqual(
            self._run_probe(fail_creation=True),
            {0: [False, False], 1: [False, False]},
        )

    def test_successful_probe_still_detects_same_node(self):
        self.assertEqual(
            self._run_probe(fail_creation=False),
            {0: [True, True], 1: [True, True]},
        )


if __name__ == "__main__":
    unittest.main()
