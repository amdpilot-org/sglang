import base64
import multiprocessing as mp
import queue
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=10, suite="stage-b-test-cpu-intel")


def _deserialize_rank_payload(rank, payloads, start, results):
    start.wait()
    try:
        restored = MultiprocessingSerializer.deserialize(payloads[rank])
        results.put((rank, "ok", restored["w"].numel()))
    except BaseException as exc:
        results.put((rank, type(exc).__name__, str(exc)))


class TestLoRATensorSerialization(CustomTestCase):
    def _serialize(self, tensors, tp_size, load_format=None):
        engine = object.__new__(Engine)
        parallel = types.SimpleNamespace(tp_size=tp_size)
        with patch("sglang.srt.entrypoints.engine.get_parallel", return_value=parallel):
            return engine._serialize_tensors_per_rank(tensors, load_format)

    def test_each_tp_rank_can_consume_cpu_tensor_payload(self):
        tp_size = 4
        numel = 1024 * 1024
        payloads = self._serialize(
            {"w": torch.zeros(numel, dtype=torch.bfloat16)}, tp_size
        )
        self.assertEqual(len(payloads), tp_size)

        ctx = mp.get_context("spawn")
        start = ctx.Event()
        results = ctx.Queue()
        workers = [
            ctx.Process(
                target=_deserialize_rank_payload,
                args=(rank, payloads, start, results),
            )
            for rank in range(tp_size)
        ]
        for worker in workers:
            worker.start()
        start.set()

        try:
            output = sorted(results.get(timeout=30) for _ in workers)
        except queue.Empty:
            self.fail("timed out waiting for TP payload consumers")
        finally:
            for worker in workers:
                worker.join(timeout=30)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

        self.assertEqual(output, [(rank, "ok", numel) for rank in range(tp_size)])
        self.assertTrue(all(worker.exitcode == 0 for worker in workers))

    def test_tp1_keeps_single_multiprocessing_payload(self):
        payloads = self._serialize({"w": torch.arange(8)}, tp_size=1)
        self.assertEqual(len(payloads), 1)
        restored = MultiprocessingSerializer.deserialize(payloads[0])
        torch.testing.assert_close(restored["w"], torch.arange(8))

    def test_flattened_bucket_payloads_are_normalized_without_reserialization(self):
        raw_payload = MultiprocessingSerializer.serialize({"rank": 0})
        base64_payload = base64.b64encode(
            MultiprocessingSerializer.serialize({"rank": 1})
        ).decode("ascii")

        payloads = self._serialize(
            [raw_payload, base64_payload], tp_size=2, load_format="flattened_bucket"
        )

        self.assertEqual(payloads[0], raw_payload)
        self.assertEqual(MultiprocessingSerializer.deserialize(payloads[0]), {"rank": 0})
        self.assertEqual(MultiprocessingSerializer.deserialize(payloads[1]), {"rank": 1})


if __name__ == "__main__":
    unittest.main()
