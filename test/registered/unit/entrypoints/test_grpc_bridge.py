import asyncio
import enum
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ChunkStatus(enum.Enum):
    Ready = 1
    Pending = 2
    Closed = 3


class _RecordingCallback:
    def __init__(self):
        self.calls = []

    def __call__(self, payload, *, finished=False, error=None):
        self.calls.append((payload, finished, error))
        return _ChunkStatus.Ready


class _FakeTokenizerManager:
    def __init__(self, responses):
        self.responses = responses

    def generate_request(self, obj, request=None):
        async def generate():
            for response in self.responses:
                yield response

        return generate()


def _make_runtime_handle(responses):
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.tokenizer_manager = _FakeTokenizerManager(responses)
    return handle


class TestGrpcOperationalState(CustomTestCase):
    def _handle(self, *, status, paused=False, exiting=False, updating=False):
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = SimpleNamespace(
            server_status=status, is_pause=paused, gracefully_exit=exiting
        )
        handle._grpc_weight_update_in_progress = updating
        return handle

    def test_pause_update_resume_lifecycle(self):
        import json
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        cases = [
            ({}, "SERVING", True, False),
            ({"paused": True}, "DRAINING", False, True),
            ({"paused": True, "updating": True}, "UPDATING_WEIGHTS", False, True),
            ({"paused": True}, "DRAINING", False, True),
            ({}, "SERVING", True, False),
        ]
        for overrides, phase, accepting, draining in cases:
            state = json.loads(
                self._handle(status=ServerStatus.Up, **overrides).get_operational_state()
            )
            self.assertEqual(state["phase"], phase)
            self.assertEqual(state["accepting_new_requests"], accepting)
            self.assertEqual(state["ready_to_serve"], accepting)
            self.assertEqual(state["draining"], draining)
            self.assertEqual(
                self._handle(
                    status=ServerStatus.Up, **overrides
                ).health_check(),
                accepting,
            )

    def test_startup_unhealthy_and_shutdown_are_not_ready(self):
        import json
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        for status, exiting, phase in (
            (ServerStatus.Starting, False, "STARTING"),
            (ServerStatus.UnHealthy, False, "NOT_SERVING"),
            (ServerStatus.Up, True, "NOT_SERVING"),
        ):
            state = json.loads(
                self._handle(status=status, exiting=exiting).get_operational_state()
            )
            self.assertEqual(state["phase"], phase)
            self.assertFalse(state["accepting_new_requests"])
            self.assertFalse(state["ready_to_serve"])


class TestNativeGrpcParallelResponses(CustomTestCase):
    def test_non_streaming_returns_every_choice_before_finishing(self):
        callback = _RecordingCallback()
        responses = [
            [
                {"output_ids": [1], "meta_info": {"id": "choice-0"}},
                {"output_ids": [2], "meta_info": {"id": "choice-1"}},
            ]
        ]
        handle = _make_runtime_handle(responses)
        obj = SimpleNamespace(rid="logical", batch_size=1, parallel_sample_num=2)

        asyncio.run(
            handle._run_generate(
                obj,
                callback,
                stream=False,
                request=None,
            )
        )

        self.assertEqual([call[0]["output_ids"] for call in callback.calls], [[1], [2]])
        self.assertEqual([call[1] for call in callback.calls], [False, True])

    def test_streaming_first_finished_choice_is_not_batch_terminal(self):
        callback = _RecordingCallback()
        responses = [
            {
                "index": 0,
                "output_ids": [1],
                "meta_info": {"id": "choice-0", "finish_reason": None},
            },
            {
                "index": 0,
                "output_ids": [2],
                "meta_info": {
                    "id": "choice-0",
                    "finish_reason": {"type": "stop"},
                },
            },
            {
                "index": 1,
                "output_ids": [3],
                "meta_info": {
                    "id": "choice-1",
                    "finish_reason": {"type": "stop"},
                },
            },
        ]
        handle = _make_runtime_handle(responses)
        obj = SimpleNamespace(rid="logical", sampling_params={"n": 2})

        asyncio.run(
            handle._run_generate(
                obj,
                callback,
                stream=True,
                request=None,
            )
        )

        self.assertEqual(
            [call[0]["output_ids"] for call in callback.calls],
            [[1], [2], [3]],
        )
        self.assertEqual([call[1] for call in callback.calls], [False, False, True])


if __name__ == "__main__":
    unittest.main(verbosity=2)
