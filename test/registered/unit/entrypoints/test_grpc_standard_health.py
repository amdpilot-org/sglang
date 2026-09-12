import socket
import time
import unittest
from types import SimpleNamespace

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

try:
    from sglang.srt.rust_extensions import _grpc
except ImportError:
    _grpc = None


class _RuntimeHandle:
    def __init__(self):
        self.healthy = True
        self.tokenizer_manager = SimpleNamespace(
            server_args=SimpleNamespace(
                tokenizer_path=None,
                model_path=None,
                tokenizer_mode=None,
            ),
            model_path=None,
            model_config=SimpleNamespace(context_len=128),
        )

    def health_check(self):
        return self.healthy


@unittest.skipIf(_grpc is None, "sglang gRPC native extension is not built")
class TestStandardGrpcHealth(unittest.TestCase):
    def setUp(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()

        self.runtime = _RuntimeHandle()
        self.server = _grpc.start_server(
            "127.0.0.1", self.port, self.runtime, worker_threads=1
        )
        self.channel = grpc.insecure_channel(f"127.0.0.1:{self.port}")
        grpc.channel_ready_future(self.channel).result(timeout=5)
        self.stub = health_pb2_grpc.HealthStub(self.channel)

    def tearDown(self):
        self.channel.close()
        self.server.shutdown()

    def test_check_supports_aggregate_and_native_service_names(self):
        for service in ("", "sglang.runtime.v1.SglangService"):
            with self.subTest(service=service):
                response = self.stub.Check(
                    health_pb2.HealthCheckRequest(service=service), timeout=5
                )
                self.assertEqual(
                    response.status, health_pb2.HealthCheckResponse.SERVING
                )

    def test_check_rejects_unknown_service(self):
        with self.assertRaises(grpc.RpcError) as caught:
            self.stub.Check(
                health_pb2.HealthCheckRequest(service="unknown.Service"), timeout=5
            )
        self.assertEqual(caught.exception.code(), grpc.StatusCode.NOT_FOUND)

    def test_watch_rejects_unknown_service(self):
        responses = self.stub.Watch(
            health_pb2.HealthCheckRequest(service="unknown.Service"), timeout=5
        )
        with self.assertRaises(grpc.RpcError) as caught:
            next(responses)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.NOT_FOUND)

    def test_watch_reports_health_transition(self):
        responses = self.stub.Watch(
            health_pb2.HealthCheckRequest(
                service="sglang.runtime.v1.SglangService"
            ),
            timeout=5,
        )
        self.assertEqual(
            next(responses).status, health_pb2.HealthCheckResponse.SERVING
        )

        self.runtime.healthy = False
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            response = next(responses)
            if response.status == health_pb2.HealthCheckResponse.NOT_SERVING:
                break
        else:
            self.fail("Watch did not publish NOT_SERVING after native health changed")
        responses.cancel()


if __name__ == "__main__":
    unittest.main()
