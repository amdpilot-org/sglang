"""Load an exact rebuilt _grpc library and probe standard health behavior."""

import importlib.util
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def load_extension(path):
    name = "sglang.srt.rust_extensions._grpc"
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeHandle:
    def __init__(self):
        self.healthy = False
        self.tokenizer_manager = SimpleNamespace(
            server_args=SimpleNamespace(
                tokenizer_path=None, model_path=None, tokenizer_mode=None
            ),
            model_path=None,
            model_config=SimpleNamespace(context_len=128),
        )

    def health_check(self):
        return self.healthy


def main():
    extension = load_extension(sys.argv[1])
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = extension.start_server(
        "127.0.0.1", port, RuntimeHandle(), worker_threads=1
    )
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=5)
    stub = health_pb2_grpc.HealthStub(channel)
    try:
        try:
            print(
                "check=",
                stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=3).status,
                sep="",
            )
        except grpc.RpcError as exc:
            print(f"check_error={exc.code().name}")

        for label, service in (
            ("watch", ""),
            ("unknown", "never.registered.Service"),
        ):
            try:
                stream = stub.Watch(
                    health_pb2.HealthCheckRequest(service=service), timeout=2.4
                )
                print(f"{label}_first={next(stream).status}")
                started = time.monotonic()
                try:
                    print(f"{label}_second={next(stream).status}")
                except grpc.RpcError as exc:
                    print(f"{label}_second_error={exc.code().name}")
                    print(
                        f"{label}_wait_seconds={time.monotonic() - started:.2f}"
                    )
            except grpc.RpcError as exc:
                print(f"{label}_open_error={exc.code().name}")
    finally:
        channel.close()
        server.shutdown()


if __name__ == "__main__":
    main()
