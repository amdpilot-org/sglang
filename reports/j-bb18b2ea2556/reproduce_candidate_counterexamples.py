import importlib.util
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from types import SimpleNamespace

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def load_extension(path: Path):
    name = "sglang.srt.rust_extensions._grpc"
    spec = importlib.util.spec_from_file_location(name, path)
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
    extension = load_extension(Path(sys.argv[1]).resolve())
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    runtime = RuntimeHandle()
    server = extension.start_server("127.0.0.1", port, runtime, worker_threads=1)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=5)
    stub = health_pb2_grpc.HealthStub(channel)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        watch = stub.Watch(health_pb2.HealthCheckRequest(service=""), timeout=5)
        first = next(watch).status
        try:
            duplicate = pool.submit(next, watch).result(timeout=1.8).status
        except TimeoutError:
            duplicate = "NO_DUPLICATE"
        print(f"unchanged_watch: first={first} next={duplicate}")
        watch.cancel()

        unknown = stub.Watch(
            health_pb2.HealthCheckRequest(service="never.registered.Service"),
            timeout=5,
        )
        try:
            value = next(unknown).status
            print(f"unknown_watch: status={value} stream_open=true")
        except grpc.RpcError as error:
            print(f"unknown_watch: rpc_error={error.code().name} stream_open=false")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        channel.close()
        server.shutdown()


if __name__ == "__main__":
    main()
