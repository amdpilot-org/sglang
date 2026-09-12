import importlib.util
import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import grpc


def load_extension(path):
    name = "sglang.srt.rust_extensions._grpc"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Runtime:
    def __init__(self):
        self.phase = "SERVING"
        self.requests = []
        self.tokenizer_manager = SimpleNamespace(
            server_args=SimpleNamespace(tokenizer_path=None, model_path=None, tokenizer_mode=None),
            model_path=None,
            model_config=SimpleNamespace(context_len=128),
        )

    def health_check(self):
        return self.phase == "SERVING"

    def get_operational_state(self):
        accepting = self.phase == "SERVING"
        return json.dumps({
            "phase": self.phase,
            "accepting_new_requests": accepting,
            "draining": self.phase == "DRAINING",
            "ready_to_serve": accepting,
            "weight_update_in_progress": self.phase == "UPDATING_WEIGHTS",
        })

    def submit_request(self, *, req_type, req_dict, chunk_callback, is_disconnected_fn=None):
        self.requests.append((req_type, req_dict))
        chunk_callback({"output_ids": [42], "meta_info": {}}, finished=True)


def main():
    extension = load_extension(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory() as generated:
        subprocess.run(["protoc", f"--python_out={generated}", "-Iproto", "proto/sglang/runtime/v1/sglang.proto"], check=True)
        pb_path = Path(generated) / "sglang/runtime/v1/sglang_pb2.py"
        pb_spec = importlib.util.spec_from_file_location("probe_sglang_pb2", pb_path)
        pb = importlib.util.module_from_spec(pb_spec)
        pb_spec.loader.exec_module(pb)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        runtime = Runtime()
        server = extension.start_server("127.0.0.1", port, runtime, worker_threads=1)
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(channel).result(timeout=5)
        state_rpc = channel.unary_unary(
            "/sglang.runtime.v1.SglangService/GetOperationalState",
            request_serializer=pb.GetOperationalStateRequest.SerializeToString,
            response_deserializer=pb.GetOperationalStateResponse.FromString,
        )
        generate_rpc = channel.unary_stream(
            "/sglang.runtime.v1.SglangService/TextGenerate",
            request_serializer=pb.TextGenerateRequest.SerializeToString,
            response_deserializer=pb.TextGenerateResponse.FromString,
        )
        try:
            for phase in ("SERVING", "DRAINING", "UPDATING_WEIGHTS", "SERVING", "NOT_SERVING"):
                runtime.phase = phase
                state = state_rpc(pb.GetOperationalStateRequest(), timeout=5)
                print("state", phase, state.phase, state.accepting_new_requests, state.draining, state.ready_to_serve, state.weight_update_in_progress)
            runtime.phase = "SERVING"
            response = list(generate_rpc(pb.TextGenerateRequest(
                text="describe",
                images=[pb.MediaInput(uri="https://example/image.png")],
                audio=[pb.MediaInput(data=b"abc", mime_type="audio/wav")],
                videos=[pb.MediaInput(uri="data:video/mp4;base64,AA==")],
                disaggregated_params=pb.DisaggregatedParams(bootstrap_host="prefill", bootstrap_port=30000, bootstrap_room=9),
            ), timeout=5))
            print("response_finished", response[-1].finished)
            print("request", json.dumps(runtime.requests[-1][1], sort_keys=True))
        finally:
            channel.close()
            server.shutdown()


if __name__ == "__main__":
    main()
