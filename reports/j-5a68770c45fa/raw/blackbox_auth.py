import http.client
import json
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BINARY = "/tmp/amdpilot-repo-j-5a68770c45fa/cargo-target/release/sgl-router"
TOKENIZER = "/job/review-evidence-j-5a68770c45fa/tokenizer.json"


class Worker(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        if self.path == "/flush_cache":
            Worker.calls.append(dict(self.headers.items()))
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args):
        pass


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post(port, authorization=None):
    headers = {}
    if authorization is not None:
        headers["Authorization"] = authorization
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    conn.request("POST", "/flush_cache", headers=headers)
    response = conn.getresponse()
    body = response.read().decode()
    result = (response.status, dict(response.getheaders()), body)
    conn.close()
    return result


def wait_ready(port, proc):
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError(f"router exited early: {proc.returncode}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.2)
            conn.request("GET", "/healthz")
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status == 200:
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("router did not become ready")


def run_router(worker_port, admin_key=None):
    port = free_port()
    command = [
        BINARY,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model-id", "review-model",
        "--tokenizer-path", TOKENIZER,
        "--worker-urls", f"http://127.0.0.1:{worker_port}",
    ]
    if admin_key is not None:
        command.extend(["--admin-api-key", admin_key])
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wait_ready(port, proc)
    time.sleep(0.5)
    return port, proc


def stop(proc):
    proc.terminate()
    output, _ = proc.communicate(timeout=5)
    return output


worker = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
thread = threading.Thread(target=worker.serve_forever, daemon=True)
thread.start()
worker_port = worker.server_address[1]
results = {}

port, proc = run_router(worker_port)
before = len(Worker.calls)
results["no_key"] = post(port)
assert results["no_key"][0] == 200
assert len(Worker.calls) == before + 1
results["no_key_router_log"] = stop(proc)

port, proc = run_router(worker_port, "review-secret")
before = len(Worker.calls)
for label, credential in [
    ("missing", None),
    ("raw", "review-secret"),
    ("wrong_scheme", "Basic review-secret"),
    ("lowercase_scheme", "bearer review-secret"),
    ("wrong_key", "Bearer wrong"),
    ("extra_suffix", "Bearer review-secret extra"),
]:
    results[label] = post(port, credential)
    assert results[label][0] == 401, (label, results[label])
    assert results[label][1].get("www-authenticate") == "Bearer"
assert len(Worker.calls) == before, "unauthorized request reached worker"

results["valid"] = post(port, "Bearer review-secret")
assert results["valid"][0] == 200
assert len(Worker.calls) == before + 1
assert "Authorization" not in Worker.calls[-1]
results["keyed_router_log"] = stop(proc)
worker.shutdown()

print(json.dumps({
    "statuses": {key: value[0] for key, value in results.items() if isinstance(value, tuple)},
    "worker_calls": len(Worker.calls),
    "valid_worker_headers": Worker.calls[-1],
}, indent=2, sort_keys=True))
