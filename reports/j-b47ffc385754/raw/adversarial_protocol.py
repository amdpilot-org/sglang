import socket
import threading
import time
import json

from sglang.srt.distributed.nccl_ras import RasSocketClient


observed = []
ready = threading.Event()
port_holder = []


def server():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port_holder.append(listener.getsockname()[1])
        ready.set()
        conn, _ = listener.accept()
        with conn:
            first = conn.recv(4096)
            observed.append(first)
            assert first == b"SET FORMAT json\nSTATUS\n", first
            # NCCL documents that STATUS can take several seconds while it
            # gathers job-wide communicator data. This is only slightly above
            # the candidate's hard-coded one-second read timeout.
            time.sleep(1.2)
            payload = {
                "nccl_version": "2.30.7",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "communicators_count": 0,
                "communicators": [],
            }
            conn.sendall(b"OK\n" + json.dumps(payload).encode())


t = threading.Thread(target=server)
t.start()
ready.wait()
result = RasSocketClient(
    addr="127.0.0.1", port=port_holder[0], connect_timeout=1, read_timeout=1
).poll_status()
t.join()
print(f"first_client_bytes={observed[0]!r}")
print(f"poll_result={result!r}")
assert result is not None, "valid delayed STATUS was misclassified as poll failure"
