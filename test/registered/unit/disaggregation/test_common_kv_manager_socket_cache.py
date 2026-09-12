import os
import subprocess
import sys
import threading
import unittest
from collections import OrderedDict
from unittest.mock import patch

import zmq

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.environ import _default_disaggregation_zmq_socket_cache_size
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestCommonKVManagerSocketCache(unittest.TestCase):
    def setUp(self):
        self.manager = CommonKVManager.__new__(CommonKVManager)
        self.manager._zmq_ctx = zmq.Context()
        self.manager._socket_cache = OrderedDict()
        self.manager._monitor_cache = {}
        self.manager._socket_send_locks = {}
        self.manager._socket_lock = threading.Lock()
        self.manager._socket_cache_size = 3

    def tearDown(self):
        for sock in self.manager._socket_cache.values():
            sock.close(linger=0)
        for monitor in self.manager._monitor_cache.values():
            monitor.close(linger=0)
        self.manager._zmq_ctx.term()

    def test_unique_endpoints_are_bounded_and_evicted_sockets_are_closed(self):
        created = []
        for port in range(31000, 31006):
            endpoint = f"tcp://127.0.0.1:{port}"
            sock = self.manager._connect(endpoint)
            created.append((sock, sock._monitor_socket))

        self.assertEqual(
            list(self.manager._socket_cache),
            [
                "tcp://127.0.0.1:31003",
                "tcp://127.0.0.1:31004",
                "tcp://127.0.0.1:31005",
            ],
        )
        self.assertEqual(
            set(self.manager._socket_cache), set(self.manager._monitor_cache)
        )
        for sock, monitor in created[:3]:
            self.assertTrue(sock.closed)
            self.assertTrue(monitor.closed)
        for sock, monitor in created[3:]:
            self.assertFalse(sock.closed)
            self.assertFalse(monitor.closed)

    def test_reusing_endpoint_does_not_consume_another_cache_entry(self):
        endpoint = "tcp://127.0.0.1:32000"
        first = self.manager._connect(endpoint)
        second = self.manager._connect(endpoint)

        self.assertIs(first, second)
        self.assertEqual(len(self.manager._socket_cache), 1)
        self.assertEqual(len(self.manager._monitor_cache), 1)

    def test_recently_reused_endpoint_is_not_evicted(self):
        endpoints = [f"tcp://127.0.0.1:{port}" for port in range(33000, 33004)]
        for endpoint in endpoints[:3]:
            self.manager._connect(endpoint)

        reused = self.manager._connect(endpoints[0])
        self.manager._connect(endpoints[3])

        self.assertIs(self.manager._socket_cache[endpoints[0]], reused)
        self.assertNotIn(endpoints[1], self.manager._socket_cache)
        self.assertEqual(len(self.manager._socket_cache), 3)

    def test_busy_socket_is_not_closed_during_eviction(self):
        self.manager._socket_cache_size = 1
        first_endpoint = "tcp://127.0.0.1:34000"
        first = self.manager._connect(first_endpoint)
        first_lock = self.manager._socket_send_locks[first_endpoint]

        first_lock.acquire()
        try:
            self.manager._connect("tcp://127.0.0.1:34001")
            self.assertFalse(first.closed)
            self.assertEqual(len(self.manager._socket_cache), 2)
        finally:
            first_lock.release()

        self.manager._connect("tcp://127.0.0.1:34002")
        self.assertTrue(first.closed)
        self.assertEqual(len(self.manager._socket_cache), 1)

    def test_monitor_creation_failure_does_not_cache_push_socket(self):
        class FailingSocket:
            closed = False

            def setsockopt(self, *_args):
                pass

            def connect(self, _endpoint):
                pass

            def get_monitor_socket(self, _events):
                raise zmq.ZMQError("monitor failure")

            def disable_monitor(self):
                pass

            def close(self, linger=None):
                self.closed = True

        failing_socket = FailingSocket()
        self.manager._zmq_ctx.term()
        self.manager._zmq_ctx = type(
            "FailingContext",
            (),
            {
                "socket": lambda _self, _kind: failing_socket,
                "term": lambda _self: None,
            },
        )()

        with self.assertRaisesRegex(zmq.ZMQError, "monitor failure"):
            self.manager._connect("tcp://127.0.0.1:35000")

        self.assertTrue(failing_socket.closed)
        self.assertEqual(self.manager._socket_cache, {})
        self.assertEqual(self.manager._monitor_cache, {})

    def test_default_cache_size_respects_low_process_fd_limit(self):
        script = """
import os
import resource
import threading
from collections import OrderedDict

import zmq

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.environ import envs

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(128, hard), hard))

manager = CommonKVManager.__new__(CommonKVManager)
manager._zmq_ctx = zmq.Context()
manager._socket_cache = OrderedDict()
manager._monitor_cache = {}
manager._socket_send_locks = {}
manager._socket_lock = threading.Lock()
manager._socket_cache_size = max(
    1, envs.SGLANG_DISAGGREGATION_ZMQ_SOCKET_CACHE_SIZE.get()
)

try:
    for port in range(1000):
        manager._connect(f\"tcp://127.0.0.1:{30000 + port}\")
    print(f\"cache_size={manager._socket_cache_size}\")
    print(f\"cached={len(manager._socket_cache)}\")
    print(f\"open_fds={len(os.listdir('/proc/self/fd'))}\")
finally:
    for endpoint in list(manager._socket_cache):
        manager._close_cached_socket(endpoint)
    manager._zmq_ctx.term()
"""
        env = os.environ.copy()
        env.pop("SGLANG_DISAGGREGATION_ZMQ_SOCKET_CACHE_SIZE", None)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env=env,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        measurements = dict(
            line.split("=", maxsplit=1)
            for line in result.stdout.splitlines()
            if "=" in line
        )
        cache_size = int(measurements["cache_size"])
        self.assertLess(cache_size, 32)
        self.assertEqual(int(measurements["cached"]), cache_size)
        self.assertLess(int(measurements["open_fds"]), 128)

    @patch("sglang.srt.environ.os.listdir", return_value=[str(i) for i in range(8)])
    @patch("sglang.srt.environ.resource.getrlimit", return_value=(128, 128))
    def test_default_cache_size_uses_fd_headroom(self, _getrlimit, _listdir):
        self.assertEqual(_default_disaggregation_zmq_socket_cache_size(), 22)

    @patch("sglang.srt.environ.os.listdir", return_value=[])
    @patch("sglang.srt.environ.resource.getrlimit", return_value=(1_048_576, 1_048_576))
    def test_default_cache_size_keeps_normal_cap(self, _getrlimit, _listdir):
        self.assertEqual(_default_disaggregation_zmq_socket_cache_size(), 1024)


if __name__ == "__main__":
    unittest.main()
