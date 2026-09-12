import threading
import unittest
from collections import OrderedDict

import zmq

from sglang.srt.disaggregation.common.conn import CommonKVManager
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


if __name__ == "__main__":
    unittest.main()
