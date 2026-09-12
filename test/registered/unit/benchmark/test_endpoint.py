from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.benchmark import endpoint


class TestBenchmarkEndpoint(unittest.TestCase):
    def test_launch_uses_spawn_after_resolving_server_args(self):
        server_args = SimpleNamespace(
            host="127.0.0.1", port=30000, resolve_once=Mock()
        )
        process = Mock()
        process.is_alive.return_value = True
        context = Mock()
        context.Process.return_value = process

        with (
            patch.object(endpoint.multiprocessing, "get_context", return_value=context)
            as get_context,
            patch.object(endpoint, "server_is_up", side_effect=[False, True]),
        ):
            launched_process, base_url = endpoint.launch_or_reuse_server(
                Mock(), server_args
            )

        server_args.resolve_once.assert_called_once_with()
        get_context.assert_called_once_with("spawn")
        context.Process.assert_called_once()
        process.start.assert_called_once_with()
        self.assertIs(launched_process, process)
        self.assertEqual(base_url, "http://127.0.0.1:30000")

    def test_running_server_is_reused_without_starting_a_process(self):
        server_args = SimpleNamespace(
            host="127.0.0.1", port=30000, resolve_once=Mock()
        )

        with (
            patch.object(endpoint, "server_is_up", return_value=True),
            patch.object(endpoint.multiprocessing, "get_context") as get_context,
        ):
            process, base_url = endpoint.launch_or_reuse_server(Mock(), server_args)

        server_args.resolve_once.assert_called_once_with()
        get_context.assert_not_called()
        self.assertIsNone(process)
        self.assertEqual(base_url, "http://127.0.0.1:30000")

    def test_explicit_base_url_does_not_resolve_or_launch(self):
        server_args = SimpleNamespace(host="0.0.0.0", port=30000)

        with (
            patch.object(endpoint, "launch_or_reuse_server") as launch,
            patch.object(
                endpoint,
                "_SERVER_ARGS_DEFAULTS",
                {"host": "0.0.0.0", "port": 30000},
            ),
        ):
            acquired = endpoint.acquire_endpoint(
                server_args, base_url="http://example.test:8000"
            )

        launch.assert_not_called()
        self.assertEqual(acquired.base_url, "http://example.test:8000")
        self.assertIsNone(acquired._proc)


if __name__ == "__main__":
    unittest.main()
