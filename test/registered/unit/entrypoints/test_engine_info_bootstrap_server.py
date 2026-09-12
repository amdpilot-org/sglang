import socket
import time
import unittest

import requests

from sglang.srt.entrypoints.engine_info_bootstrap_server import (
    EngineInfoBootstrapServer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestEngineInfoBootstrapServer(unittest.TestCase):
    def setUp(self):
        port = _get_free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self.server = EngineInfoBootstrapServer("127.0.0.1", port)
        self.addCleanup(self.server.close)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if requests.get(f"{self.base_url}/health", timeout=0.1).ok:
                    break
            except requests.RequestException:
                pass
        else:
            self.fail("EngineInfoBootstrapServer did not become ready")

    def _register(self, rank: int, session_id: str, weight_name: str) -> None:
        response = requests.put(
            f"{self.base_url}/register_transfer_engine_info",
            json={
                "rank": rank,
                "transfer_engine_info": {
                    "session_id": session_id,
                    "weights_info_dict": {weight_name: [1, 2, 4]},
                },
            },
            timeout=1,
        )
        response.raise_for_status()

    def _get(self, rank: int) -> dict:
        response = requests.get(
            f"{self.base_url}/get_transfer_engine_info",
            params={"rank": rank},
            timeout=1,
        )
        response.raise_for_status()
        return response.json()

    def test_metadata_is_kept_for_each_pipeline_rank(self):
        # TP1/PP2 maps the two stages to world ranks 0 and 1, even though both
        # stages have tp_rank=0.
        self._register(0, "pp0-session", "model.layers.0.weight")
        self._register(1, "pp1-session", "model.layers.9.weight")

        self.assertEqual(
            self._get(0)["remote_instance_transfer_engine_info"][0],
            "pp0-session",
        )
        self.assertEqual(
            self._get(1)["remote_instance_transfer_engine_info"][0],
            "pp1-session",
        )

    def test_nonzero_tp_and_pp_world_rank_is_preserved(self):
        # TP4, pp_rank=2, tp_rank=1 maps to world rank 9.
        self._register(9, "pp2-tp1-session", "model.layers.20.weight")

        self.assertEqual(
            self._get(9)["remote_instance_transfer_engine_info"][0],
            "pp2-tp1-session",
        )


if __name__ == "__main__":
    unittest.main()
