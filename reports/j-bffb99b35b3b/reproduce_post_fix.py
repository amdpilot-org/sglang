"""Exercise config-only serve dispatch without requiring model weights."""

from pathlib import Path
from unittest.mock import patch

from sglang.cli.serve import serve

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def capture_launch(server_args):
    print(f"run_server model_path={server_args.model_path}")


with patch("sglang.launch_server.run_server", capture_launch), patch(
    "sglang.cli.serve.kill_process_tree"
):
    serve(
        None,
        ["--config", str(CONFIG_PATH)],
    )
