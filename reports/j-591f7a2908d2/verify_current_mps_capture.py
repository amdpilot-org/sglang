"""Platform-neutral checks for the mocked MPS capture state machine.

This intentionally does not claim Metal hardware coverage.  It exercises the
current implementation's MPS strategy with a fake context on the prepared
Linux/ROCm host, where the checked-in Apple-Silicon test class is skipped.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.mlx.profiler import MetalCaptureProfiler


def main() -> None:
    with TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "success.gputrace"
        capture = MagicMock()
        with patch.object(
            torch.mps.profiler, "metal_capture", return_value=capture
        ) as metal_capture:
            profiler, result = MetalCaptureProfiler.start_mps(trace_path)

        assert result.success, result.message
        assert profiler is not None
        metal_capture.assert_called_once_with(str(trace_path))
        capture.__enter__.assert_called_once_with()
        profiler.stop()
        capture.__exit__.assert_called_once_with(None, None, None)

    with TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "failure.gputrace"
        with patch.object(
            torch.mps.profiler,
            "metal_capture",
            side_effect=RuntimeError("injected capture failure"),
        ):
            profiler, result = MetalCaptureProfiler.start_mps(trace_path)

        assert profiler is None
        assert not result.success
        assert "injected capture failure" in result.message
        assert "MTL_CAPTURE_ENABLED=1" in result.message

    print("verified mocked MPS success, stop, and RuntimeError boundaries")


if __name__ == "__main__":
    main()
