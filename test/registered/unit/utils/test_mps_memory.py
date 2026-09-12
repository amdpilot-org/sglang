from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang._mps_memory import (
    get_mps_available_memory,
    get_mps_recommended_memory,
)
from sglang._platform_stubs import get_device_properties
from sglang.multimodal_gen.runtime.platforms.mps import MpsPlatform
from sglang.srt.utils.common import get_available_gpu_memory
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_mlx_ci(est_time=12, suite="stage-a-unit-test-mlx")

GiB = 1 << 30


def virtual_memory(*, total: int, available: int):
    return SimpleNamespace(total=total, available=available)


def test_recommended_memory_uses_metal_limit():
    mps = SimpleNamespace(recommended_max_memory=lambda: 48 * GiB)
    with patch(
        "sglang._mps_memory.psutil.virtual_memory",
        return_value=virtual_memory(total=64 * GiB, available=60 * GiB),
    ):
        assert get_mps_recommended_memory(mps) == 48 * GiB


@pytest.mark.parametrize("invalid_limit", [0, None, "invalid"])
def test_recommended_memory_falls_back_for_unsupported_runtime(invalid_limit):
    def recommended_max_memory():
        if invalid_limit is None:
            raise RuntimeError("MPS memory API unavailable")
        return invalid_limit

    mps = SimpleNamespace(recommended_max_memory=recommended_max_memory)
    with patch(
        "sglang._mps_memory.psutil.virtual_memory",
        return_value=virtual_memory(total=64 * GiB, available=60 * GiB),
    ):
        assert get_mps_recommended_memory(mps) == 64 * GiB


def test_available_memory_is_capped_by_remaining_metal_headroom():
    mps = SimpleNamespace(
        recommended_max_memory=lambda: 48 * GiB,
        driver_allocated_memory=lambda: 8 * GiB,
    )
    with patch(
        "sglang._mps_memory.psutil.virtual_memory",
        return_value=virtual_memory(total=64 * GiB, available=56 * GiB),
    ):
        assert get_mps_available_memory(mps) == 40 * GiB


def test_available_memory_respects_tighter_system_pressure():
    mps = SimpleNamespace(
        recommended_max_memory=lambda: 48 * GiB,
        driver_allocated_memory=lambda: 8 * GiB,
    )
    with patch(
        "sglang._mps_memory.psutil.virtual_memory",
        return_value=virtual_memory(total=64 * GiB, available=12 * GiB),
    ):
        assert get_mps_available_memory(mps) == 12 * GiB


def test_available_memory_never_becomes_negative():
    mps = SimpleNamespace(
        recommended_max_memory=lambda: 48 * GiB,
        driver_allocated_memory=lambda: 52 * GiB,
    )
    with patch(
        "sglang._mps_memory.psutil.virtual_memory",
        return_value=virtual_memory(total=64 * GiB, available=12 * GiB),
    ):
        assert get_mps_available_memory(mps) == 0


def test_srt_mps_memory_path_uses_safe_headroom():
    with (
        patch("sglang._mps_memory.get_mps_recommended_memory", return_value=48 * GiB),
        patch(
            "sglang._mps_memory.psutil.virtual_memory",
            return_value=virtual_memory(total=64 * GiB, available=56 * GiB),
        ),
        patch("torch.mps.driver_allocated_memory", return_value=8 * GiB),
    ):
        assert get_available_gpu_memory("mps", 0, empty_cache=False) == 40.0


def test_mps_stub_reports_recommended_capacity():
    with (
        patch("sglang._mps_memory.get_mps_recommended_memory", return_value=48 * GiB),
        patch("sglang._platform_stubs._cached_props", None),
    ):
        assert get_device_properties().total_memory == 48 * GiB


def test_multimodal_mps_platform_uses_safe_memory_values():
    MpsPlatform.get_device_total_memory.cache_clear()
    with (
        patch(
            "sglang.multimodal_gen.runtime.platforms.mps.get_mps_recommended_memory",
            return_value=48 * GiB,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.mps.get_mps_available_memory",
            return_value=40 * GiB,
        ),
    ):
        assert MpsPlatform.get_device_total_memory() == 48 * GiB
        assert MpsPlatform.get_available_gpu_memory(empty_cache=False) == 40.0
    MpsPlatform.get_device_total_memory.cache_clear()
