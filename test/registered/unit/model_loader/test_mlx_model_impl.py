from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1)
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


def test_mlx_skips_hunyuan_architecture_resolution():
    model_config = SimpleNamespace()

    with (
        patch("sglang.srt.hardware_backend.mlx.runtime.use_mlx", return_value=True),
        patch(
            "sglang.srt.model_loader.utils.get_model_architecture",
            side_effect=ValueError(
                "Cannot find model module. 'HunYuanForCausalLM' is not registered"
            ),
        ) as resolve_architecture,
    ):
        assert get_resolved_model_impl(model_config) == ModelImpl.SGLANG

    resolve_architecture.assert_not_called()


def test_non_mlx_still_resolves_model_architecture():
    model_config = SimpleNamespace()

    with (
        patch("sglang.srt.hardware_backend.mlx.runtime.use_mlx", return_value=False),
        patch(
            "sglang.srt.model_loader.utils.get_model_architecture",
            return_value=(object(), "TransformersForCausalLM"),
        ) as resolve_architecture,
    ):
        assert get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS

    resolve_architecture.assert_called_once_with(model_config)


def test_non_mlx_preserves_cached_model_implementation():
    model_config = SimpleNamespace(_resolved_model_impl=ModelImpl.MINDSPORE)

    with (
        patch("sglang.srt.hardware_backend.mlx.runtime.use_mlx", return_value=False),
        patch(
            "sglang.srt.model_loader.utils.get_model_architecture"
        ) as resolve_architecture,
    ):
        assert get_resolved_model_impl(model_config) == ModelImpl.MINDSPORE

    resolve_architecture.assert_not_called()
