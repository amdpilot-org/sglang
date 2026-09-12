from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.benchmark import one_batch
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


def args(backend):
    value = ServerArgs(
        model_path="dummy",
        load_format="dummy",
        moe_runner_backend=backend,
        disable_cuda_graph=True,
    )
    value.resolve_once()
    return value


@pytest.mark.parametrize("backend", ["flashinfer_mxfp4", "aiter", "auto"])
def test_direct_load_model_materializes_exact_backend(backend):
    reset_context()
    server_args = args(backend)
    publish(server_args, role="scheduler")

    def observe(*_args, **_kwargs):
        assert get_moe_runner_backend().value == backend
        raise RuntimeError("observed")

    with patch.object(one_batch.ModelConfig, "from_server_args", side_effect=observe):
        with pytest.raises(RuntimeError, match="observed"):
            one_batch.load_model(server_args, SimpleNamespace(nccl_port=1), 0, 0)


def test_correctness_path_materializes_original_issue_backend():
    reset_context()
    server_args = args("flashinfer_mxfp4")
    bench_args = one_batch.BenchArgs(
        batch_size=(1,), input_len=(1,), output_len=(1,), correctness_test=True
    )

    def observe(*_args, **_kwargs):
        assert get_moe_runner_backend().value == "flashinfer_mxfp4"
        raise RuntimeError("observed")

    with patch.object(one_batch, "configure_logger"), patch.object(
        one_batch.ModelConfig, "from_server_args", side_effect=observe
    ):
        with pytest.raises(RuntimeError, match="observed"):
            one_batch.correctness_test(
                server_args, SimpleNamespace(nccl_port=1), bench_args, 0, 0
            )


def test_republish_replaces_prior_backend_before_load():
    reset_context()
    publish(args("aiter"), role="scheduler")
    server_args = args("flashinfer_mxfp4")
    publish(server_args, role="scheduler")

    def observe(*_args, **_kwargs):
        assert get_moe_runner_backend().value == "flashinfer_mxfp4"
        raise RuntimeError("observed")

    with patch.object(one_batch.ModelConfig, "from_server_args", side_effect=observe):
        with pytest.raises(RuntimeError, match="observed"):
            one_batch.load_model(server_args, SimpleNamespace(nccl_port=1), 0, 0)
