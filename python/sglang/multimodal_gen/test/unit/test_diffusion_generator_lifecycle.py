from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.memory_occupation_controller import (
    MemoryOccupationController,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


def _generator() -> DiffGenerator:
    generator = object.__new__(DiffGenerator)
    generator.server_args = SimpleNamespace(model_path="original")
    return generator


def test_diffusion_generator_sleep_wake_and_refit(monkeypatch):
    requests = []

    def forward(req):
        requests.append(req)
        sleeping = isinstance(req, ReleaseMemoryOccupationReqInput)
        return OutputBatch(
            output={"success": True, "sleeping": sleeping, "message": "ok"}
        )

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator."
        "sync_scheduler_client.forward",
        forward,
    )
    generator = _generator()

    assert generator.release_memory_occupation()["sleeping"] is True
    assert generator.resume_memory_occupation()["sleeping"] is False
    assert (
        generator.update_weights_from_disk(
            "/weights/refit", flush_cache=False, target_modules=["transformer"]
        )["success"]
        is True
    )

    assert isinstance(requests[0], ReleaseMemoryOccupationReqInput)
    assert isinstance(requests[1], ResumeMemoryOccupationReqInput)
    assert requests[2] == UpdateWeightFromDiskReqInput(
        model_path="/weights/refit",
        flush_cache=False,
        target_modules=["transformer"],
    )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (OutputBatch(error="engine is sleeping"), "engine is sleeping"),
        (OutputBatch(output=None), "no status payload"),
        (
            OutputBatch(output={"success": False, "message": "refit failed"}),
            "refit failed",
        ),
    ],
)
def test_diffusion_generator_lifecycle_surfaces_failures(
    monkeypatch, response, message
):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator."
        "sync_scheduler_client.forward",
        lambda _req: response,
    )

    with pytest.raises(RuntimeError, match=message):
        _generator().update_weights_from_disk("/weights/refit")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (None, "scheduler returned no response"),
        (SimpleNamespace(error=None), "scheduler returned no status payload"),
    ],
)
def test_diffusion_generator_lifecycle_rejects_malformed_responses(
    monkeypatch, response, message
):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator."
        "sync_scheduler_client.forward",
        lambda _req: response,
    )

    with pytest.raises(
        RuntimeError, match=f"Failed to release memory occupation: {message}"
    ):
        _generator().release_memory_occupation()


@pytest.mark.parametrize(
    "transport_error",
    [ConnectionError("scheduler transport down"), TimeoutError("scheduler timed out")],
)
def test_diffusion_generator_lifecycle_normalizes_transport_failures(
    monkeypatch, transport_error
):
    def forward(_req):
        raise transport_error

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator."
        "sync_scheduler_client.forward",
        forward,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to release memory occupation: scheduler request failed: "
            f"{transport_error}"
        ),
    ) as exc_info:
        _generator().release_memory_occupation()

    assert exc_info.value.__cause__ is transport_error


def test_diffusion_generator_refit_rejects_empty_path(monkeypatch):
    forward = lambda _req: pytest.fail("empty model path must fail client-side")
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator."
        "sync_scheduler_client.forward",
        forward,
    )

    with pytest.raises(ValueError, match="model_path must not be empty"):
        _generator().update_weights_from_disk("")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_memory_occupation_round_trip_matches_cpu_reference():
    torch.manual_seed(7)
    module = torch.nn.Linear(64, 32).eval()
    input_cpu = torch.randn(3, 64)
    reference = torch.nn.functional.linear(
        input_cpu, module.weight.detach(), module.bias.detach()
    )
    module.cuda()
    pipeline = SimpleNamespace(modules={"transformer": module})
    controller = MemoryOccupationController(pipeline, rank=0, use_fsdp_inference=False)

    released = controller.release_memory_occupation()
    assert released["success"] is True
    assert released["sleeping"] is True
    assert module.weight.device.type == "cpu"
    assert controller.release_memory_occupation()["message"] == "already sleeping"

    resumed = controller.resume_memory_occupation()
    assert resumed["success"] is True
    assert resumed["sleeping"] is False
    assert module.weight.device.type == "cuda"
    actual = module(input_cpu.cuda()).cpu()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    assert controller.resume_memory_occupation()["message"] == "already awake"
