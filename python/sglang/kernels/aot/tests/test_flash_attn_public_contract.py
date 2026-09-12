import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

import pytest
import torch


FLASH_ATTN_PATH = Path(
    os.environ.get(
        "TEST_FLASH_ATTN_SOURCE",
        Path(__file__).parents[1] / "python" / "sgl_kernel" / "flash_attn.py",
    )
)
CMAKE_PATH = Path(__file__).parents[1] / "CMakeLists.txt"


class _BackendBoundary:
    def __init__(self):
        self.calls = []
        self.default = self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ("out", "lse", "unused", "unused")


class _Tensor:
    shape = (1, 1, 1, 64)
    dtype = torch.float16

    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)

    def stride(self, dim):
        return 1

    def contiguous(self):
        return self


@pytest.fixture
def flash_attn_module(monkeypatch):
    fake_package = types.ModuleType("sgl_kernel")
    fake_package.__path__ = []
    fake_package.flash_ops = object()
    fake_debug_utils = types.ModuleType("sgl_kernel.debug_utils")
    fake_debug_utils.maybe_wrap_debug_kernel = lambda function: function
    monkeypatch.setitem(sys.modules, "sgl_kernel", fake_package)
    monkeypatch.setitem(sys.modules, "sgl_kernel.debug_utils", fake_debug_utils)
    spec = importlib.util.spec_from_file_location("sgl_kernel.flash_attn", FLASH_ATTN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_pin_contains_native_sm8x_fix():
    cmake = CMAKE_PATH.read_text()
    declaration = cmake.split("# flash-attention", 1)[1].split(
        "FetchContent_Populate(repo-flash-attention)", 1
    )[0]
    assert "149a54916c6a1b940f7ff1afd95f078349a9c471" in declaration
    assert "98cbb200b5057db2b4a40fdd80fb6eae65e784c751ce6d05a9b220f70b797879" in declaration


@pytest.mark.parametrize("wrapper", ["flash_attn_with_kvcache", "flash_attn_varlen_func"])
def test_wrappers_do_not_advertise_an_unimplemented_version_selector(
    flash_attn_module, wrapper
):
    assert "ver" not in inspect.signature(getattr(flash_attn_module, wrapper)).parameters


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 0), True), ((8, 6), True), ((8, 7), True), ((8, 9), True), ((9, 0), True), ((9, 1), False)],
)
def test_is_fa3_supported_matches_compatible_architectures(
    flash_attn_module, monkeypatch, capability, expected
):
    monkeypatch.setattr(torch.version, "cuda", "12.3")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    assert flash_attn_module.is_fa3_supported() is expected


@pytest.mark.parametrize("cuda_version", [None, "12.2"])
def test_is_fa3_supported_requires_cuda_12_3(
    flash_attn_module, monkeypatch, cuda_version
):
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert flash_attn_module.is_fa3_supported() is False


def test_is_fa3_supported_compares_cuda_versions_numerically(
    flash_attn_module, monkeypatch
):
    monkeypatch.setattr(torch.version, "cuda", "12.10")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert flash_attn_module.is_fa3_supported() is True


@pytest.mark.parametrize("wrapper", ["kvcache", "varlen"])
def test_wrapper_validates_the_input_tensor_device(
    flash_attn_module, monkeypatch, wrapper
):
    boundary = _BackendBoundary()
    checked_devices = []
    monkeypatch.setattr(torch.ops.sgl_kernel, "fwd", boundary, raising=False)
    monkeypatch.setattr(
        flash_attn_module,
        "is_fa3_supported",
        lambda device=None: checked_devices.append(device) or True,
    )
    tensor = _Tensor("cuda:1")
    if wrapper == "kvcache":
        result = flash_attn_module.flash_attn_with_kvcache(tensor, tensor, tensor)
    else:
        result = flash_attn_module.flash_attn_varlen_func(
            tensor, tensor, tensor, tensor, tensor, max_seqlen_q=1, max_seqlen_k=1
        )
    assert result == "out"
    assert checked_devices == [torch.device("cuda:1")]
    assert len(boundary.calls) == 1


@pytest.mark.parametrize("wrapper", ["kvcache", "varlen"])
def test_wrapper_rejects_unsupported_input_device_before_backend(
    flash_attn_module, monkeypatch, wrapper
):
    boundary = _BackendBoundary()
    monkeypatch.setattr(torch.ops.sgl_kernel, "fwd", boundary, raising=False)
    monkeypatch.setattr(flash_attn_module, "is_fa3_supported", lambda device=None: False)
    tensor = _Tensor("cuda:1")
    with pytest.raises(NotImplementedError, match="FA3 is not supported"):
        if wrapper == "kvcache":
            flash_attn_module.flash_attn_with_kvcache(tensor, tensor, tensor)
        else:
            flash_attn_module.flash_attn_varlen_func(
                tensor, tensor, tensor, tensor, tensor, max_seqlen_q=1, max_seqlen_k=1
            )
    assert boundary.calls == []
