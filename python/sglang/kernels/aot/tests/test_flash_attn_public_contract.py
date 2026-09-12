import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest
import torch


FLASH_ATTN_PATH = (
    Path(__file__).parents[1] / "python" / "sgl_kernel" / "flash_attn.py"
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
    device = torch.device("cpu")

    def stride(self, dim):
        return 1

    def contiguous(self):
        return self


def test_fa3_compiler_targets_are_explicit():
    cmake = CMAKE_PATH.read_text()
    fa3_flags = cmake.split("set(SGL_FLASH_KERNEL_CUDA_FLAGS", 1)[1].split(
        "set(FA3_GEN_SRCS", 1
    )[0]

    assert set(re.findall(r"code=sm_(\d+[a-z]?)", fa3_flags)) == {
        "80",
        "86",
        "90a",
    }


@pytest.fixture
def flash_attn_module(monkeypatch):
    fake_package = types.ModuleType("sgl_kernel")
    fake_package.__path__ = []
    fake_package.flash_ops = object()

    fake_debug_utils = types.ModuleType("sgl_kernel.debug_utils")
    fake_debug_utils.maybe_wrap_debug_kernel = lambda function: function

    monkeypatch.setitem(sys.modules, "sgl_kernel", fake_package)
    monkeypatch.setitem(sys.modules, "sgl_kernel.debug_utils", fake_debug_utils)

    spec = importlib.util.spec_from_file_location(
        "sgl_kernel.flash_attn", FLASH_ATTN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 0), True),
        ((8, 6), True),
        ((9, 0), True),
        ((8, 7), False),
        ((8, 9), False),
        ((9, 1), False),
        ((10, 0), False),
    ],
)
def test_is_fa3_supported_matches_compiled_targets(
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


def test_kvcache_version_3_reaches_fa3_backend(flash_attn_module, monkeypatch):
    boundary = _BackendBoundary()
    monkeypatch.setattr(torch.ops.sgl_kernel, "fwd", boundary, raising=False)
    monkeypatch.setattr(flash_attn_module, "is_fa3_supported", lambda device=None: True)

    result = flash_attn_module.flash_attn_with_kvcache(
        _Tensor(), _Tensor(), _Tensor(), ver=3
    )

    assert result == "out"
    assert len(boundary.calls) == 1


@pytest.mark.parametrize("version", [2, 4, 0, None, "3"])
def test_kvcache_rejects_unsupported_versions_before_backend(
    flash_attn_module, monkeypatch, version
):
    boundary = _BackendBoundary()
    monkeypatch.setattr(torch.ops.sgl_kernel, "fwd", boundary, raising=False)
    monkeypatch.setattr(flash_attn_module, "is_fa3_supported", lambda device=None: True)

    with pytest.raises(ValueError, match="only supports ver=3"):
        flash_attn_module.flash_attn_with_kvcache(
            _Tensor(), _Tensor(), _Tensor(), ver=version
        )

    assert boundary.calls == []


def test_kvcache_rejects_unsupported_architecture_before_backend(
    flash_attn_module, monkeypatch
):
    boundary = _BackendBoundary()
    monkeypatch.setattr(torch.ops.sgl_kernel, "fwd", boundary, raising=False)
    monkeypatch.setattr(flash_attn_module, "is_fa3_supported", lambda device=None: False)

    with pytest.raises(NotImplementedError, match="FA3 is not supported"):
        flash_attn_module.flash_attn_with_kvcache(
            _Tensor(), _Tensor(), _Tensor(), ver=3
        )

    assert boundary.calls == []


def test_varlen_version_3_reaches_fa3_backend(flash_attn_module, monkeypatch):
    boundary = _BackendBoundary()
    monkeypatch.setattr(torch.ops.sgl_kernel, "fwd", boundary, raising=False)
    monkeypatch.setattr(flash_attn_module, "is_fa3_supported", lambda device=None: True)

    result = flash_attn_module.flash_attn_varlen_func(
        _Tensor(),
        _Tensor(),
        _Tensor(),
        _Tensor(),
        _Tensor(),
        max_seqlen_q=1,
        max_seqlen_k=1,
        ver=3,
    )

    assert result == "out"
    assert len(boundary.calls) == 1


@pytest.mark.parametrize("version", [2, 4, -1])
def test_varlen_rejects_unsupported_versions_before_backend(
    flash_attn_module, monkeypatch, version
):
    boundary = _BackendBoundary()
    monkeypatch.setattr(torch.ops.sgl_kernel, "fwd", boundary, raising=False)
    monkeypatch.setattr(flash_attn_module, "is_fa3_supported", lambda device=None: True)

    with pytest.raises(ValueError, match="only supports ver=3"):
        flash_attn_module.flash_attn_varlen_func(
            _Tensor(),
            _Tensor(),
            _Tensor(),
            _Tensor(),
            _Tensor(),
            max_seqlen_q=1,
            max_seqlen_k=1,
            ver=version,
        )

    assert boundary.calls == []
