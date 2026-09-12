import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import sglang.srt.utils.torch_memory_saver_adapter as adapter_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def _upstream_configure_subprocess():
    os.environ["LD_PRELOAD"] = "/tmp/fake-memory-saver-hook.so"
    try:
        yield
    finally:
        os.environ.pop("LD_PRELOAD", None)


@pytest.fixture
def real_adapter(monkeypatch):
    monkeypatch.setattr(adapter_module, "is_xpu", lambda: False)
    monkeypatch.setattr(
        adapter_module,
        "torch_memory_saver",
        type(
            "FakeTorchMemorySaverModule",
            (),
            {"configure_subprocess": staticmethod(_upstream_configure_subprocess)},
        ),
        raising=False,
    )
    return adapter_module._TorchMemorySaverAdapterReal()


def test_configure_subprocess_adds_pip_cuda_runtime_path(
    monkeypatch, tmp_path: Path, real_adapter
):
    runtime_dir = tmp_path / "site-packages" / "nvidia" / "cu13" / "lib"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "libcudart.so.13").touch()
    monkeypatch.setattr(
        adapter_module.site,
        "getsitepackages",
        lambda: [str(tmp_path / "site-packages")],
    )
    monkeypatch.setattr(adapter_module.torch.version, "cuda", "13.0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")

    with real_adapter.configure_subprocess():
        assert os.environ["LD_PRELOAD"] == "/tmp/fake-memory-saver-hook.so"
        assert os.environ["LD_LIBRARY_PATH"] == f"{runtime_dir}:/existing/lib"

    assert os.environ["LD_LIBRARY_PATH"] == "/existing/lib"


@pytest.mark.parametrize("cuda_version", [None, "12.9"])
def test_configure_subprocess_ignores_missing_matching_runtime(
    monkeypatch, tmp_path: Path, real_adapter, cuda_version
):
    runtime_dir = tmp_path / "site-packages" / "nvidia" / "cu13" / "lib"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "libcudart.so.13").touch()
    monkeypatch.setattr(
        adapter_module.site,
        "getsitepackages",
        lambda: [str(tmp_path / "site-packages")],
    )
    monkeypatch.setattr(adapter_module.torch.version, "cuda", cuda_version)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    with real_adapter.configure_subprocess():
        assert "LD_LIBRARY_PATH" not in os.environ

    assert "LD_LIBRARY_PATH" not in os.environ
