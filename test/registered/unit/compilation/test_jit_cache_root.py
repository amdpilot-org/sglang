import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sglang.srt.environ import (
    deep_gemm_cache_dir,
    envs,
    get_jit_cache_subdir,
    redirect_third_party_caches,
    third_party_cache_defaults,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_CACHE_ENV_VARS = (
    "SGLANG_JIT_CACHE_ROOT",
    "SGLANG_CACHE_DIR",
    "SGLANG_TRITON_CACHE_DIR",
    "SGLANG_TORCHINDUCTOR_CACHE_DIR",
    "SGLANG_DG_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "DG_JIT_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "FLASHINFER_WORKSPACE_BASE",
    "XDG_CACHE_HOME",
)


@pytest.fixture(autouse=True)
def restore_cache_environment(monkeypatch):
    for name in _CACHE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_standard_layout_and_xdg_default(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert envs.SGLANG_JIT_CACHE_ROOT.get() == str(tmp_path / "sglang")
    assert third_party_cache_defaults()["TRITON_CACHE_DIR"] == str(
        tmp_path / "sglang" / "triton"
    )
    assert third_party_cache_defaults()["TORCHINDUCTOR_CACHE_DIR"] == str(
        tmp_path / "sglang" / "inductor"
    )
    assert envs.SGLANG_DG_CACHE_DIR.get() == str(tmp_path / "sglang" / "deep_gemm")
    assert get_jit_cache_subdir("torch_compile") == str(
        tmp_path / "sglang" / "torch_compile"
    )


def test_legacy_sglang_cache_dir_is_root_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CACHE_DIR", str(tmp_path / "legacy"))
    assert envs.SGLANG_JIT_CACHE_ROOT.get() == str(tmp_path / "legacy")


def test_explicit_jit_root_wins_over_legacy_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CACHE_DIR", str(tmp_path / "legacy"))
    monkeypatch.setenv("SGLANG_JIT_CACHE_ROOT", str(tmp_path / "jit"))
    assert envs.SGLANG_JIT_CACHE_ROOT.get() == str(tmp_path / "jit")


@pytest.mark.parametrize(
    ("sglang_name", "native_name", "subdir"),
    (
        ("SGLANG_TRITON_CACHE_DIR", "TRITON_CACHE_DIR", "triton"),
        (
            "SGLANG_TORCHINDUCTOR_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "inductor",
        ),
    ),
)
def test_cache_override_precedence(
    monkeypatch, tmp_path, sglang_name, native_name, subdir
):
    root = tmp_path / "root"
    native = tmp_path / "native"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("SGLANG_JIT_CACHE_ROOT", str(root))
    monkeypatch.setenv(native_name, str(native))

    redirect_third_party_caches()
    assert os.environ[native_name] == str(native)

    monkeypatch.setenv(sglang_name, str(explicit))
    redirect_third_party_caches()
    assert os.environ[native_name] == str(explicit)

    monkeypatch.delenv(sglang_name)
    monkeypatch.delenv(native_name)
    redirect_third_party_caches()
    assert os.environ[native_name] == str(root / subdir)


def test_deep_gemm_override_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_JIT_CACHE_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path / "native"))
    assert deep_gemm_cache_dir() == str(tmp_path / "native")

    monkeypatch.setenv("SGLANG_DG_CACHE_DIR", str(tmp_path / "explicit"))
    assert deep_gemm_cache_dir() == str(tmp_path / "explicit")


def test_inductor_adapter_cannot_move_unified_caches(monkeypatch, tmp_path):
    root = tmp_path / "root"
    monkeypatch.setenv("SGLANG_JIT_CACHE_ROOT", str(root))
    redirect_third_party_caches()

    from sglang.srt.compilation.compiler_interface import InductorAdaptor

    InductorAdaptor().initialize_cache(str(tmp_path / "graph"))
    assert os.environ["TRITON_CACHE_DIR"] == str(root / "triton")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(root / "inductor")


def test_import_time_redirect_in_clean_process(tmp_path):
    code = textwrap.dedent(
        """
        import json, os
        import sglang
        from sglang.srt.environ import envs
        print(json.dumps({
            "root": envs.SGLANG_JIT_CACHE_ROOT.get(),
            "triton": os.environ["TRITON_CACHE_DIR"],
            "inductor": os.environ["TORCHINDUCTOR_CACHE_DIR"],
        }))
        """
    )
    child_env = os.environ.copy()
    for name in _CACHE_ENV_VARS:
        child_env.pop(name, None)
    child_env["SGLANG_JIT_CACHE_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(result.stdout.strip().splitlines()[-1])
    assert values == {
        "root": str(tmp_path),
        "triton": str(tmp_path / "triton"),
        "inductor": str(tmp_path / "inductor"),
    }


def test_deep_gemm_cache_is_configured_before_configurer_probe(tmp_path):
    configurer = (
        Path(__file__).parents[4]
        / "python/sglang/srt/layers/deep_gemm_wrapper/configurer.py"
    )
    code = textwrap.dedent(
        f"""
        import importlib.abc
        import importlib.machinery
        import importlib.util
        import os
        import sys
        import types

        observed = []

        class DeepGemmLoader(importlib.abc.Loader):
            def exec_module(self, module):
                observed.append(os.environ.get("DG_JIT_CACHE_DIR"))

        class DeepGemmFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "deep_gemm":
                    return importlib.machinery.ModuleSpec(fullname, DeepGemmLoader())

        sys.meta_path.insert(0, DeepGemmFinder())
        for name in ("sglang", "sglang.srt"):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

        environ = types.ModuleType("sglang.srt.environ")
        environ.deep_gemm_cache_dir = lambda: {str(tmp_path / "deep_gemm")!r}
        environ.envs = types.SimpleNamespace(
            SGLANG_ENABLE_JIT_DEEPGEMM=types.SimpleNamespace(get=lambda: True)
        )
        sys.modules[environ.__name__] = environ

        runtime_context = types.ModuleType("sglang.srt.runtime_context")
        runtime_context.get_platform = lambda: types.SimpleNamespace(is_sm100=False)
        sys.modules[runtime_context.__name__] = runtime_context

        utils = types.ModuleType("sglang.srt.utils")
        utils.get_device_sm = lambda: 90
        utils.is_cuda = lambda: True
        utils.is_musa = lambda: False
        sys.modules[utils.__name__] = utils

        spec = importlib.util.spec_from_file_location("test_configurer", {str(configurer)!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert observed == [{str(tmp_path / "deep_gemm")!r}], observed
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
