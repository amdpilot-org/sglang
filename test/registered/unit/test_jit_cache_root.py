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


def test_deep_gemm_cache_is_configured_before_import():
    source = (
        Path(__file__).parents[3]
        / "python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py"
    ).read_text()
    configure_at = source.index(
        'os.environ["DG_JIT_CACHE_DIR"] = deep_gemm_cache_dir()'
    )
    import_at = source.index("    import deep_gemm")
    assert configure_at < import_at
