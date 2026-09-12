import json
import os
from pathlib import Path

import pytest

from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.warmup_marker import (
    make_marker_payload,
    marker_matches,
    marker_path,
    write_marker_atomic,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=10, suite="stage-b-test-cpu-intel")


def _payload(**overrides):
    values = {
        "identity": {
            "sglang": "1.2.3",
            "deep_gemm": "0.1.0",
            "torch": "2.9.0+cu129",
            "cuda": "12.9",
            "triton": "3.3.0",
            "gpu_arch": "sm100",
        },
        "settings": {"fast_warmup": False, "dg_jit_use_nvrtc": "0"},
        "kernel_type": "GEMM_NT_F8F8BF16",
        "n": 4096,
        "k": 7168,
        "num_groups": 1,
        "m_values": [3, 1, 2, 2],
    }
    values.update(overrides)
    return make_marker_payload(**values)


def test_marker_round_trip_and_normalizes_exact_m_set(tmp_path: Path):
    payload = _payload()
    path = marker_path(tmp_path, payload)

    write_marker_atomic(path, payload)

    assert marker_matches(path, payload)
    assert json.loads(path.read_text())["kernel"]["m_values"] == [1, 2, 3]
    assert not list(path.parent.glob("*.tmp"))


def test_missing_corrupt_and_incomplete_markers_are_cache_misses(tmp_path: Path):
    payload = _payload()
    path = marker_path(tmp_path, payload)
    assert not marker_matches(path, payload)

    path.parent.mkdir(parents=True)
    path.write_text("not json")
    assert not marker_matches(path, payload)

    path.write_text(json.dumps({**payload, "complete": False}))
    assert not marker_matches(path, payload)


def test_every_compatibility_dimension_invalidates_marker(tmp_path: Path):
    baseline = _payload()
    path = marker_path(tmp_path, baseline)
    write_marker_atomic(path, baseline)

    incompatible = [
        _payload(identity={**baseline["identity"], "cuda": "13.0"}),
        _payload(settings={**baseline["settings"], "dg_jit_use_nvrtc": "1"}),
        _payload(kernel_type="GROUPED_GEMM_NT_F8F8BF16_CONTIG"),
        _payload(n=8192),
        _payload(k=8192),
        _payload(num_groups=8),
        _payload(m_values=[1, 2]),
    ]
    for candidate in incompatible:
        assert marker_path(tmp_path, candidate) != path
        assert not marker_matches(path, candidate)


@pytest.fixture
def compile_marker_state(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(compile_utils, "_ENABLE_JIT_DEEPGEMM_PRECOMPILE", True)
    monkeypatch.setattr(compile_utils, "_DO_COMPILE_ALL", True)
    monkeypatch.setattr(compile_utils, "_USE_WARMUP_MARKER", True)
    monkeypatch.setattr(compile_utils, "_IN_PRECOMPILE_STAGE", True)
    monkeypatch.setattr(compile_utils, "_BUILTIN_M_LIST", [3, 1, 2, 2])
    monkeypatch.setattr(
        compile_utils, "_normalize_warmup_m_values", lambda _t, m: sorted(set(m))
    )
    monkeypatch.setattr(compile_utils, "_warmup_identity", lambda: {"test": "identity"})
    compile_utils._INITIALIZATION_DICT.clear()
    yield
    compile_utils._INITIALIZATION_DICT.clear()


def test_successful_warmup_marks_then_restored_cache_skips(
    compile_marker_state, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        compile_utils,
        "_compile_deep_gemm_one_type_all",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    args = (compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16, 16, 32, 1)

    compile_utils._maybe_compile_deep_gemm_one_type_all(*args)
    assert len(calls) == 1

    compile_utils._INITIALIZATION_DICT.clear()  # simulate a restored-cache process
    compile_utils._maybe_compile_deep_gemm_one_type_all(*args)
    assert len(calls) == 1


def test_opt_out_preserves_repeated_warmup(compile_marker_state, monkeypatch):
    calls = []
    monkeypatch.setattr(compile_utils, "_USE_WARMUP_MARKER", False)
    monkeypatch.setattr(
        compile_utils,
        "_compile_deep_gemm_one_type_all",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    args = (compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16, 16, 32, 1)

    compile_utils._maybe_compile_deep_gemm_one_type_all(*args)
    compile_utils._INITIALIZATION_DICT.clear()
    compile_utils._maybe_compile_deep_gemm_one_type_all(*args)

    assert len(calls) == 2
    assert not list(Path(os.environ["DG_JIT_CACHE_DIR"]).rglob("*.json"))


def test_failed_or_partial_warmup_never_marks(compile_marker_state, monkeypatch):
    args = (compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16, 16, 32, 1)
    monkeypatch.setattr(
        compile_utils, "_compile_deep_gemm_one_type_all", lambda **_kwargs: False
    )
    compile_utils._maybe_compile_deep_gemm_one_type_all(*args)
    assert not list(Path(os.environ["DG_JIT_CACHE_DIR"]).rglob("*.json"))

    compile_utils._INITIALIZATION_DICT.clear()
    monkeypatch.setattr(
        compile_utils,
        "_compile_deep_gemm_one_type_all",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("warmup failed")),
    )
    with pytest.raises(RuntimeError, match="warmup failed"):
        compile_utils._maybe_compile_deep_gemm_one_type_all(*args)
    assert not list(Path(os.environ["DG_JIT_CACHE_DIR"]).rglob("*.json"))


def test_corrupt_marker_falls_back_to_warmup(compile_marker_state, monkeypatch):
    args = (compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16, 16, 32, 1)
    payload = compile_utils._warmup_marker_payload(*args, [1, 2, 3])
    path = marker_path(os.environ["DG_JIT_CACHE_DIR"], payload)
    path.parent.mkdir(parents=True)
    path.write_text("truncated{")
    calls = []
    monkeypatch.setattr(
        compile_utils,
        "_compile_deep_gemm_one_type_all",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    compile_utils._maybe_compile_deep_gemm_one_type_all(*args)
    assert len(calls) == 1
    assert marker_matches(path, payload)


def test_memory_reduced_coverage_is_not_complete(monkeypatch):
    class Executor:
        executed = []

        def execute(self, m):
            self.executed.append(m)

    executor = Executor()
    monkeypatch.setattr(compile_utils, "deep_gemm", object(), raising=False)
    monkeypatch.setattr(compile_utils, "disable_symmetric_memory_context", lambda: None)
    monkeypatch.setattr(
        compile_utils, "restore_symmetric_memory_context", lambda _x: None
    )
    monkeypatch.setattr(compile_utils, "get_available_gpu_memory", lambda **_kwargs: 1)
    monkeypatch.setattr(compile_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        compile_utils.torch.cuda,
        "current_stream",
        lambda: type("S", (), {"synchronize": lambda self: None})(),
    )
    monkeypatch.setattr(compile_utils.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        compile_utils._BaseWarmupExecutor, "create", lambda *_args, **_kwargs: executor
    )
    monkeypatch.setattr(
        compile_utils._BaseWarmupExecutor,
        "get_memory_requirement",
        lambda _type, max_m, **_kwargs: 2 if max_m > 2048 else 0,
    )

    complete = compile_utils._compile_deep_gemm_one_type_all(
        compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16,
        n=16,
        k=32,
        num_groups=1,
        m_list=[1, 4096],
    )

    assert not complete
    assert executor.executed == [1]


def test_compile_mode_is_restored_when_warmup_raises(monkeypatch):
    class DeepGemm:
        mode = 7

        @classmethod
        def get_compile_mode(cls):
            return cls.mode

        @classmethod
        def set_compile_mode(cls, mode):
            cls.mode = mode

    class Executor:
        def execute(self, m):
            del m
            raise RuntimeError("kernel failure")

    monkeypatch.setattr(compile_utils, "deep_gemm", DeepGemm, raising=False)
    monkeypatch.setattr(compile_utils, "disable_symmetric_memory_context", lambda: None)
    monkeypatch.setattr(
        compile_utils, "restore_symmetric_memory_context", lambda _x: None
    )
    monkeypatch.setattr(compile_utils, "get_available_gpu_memory", lambda **_kwargs: 10)
    monkeypatch.setattr(compile_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        compile_utils._BaseWarmupExecutor,
        "create",
        lambda *_args, **_kwargs: Executor(),
    )
    monkeypatch.setattr(
        compile_utils._BaseWarmupExecutor,
        "get_memory_requirement",
        lambda *_args, **_kwargs: 0,
    )

    with pytest.raises(RuntimeError, match="kernel failure"):
        compile_utils._compile_deep_gemm_one_type_all(
            compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16,
            n=16,
            k=32,
            num_groups=1,
            m_list=[1],
        )
    assert DeepGemm.mode == 7
