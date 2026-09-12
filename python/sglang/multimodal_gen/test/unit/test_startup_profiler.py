import time
from unittest.mock import patch

import pytest

import sglang.multimodal_gen.runtime.utils.startup_profiler as startup_profiler
from sglang.multimodal_gen.runtime.utils.startup_profiler import StartupProfiler


def test_disabled_profiler_is_zero_overhead_noop():
    """When disabled, phase() must not build a tree at all (see #19087 -- the
    profiler must cost nothing when off, since it wraps hot startup code)."""
    profiler = StartupProfiler(enabled=False)
    with patch.object(startup_profiler.time, "perf_counter") as perf_counter:
        with profiler.phase("a"):
            with profiler.phase("b"):
                pass
    assert profiler._root.children == []
    assert profiler.render() == ""
    perf_counter.assert_not_called()


def test_nested_phases_report_percent_of_immediate_parent():
    """A child's percentage is relative to its parent's duration, not the
    grand total -- this is what makes `load_component.text_encoder: 52%`
    meaningful instead of misleading once there's more than one level."""
    profiler = StartupProfiler(enabled=True)
    with profiler.phase("outer"):
        time.sleep(0.02)
        with profiler.phase("inner_a"):
            time.sleep(0.01)
        with profiler.phase("inner_b"):
            time.sleep(0.005)

    lines = profiler.render().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("outer: ")
    assert "(100.0%)" in lines[0]  # top-level phase is 100% of itself
    assert lines[1].startswith("outer.inner_a: ")
    assert lines[2].startswith("outer.inner_b: ")

    # inner_a took ~2x inner_b (10ms vs 5ms), so its percentage should be
    # roughly double -- a derived property of the timings, not a fixed value.
    pct_a = float(lines[1].split("(")[1].rstrip("%)"))
    pct_b = float(lines[2].split("(")[1].rstrip("%)"))
    assert pct_a > pct_b


def test_sibling_phases_at_top_level_are_independent():
    """Two unrelated top-level phases (e.g. init_distributed_environment and
    build_pipeline) must not be nested under each other."""
    profiler = StartupProfiler(enabled=True)
    with profiler.phase("first"):
        pass
    with profiler.phase("second"):
        pass

    lines = profiler.render().splitlines()
    assert lines[0].startswith("first: ")
    assert lines[1].startswith("second: ")


def test_exception_restores_parent_phase_stack():
    profiler = StartupProfiler(enabled=True)
    with pytest.raises(RuntimeError, match="expected"):
        with profiler.phase("outer"):
            with profiler.phase("failing"):
                raise RuntimeError("expected")

    assert profiler._stack == [profiler._root]
    assert [node.name for node in profiler._root.children] == ["outer"]
    assert [node.name for node in profiler._root.children[0].children] == ["failing"]


def test_global_profiler_is_reset_after_fork():
    """A forked worker must not inherit the parent's open phase stack."""
    old_profiler = startup_profiler._profiler
    old_pid = startup_profiler._profiler_pid
    try:
        with patch.object(startup_profiler.envs, "SGLANG_DIFFUSION_STARTUP_PROFILE", True):
            with patch.object(startup_profiler.os, "getpid", return_value=100):
                parent = startup_profiler.get_startup_profiler()
                parent._stack.append(startup_profiler._Phase("parent_open_phase"))

            with patch.object(startup_profiler.os, "getpid", return_value=101):
                child = startup_profiler.get_startup_profiler()

        assert child is not parent
        assert child._stack == [child._root]
        assert child._root.children == []
    finally:
        startup_profiler._profiler = old_profiler
        startup_profiler._profiler_pid = old_pid
