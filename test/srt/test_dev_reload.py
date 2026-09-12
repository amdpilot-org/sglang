import importlib
import sys

import pytest

import sglang
from sglang.srt.dev_reload import reload_modules
from sglang.srt.managers.io_struct import DevReloadReqInput
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)


def test_reload_rebinds_existing_instances_and_imported_functions(
    tmp_path, monkeypatch
):
    module_path = tmp_path / "reload_fixture.py"
    module_path.write_text(
        "def value(): return 1\nclass Holder:\n    def value(self): return 1\n"
    )
    monkeypatch.setattr(sglang, "__path__", [*sglang.__path__, str(tmp_path)])
    module = importlib.import_module("sglang.reload_fixture")
    holder = module.Holder()
    old_function = module.value

    consumer = importlib.import_module("sglang.srt.dev_reload")
    monkeypatch.setattr(consumer, "_fixture_import", old_function, raising=False)
    module_path.write_text(
        "def value(): return 222\nclass Holder:\n    def value(self): return 222\n"
    )
    importlib.invalidate_caches()

    result = reload_modules(["sglang.reload_fixture"])

    assert holder.value() == 222
    assert consumer._fixture_import() == 222
    assert result.modules == ("sglang.reload_fixture",)
    assert result.rebound_references >= 1
    sys.modules.pop("sglang.reload_fixture", None)


def test_reload_rebinds_zero_argument_super_for_existing_instances(
    tmp_path, monkeypatch
):
    module_path = tmp_path / "reload_super_fixture.py"
    module_path.write_text(
        "class Base:\n"
        "    def value(self): return 'base-v1'\n"
        "class Child(Base):\n"
        "    def value(self): return super().value() + '-child-v1'\n"
    )
    monkeypatch.setattr(sglang, "__path__", [*sglang.__path__, str(tmp_path)])
    module = importlib.import_module("sglang.reload_super_fixture")
    child = module.Child()
    old_child_class = module.Child

    module_path.write_text(
        "class Base:\n"
        "    def value(self): return 'base-version-2'\n"
        "class Child(Base):\n"
        "    def value(self): return super().value() + '-child-version-2'\n"
    )
    importlib.invalidate_caches()

    reload_modules(["sglang.reload_super_fixture"])

    assert module.Child is old_child_class
    assert child.value() == "base-version-2-child-version-2"
    sys.modules.pop("sglang.reload_super_fixture", None)


@pytest.mark.parametrize(
    "name", ["json", "sglang.not_loaded", "sglang.srt._custom_ops"]
)
def test_reload_rejects_unsafe_or_unavailable_modules(name):
    with pytest.raises(ValueError):
        reload_modules([name])


def test_worker_reload_recaptures_without_weight_loading(monkeypatch):
    worker = type("Worker", (), {})()
    worker.recapture_cuda_graph_for_dev_reload = lambda: setattr(
        worker, "recaptured", True
    )
    updater = object.__new__(SchedulerWeightUpdaterManager)
    updater.tp_worker = worker
    updater.draft_worker = None
    monkeypatch.setattr(
        "sglang.srt.dev_reload.reload_modules",
        lambda modules: type(
            "Result", (), {"modules": tuple(modules), "rebound_references": 3}
        )(),
    )

    result = updater.dev_reload(
        DevReloadReqInput(
            modules=["sglang.srt.layers.activation"], recapture_cuda_graph=True
        )
    )

    assert result.success
    assert result.rebound_references == 3
    assert worker.recaptured is True
