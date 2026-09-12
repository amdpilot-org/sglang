import importlib
import sys
import tempfile
from pathlib import Path

import sglang

from sglang.srt.dev_reload import reload_modules


def write(path: Path, body: str) -> None:
    path.write_text(body)
    importlib.invalidate_caches()


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    module_path = root / "reload_super_fixture.py"
    write(
        module_path,
        "class Base:\n"
        "    def value(self): return 'base-v1'\n"
        "class Child(Base):\n"
        "    def value(self): return super().value() + '-child-v1'\n",
    )
    sglang.__path__.append(str(root))
    module = importlib.import_module("sglang.reload_super_fixture")
    child = module.Child()
    print("before", child.value())
    write(
        module_path,
        "class Base:\n"
        "    def value(self): return 'base-v2'\n"
        "class Child(Base):\n"
        "    def value(self): return super().value() + '-child-v2'\n",
    )
    result = reload_modules(["sglang.reload_super_fixture"])
    print("reload", result)
    try:
        print("after", child.value())
    except Exception as exc:
        print("after_error", type(exc).__name__, str(exc))
        raise
