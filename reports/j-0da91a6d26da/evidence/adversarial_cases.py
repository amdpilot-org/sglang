import importlib
import tempfile
import time
from pathlib import Path

import sglang
from sglang.srt.dev_reload import reload_modules


def load_fixture(name, source):
    root = Path(tempfile.mkdtemp(prefix="sglang-reload-correction-"))
    path = root / f"{name}.py"
    path.write_text(source)
    sglang.__path__.append(str(root))
    return path, importlib.import_module(f"sglang.{name}")


path, module = load_fixture(
    "correction_delete_fixture",
    "def removed_function(): return 'stale-function'\n"
    "class RemovedClass:\n    pass\n"
    "kept = 'before'\n",
)
time.sleep(1.1)
path.write_text("kept = 'after-with-longer-source-to-avoid-pyc-aliasing'\n")
reload_modules([module.__name__])
print(
    "deleted_names_present",
    hasattr(module, "removed_function"),
    hasattr(module, "RemovedClass"),
)

path, module = load_fixture(
    "correction_bases_fixture",
    "class BaseA:\n    def value(self): return 'A-old'\n"
    "class BaseB:\n    def value(self): return 'B-old'\n"
    "class Child(BaseA):\n    def value(self): return super().value() + '-child-old'\n",
)
instance = module.Child()
old_child = module.Child
time.sleep(1.1)
path.write_text(
    "class BaseA:\n    def value(self): return 'A-new-and-longer'\n"
    "class BaseB:\n    def value(self): return 'B-new-and-longer'\n"
    "class Child(BaseB):\n"
    "    def value(self): return super().value() + '-child-new-and-longer'\n"
)
reload_modules([module.__name__])
print("class_identity_preserved", module.Child is old_child)
print("preserved_bases", [base.__name__ for base in module.Child.__bases__])
print("existing_instance", instance.value())
print("new_instance", module.Child().value())
