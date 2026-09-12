import importlib
import tempfile
import time
from pathlib import Path

import sglang
from sglang.srt.dev_reload import reload_modules


def load_fixture(name, source):
    root = Path(tempfile.mkdtemp(prefix="sglang-reload-review-"))
    path = root / f"{name}.py"
    path.write_text(source)
    sglang.__path__.append(str(root))
    return path, importlib.import_module(f"sglang.{name}")


path, module = load_fixture(
    "review_super_fixture",
    "class Base:\n    def value(self): return 'base-v1'\n"
    "class Child(Base):\n    def value(self): return super().value() + '-child-v1'\n",
)
instance = module.Child()
time.sleep(1.1)
path.write_text(
    "class Base:\n    def value(self): return 'base-version-two'\n"
    "class Child(Base):\n    def value(self): return super().value() + '-child-version-two'\n"
)
reload_modules([module.__name__])
print("zero_arg_super", instance.value())

path, module = load_fixture(
    "review_delete_fixture",
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
print("deleted_function_result", module.removed_function())

path, module = load_fixture(
    "review_bases_fixture",
    "class BaseA:\n    def value(self): return 'A-old'\n"
    "class BaseB:\n    def value(self): return 'B-old'\n"
    "class Child(BaseA):\n    def value(self): return super().value() + '-child-old'\n",
)
instance = module.Child()
time.sleep(1.1)
path.write_text(
    "class BaseA:\n    def value(self): return 'A-new-and-longer'\n"
    "class BaseB:\n    def value(self): return 'B-new-and-longer'\n"
    "class Child(BaseB):\n"
    "    def value(self): return super().value() + '-child-new-and-longer'\n"
)
reload_modules([module.__name__])
print("preserved_bases", [base.__name__ for base in module.Child.__bases__])
print("changed_inheritance_result", instance.value())
