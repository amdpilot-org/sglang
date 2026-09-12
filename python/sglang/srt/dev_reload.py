"""Development-only helpers for reloading pure-Python SGLang modules in place."""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable


@dataclass(frozen=True)
class ReloadResult:
    modules: tuple[str, ...]
    rebound_references: int


def _validate_module(name: str) -> ModuleType:
    if not name.startswith("sglang."):
        raise ValueError(f"Refusing to reload non-SGLang module: {name!r}")
    module = sys.modules.get(name)
    if module is None:
        raise ValueError(f"Module is not loaded in this process: {name!r}")
    spec = module.__spec__
    origin = None if spec is None else spec.origin
    if not origin or Path(origin).suffix not in {".py", ".pyc"}:
        raise ValueError(
            f"Module is not reloadable pure Python: {name!r} (origin={origin!r})"
        )
    return module


def _replace_class_contents(old: type, new: type) -> None:
    """Make existing instances dispatch through definitions from ``new``."""
    protected = {"__dict__", "__weakref__", "__module__", "__name__", "__qualname__"}
    for name in set(vars(old)) - set(vars(new)) - protected:
        delattr(old, name)
    for name, value in vars(new).items():
        if name not in protected:
            setattr(old, name, value)


def reload_modules(module_names: Iterable[str]) -> ReloadResult:
    """Reload selected modules and repair common references to replaced objects.

    Existing class identities are retained and updated, which makes already-created
    scheduler/model instances see new methods. Module globals that imported replaced
    functions or classes directly are also rebound.
    """
    names = tuple(dict.fromkeys(module_names))
    if not names:
        raise ValueError("At least one module must be specified")
    modules = [_validate_module(name) for name in names]
    replacements: dict[int, object] = {}

    for module in modules:
        old_values = dict(vars(module))
        importlib.invalidate_caches()
        importlib.reload(module)
        for name, old in old_values.items():
            new = vars(module).get(name)
            if new is None or new is old:
                continue
            if inspect.isclass(old) and inspect.isclass(new):
                _replace_class_contents(old, new)
                setattr(module, name, old)
                replacements[id(new)] = old
            elif inspect.isfunction(old) and inspect.isfunction(new):
                replacements[id(old)] = new

    rebound = 0
    for module in tuple(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("sglang."):
            continue
        for name, value in tuple(vars(module).items()):
            replacement = replacements.get(id(value))
            if replacement is not None and replacement is not value:
                setattr(module, name, replacement)
                rebound += 1
    return ReloadResult(names, rebound)
