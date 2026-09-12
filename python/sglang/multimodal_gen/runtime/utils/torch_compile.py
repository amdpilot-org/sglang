# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from sglang.multimodal_gen.runtime.platforms import current_platform


def maybe_enable_inductor_compute_comm_overlap() -> None:
    try:
        import torch._inductor.config as _inductor_cfg

        _inductor_cfg.reorder_for_compute_comm_overlap = True
    except ImportError:
        pass


def build_torch_compile_kwargs(
    *, mode: str | None, module: nn.Module | None = None
) -> dict[str, object]:
    compile_kwargs: dict[str, object] = {"fullgraph": False, "dynamic": None}
    if current_platform.is_out_of_tree():
        backend = current_platform.get_compile_backend(mode)
        compile_kwargs["backend"] = backend
        if module is not None:
            options = current_platform.get_compile_options(module)
            if options is not None:
                compile_kwargs["options"] = options
        if (
            "options" not in compile_kwargs
            and backend == "inductor"
            and mode is not None
        ):
            compile_kwargs["mode"] = mode
    elif current_platform.is_npu():
        from sglang.srt.utils.common import get_compiler_backend

        compile_kwargs["backend"] = get_compiler_backend()
        compile_kwargs["dynamic"] = False
    elif mode is not None:
        compile_kwargs["mode"] = mode
    return compile_kwargs


def resolve_torch_compile_mode(
    *env_names: str,
    config: object | None = None,
    default: str,
) -> str:
    for env_name in env_names:
        mode = os.environ.get(env_name)
        if mode:
            return mode
    mode = getattr(config, "torch_compile_mode", None)
    if mode:
        return mode
    return default


def matching_submodule_names(module: nn.Module) -> tuple[str, ...]:
    conditions = getattr(module, "_compile_conditions", ())
    return tuple(
        name
        for name, submodule in module.named_modules()
        if name and any(condition(name, submodule) for condition in conditions)
    )


def region_inventory_digest(regions: tuple[str, ...]) -> str:
    payload = json.dumps(regions, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def compile_matching_submodules(
    module: nn.Module,
    *,
    compile_kwargs: dict[str, object],
) -> int:
    names = matching_submodule_names(module)
    modules = dict(module.named_modules())
    if not names:
        raise ValueError(
            "regional compile found no matching submodules; "
            f"check {type(module).__name__}._compile_conditions"
        )

    for name in names:
        modules[name].compile(**compile_kwargs)
    return len(names)


@dataclass
class CompiledModuleRegistry:
    module_ids: set[int] = field(default_factory=set)
    region_inventories: dict[int, tuple[str, ...]] = field(default_factory=dict)
    region_compiled_calls: dict[int, dict[str, object]] = field(default_factory=dict)

    def is_compiled(self, module: nn.Module) -> bool:
        return id(module) in self.module_ids

    def compile_once(
        self,
        module: nn.Module,
        *,
        compile_kwargs: dict[str, object],
    ) -> bool:
        module_id = id(module)
        if module_id in self.module_ids:
            return False
        module.compile(**compile_kwargs)
        self.module_ids.add(module_id)
        return True

    def compile_regions_once(
        self,
        module: nn.Module,
        *,
        compile_kwargs: dict[str, object],
    ) -> int:
        module_id = id(module)
        if module_id in self.module_ids:
            return 0
        compiled_count = compile_matching_submodules(
            module,
            compile_kwargs=compile_kwargs,
        )
        self.region_inventories[module_id] = matching_submodule_names(module)
        modules = dict(module.named_modules())
        self.region_compiled_calls[module_id] = {
            name: getattr(modules[name], "_compiled_call_impl", None)
            for name in self.region_inventories[module_id]
        }
        self.module_ids.add(module_id)
        return compiled_count

    def region_inventory(self, module: nn.Module) -> tuple[str, ...]:
        return self.region_inventories.get(id(module), matching_submodule_names(module))

    def region_digest(self, module: nn.Module) -> str:
        return region_inventory_digest(self.region_inventory(module))

    def set_regions_active(self, module: nn.Module, active: bool) -> None:
        """Switch promoted regional wrappers on or off at a request boundary."""
        compiled_calls = self.region_compiled_calls.get(id(module))
        if compiled_calls is None:
            return
        modules = dict(module.named_modules())
        for name, compiled_call in compiled_calls.items():
            modules[name]._compiled_call_impl = compiled_call if active else None


class CallableModule(nn.Module):
    """Module wrapper for compiling non-forward callables with module.compile"""

    def __init__(self, fn: Callable[..., Any]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@dataclass
class ActiveTargetCompiledCallable:
    """Cache one compiled callable module for the currently active target object"""

    target_id: int | None = None
    compiled_module: CallableModule | None = None

    def get_or_compile(
        self,
        target: object,
        fn: Callable[..., Any],
        *,
        compile_kwargs: dict[str, object],
    ) -> Callable[..., Any]:
        target_id = id(target)
        if self.target_id == target_id and self.compiled_module is not None:
            return self.compiled_module

        module = CallableModule(fn)
        module.compile(**compile_kwargs)
        self.target_id = target_id
        self.compiled_module = module
        return module
