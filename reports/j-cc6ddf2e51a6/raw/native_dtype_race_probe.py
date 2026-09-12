import concurrent.futures
import threading

import torch
from transformers.modeling_utils import local_torch_dtype

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)


class NativeLoader(ComponentLoader):
    def load_customized(self, *args, **kwargs):
        raise NotImplementedError


loader = NativeLoader()
a_entered = threading.Event()
b_entered = threading.Event()
a_exited = threading.Event()
observed = {}


def load_native(_path, _args, _library, component_name):
    dtype = torch.float16 if component_name == "a" else torch.float64
    with local_torch_dtype(dtype, component_name):
        if component_name == "a":
            a_entered.set()
            b_entered.wait(timeout=0.1)
            observed[component_name] = torch.get_default_dtype()
        else:
            assert a_entered.wait(timeout=5)
            b_entered.set()
            assert a_exited.wait(timeout=5)
            observed[component_name] = torch.get_default_dtype()
    if component_name == "a":
        a_exited.set()
    return object()


loader.load_native = load_native


def run(name):
    return loader._load_native_with_context(
        f"/model/{name}", object(), name, "transformers", None, name, False
    )


original = torch.get_default_dtype()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    list(executor.map(run, ["a", "b"]))
final = torch.get_default_dtype()
result = {"observed": observed, "original": original, "final": final}
print(result)
expected = {"a": torch.float16, "b": torch.float64}
raise SystemExit(0 if observed == expected and final == original else 1)
