import threading
import time
from concurrent.futures import ThreadPoolExecutor

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)


class ProbeLoader(ComponentLoader):
    def load_customized(self, *args, **kwargs):
        raise NotImplementedError


loader = ProbeLoader()
active = 0
max_active = 0
active_lock = threading.Lock()


def fake_native(*args, **kwargs):
    global active, max_active
    with active_lock:
        active += 1
        max_active = max(max_active, active)
    time.sleep(0.2)
    with active_lock:
        active -= 1
    return object()


loader.load_native = fake_native


def run(name):
    return loader._load_native_with_context(
        f"/model/{name}", object(), name, "transformers", None, name, False
    )


started = time.monotonic()
with ThreadPoolExecutor(max_workers=2) as executor:
    list(executor.map(run, ["text_encoder", "text_encoder_2"]))
elapsed = time.monotonic() - started
print({"max_active_native_loads": max_active, "elapsed_seconds": elapsed})
if max_active != 2:
    raise SystemExit(1)
