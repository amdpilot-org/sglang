import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import ComposedPipelineBase
from sglang.multimodal_gen.runtime.platforms import current_platform


class Pipeline(ComposedPipelineBase):
    _required_config_modules = ["transformer", "text_encoder", "vae"]
    def initialize_pipeline(self, server_args): pass
    def create_pipeline_stages(self, server_args): pass


args = SimpleNamespace(
    component_direct_gpu_weight_loading=set(), component_paths={},
    parallel_loading=True, pipeline_config=SimpleNamespace(),
    resolve_component_attention_backend=lambda *_: (None, None),
)
pipeline = object.__new__(Pipeline)
pipeline.model_path = "/model"
pipeline.server_args = args
pipeline._disagg_role = RoleType.MONOLITHIC
pipeline._required_config_modules = list(Pipeline._required_config_modules)
pipeline._unfiltered_required_config_modules = tuple(pipeline._required_config_modules)
pipeline._extra_config_module_map = {}
pipeline.component_loaders = {}
pipeline.memory_usages = {}
model_index = {
    "_class_name": "Probe", "_diffusers_version": "0",
    "transformer": ["diffusers", "Transformer"],
    "text_encoder": ["transformers", "TextEncoder"],
    "vae": ["diffusers", "VAE"],
}
lock = threading.Lock()
active = 0
max_active = 0


def fake_load(*, component_name, **kwargs):
    global active, max_active
    with lock:
        active += 1
        max_active = max(max_active, active)
    time.sleep(0.15)
    with lock:
        active -= 1
    return "loaded-" + component_name, 1.0


start = time.monotonic()
with (
    patch.object(pipeline, "_load_config", return_value=model_index),
    patch.object(current_platform, "get_available_gpu_memory", return_value=16.0),
    patch("sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base.PipelineComponentLoader.load_component", side_effect=fake_load),
):
    modules = pipeline.load_modules(args)
elapsed = time.monotonic() - start
print({"max_active": max_active, "elapsed_seconds": elapsed, "modules": list(modules)})
if max_active < 2:
    raise SystemExit(1)
