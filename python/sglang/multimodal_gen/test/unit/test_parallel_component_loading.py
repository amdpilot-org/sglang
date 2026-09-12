# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser


class _Pipeline(ComposedPipelineBase):
    _required_config_modules = ["transformer", "text_encoder", "vae"]

    def initialize_pipeline(self, server_args):
        pass

    def create_pipeline_stages(self, server_args):
        pass


def _pipeline_and_args(*, parallel_loading=True):
    args = SimpleNamespace(
        component_direct_gpu_weight_loading=set(),
        component_paths={},
        parallel_loading=parallel_loading,
        pipeline_config=SimpleNamespace(),
        resolve_component_attention_backend=lambda *_: (None, None),
    )
    pipeline = object.__new__(_Pipeline)
    pipeline.model_path = "/model"
    pipeline.server_args = args
    pipeline._disagg_role = RoleType.MONOLITHIC
    pipeline._required_config_modules = list(_Pipeline._required_config_modules)
    pipeline._unfiltered_required_config_modules = tuple(
        pipeline._required_config_modules
    )
    pipeline._extra_config_module_map = {}
    pipeline.component_loaders = {}
    pipeline.memory_usages = {}
    return pipeline, args


def _model_index():
    return {
        "_class_name": "TestPipeline",
        "_diffusers_version": "0",
        "transformer": ["diffusers", "Transformer"],
        "text_encoder": ["transformers", "TextEncoder"],
        "vae": ["diffusers", "VAE"],
    }


def test_component_loads_overlap_and_preserve_results():
    pipeline, args = _pipeline_and_args()
    barrier = threading.Barrier(3)
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def load_component(*, component_name, **_kwargs):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        with active_lock:
            active -= 1
        return f"loaded-{component_name}", float(len(component_name))

    with (
        patch.object(pipeline, "_load_config", return_value=_model_index()),
        patch.object(current_platform, "get_available_gpu_memory", return_value=16.0),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base."
            "PipelineComponentLoader.load_component",
            side_effect=load_component,
        ),
    ):
        modules = pipeline.load_modules(args)

    assert max_active == 3
    assert modules == {
        "transformer": "loaded-transformer",
        "text_encoder": "loaded-text_encoder",
        "vae": "loaded-vae",
    }
    assert pipeline.memory_usages == {
        "transformer": 11.0,
        "text_encoder": 12.0,
        "vae": 3.0,
    }


def test_native_loads_overlap_outside_model_construction_contexts():
    class NativeLoader(ComponentLoader):
        def load_customized(self, *args, **kwargs):
            raise NotImplementedError

    loader = NativeLoader()
    barrier = threading.Barrier(2)
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def load_native(*args, **kwargs):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        with active_lock:
            active -= 1
        return object()

    loader.load_native = load_native

    def run(name):
        return loader._load_native_with_context(
            f"/model/{name}", object(), name, "transformers", None, name, False
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, ["text_encoder", "text_encoder_2"]))

    assert max_active == 2


def test_parallel_loading_can_be_disabled():
    pipeline, args = _pipeline_and_args(parallel_loading=False)
    thread_ids = []

    def load_component(*, component_name, **_kwargs):
        thread_ids.append((component_name, threading.get_ident()))
        return component_name, 0.0

    with (
        patch.object(pipeline, "_load_config", return_value=_model_index()),
        patch.object(current_platform, "get_available_gpu_memory", return_value=16.0),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base."
            "PipelineComponentLoader.load_component",
            side_effect=load_component,
        ),
    ):
        pipeline.load_modules(args)

    assert [name for name, _ in thread_ids] == [
        "transformer",
        "text_encoder",
        "vae",
    ]
    assert {thread_id for _, thread_id in thread_ids} == {threading.get_ident()}


def test_parallel_loading_is_disabled_for_distributed_execution():
    pipeline, args = _pipeline_and_args()
    thread_ids = []

    def load_component(*, component_name, **_kwargs):
        thread_ids.append(threading.get_ident())
        return component_name, 0.0

    with (
        patch.object(pipeline, "_load_config", return_value=_model_index()),
        patch.object(current_platform, "get_available_gpu_memory", return_value=16.0),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base."
            "PipelineComponentLoader.load_component",
            side_effect=load_component,
        ),
    ):
        pipeline.load_modules(args)

    assert set(thread_ids) == {threading.get_ident()}


def test_parallel_loading_propagates_component_failure():
    pipeline, args = _pipeline_and_args()

    def load_component(*, component_name, **_kwargs):
        if component_name == "text_encoder":
            raise RuntimeError("broken text encoder")
        return component_name, 0.0

    with (
        patch.object(pipeline, "_load_config", return_value=_model_index()),
        patch.object(current_platform, "get_available_gpu_memory", return_value=16.0),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base."
            "PipelineComponentLoader.load_component",
            side_effect=load_component,
        ),
        pytest.raises(RuntimeError, match="broken text encoder"),
    ):
        pipeline.load_modules(args)


def test_default_dtype_contexts_are_serialized_between_loader_threads():
    entered = threading.Event()
    release = threading.Event()
    observations = []

    def first():
        with set_default_torch_dtype(torch.float64):
            observations.append(("first", torch.get_default_dtype()))
            entered.set()
            release.wait(timeout=2)

    def second():
        entered.wait(timeout=2)
        with set_default_torch_dtype(torch.float16):
            observations.append(("second", torch.get_default_dtype()))

    original = torch.get_default_dtype()
    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    entered.wait(timeout=2)
    time.sleep(0.05)
    assert observations == [("first", torch.float64)]
    release.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert observations == [
        ("first", torch.float64),
        ("second", torch.float16),
    ]
    assert torch.get_default_dtype() == original


def test_parallel_loading_cli_defaults_on_and_accepts_opt_out():
    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)

    defaults, _ = parser.parse_known_args([])
    disabled, _ = parser.parse_known_args(["--parallel-loading", "false"])

    assert defaults.parallel_loading is True
    assert disabled.parallel_loading is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_parallel_loaded_gpu_modules_match_numpy_reference():
    pipeline, args = _pipeline_and_args()
    barrier = threading.Barrier(3)
    weights = {
        "transformer": np.arange(12, dtype=np.float32).reshape(3, 4) / 10,
        "text_encoder": np.arange(8, dtype=np.float32).reshape(2, 4) / 7,
        "vae": np.arange(4, dtype=np.float32).reshape(1, 4) / 3,
    }

    def load_component(*, component_name, **_kwargs):
        barrier.wait(timeout=5)
        layer = torch.nn.Linear(4, weights[component_name].shape[0], bias=False)
        layer.weight.data.copy_(torch.from_numpy(weights[component_name]))
        return layer.to("cuda"), 0.0

    with (
        patch.object(pipeline, "_load_config", return_value=_model_index()),
        patch.object(current_platform, "get_available_gpu_memory", return_value=16.0),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base."
            "PipelineComponentLoader.load_component",
            side_effect=load_component,
        ),
    ):
        modules = pipeline.load_modules(args)

    input_array = np.array([[0.25, -0.5, 1.5, 2.0]], dtype=np.float32)
    input_tensor = torch.from_numpy(input_array).to("cuda")
    for name, module in modules.items():
        actual = module(input_tensor).detach().cpu().numpy()
        expected = input_array @ weights[name].T
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
