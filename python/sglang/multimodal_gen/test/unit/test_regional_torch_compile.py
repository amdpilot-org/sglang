from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2VideoTransformer3DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.utils.compile_trajectory import (
    CompiledPlanManifest,
    CompilePlanResolution,
    CompilePlanResolver,
    CompileWorkloadSignature,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    CompiledModuleRegistry,
    build_torch_compile_kwargs,
    compile_matching_submodules,
    region_inventory_digest,
)


class _CompilableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.compile_calls = []

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)
        self._compiled_call_impl = object()


class _RegionalModel(_CompilableModule):
    _compile_conditions = [
        lambda name, _module: (
            name.startswith("transformer_blocks.") and name.count(".") == 1
        )
    ]

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_CompilableModule(), _CompilableModule()]
        )
        self.transformer_blocks[0].inner = _CompilableModule()
        self.proj_out = _CompilableModule()


@pytest.mark.parametrize(
    ("backend", "options", "expected"),
    [
        (
            "custom_backend",
            {"pass_manager_config": {"persistent_buffers": ["weight"]}},
            {
                "backend": "custom_backend",
                "options": {"pass_manager_config": {"persistent_buffers": ["weight"]}},
            },
        ),
        (
            "inductor",
            {"max_autotune": True},
            {"backend": "inductor", "options": {"max_autotune": True}},
        ),
        (
            "inductor",
            None,
            {"backend": "inductor", "mode": "max-autotune-no-cudagraphs"},
        ),
    ],
)
def test_out_of_tree_platform_controls_compile_kwargs(backend, options, expected):
    """Out-of-tree hooks select valid backend, mode, and option combinations."""
    module = _CompilableModule()
    with (
        patch(
            "sglang.multimodal_gen.runtime.utils.torch_compile."
            "current_platform.is_out_of_tree",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.utils.torch_compile."
            "current_platform.get_compile_backend",
            return_value=backend,
        ) as get_compile_backend,
        patch(
            "sglang.multimodal_gen.runtime.utils.torch_compile."
            "current_platform.get_compile_options",
            return_value=options,
        ) as get_compile_options,
    ):
        compile_kwargs = build_torch_compile_kwargs(
            mode="max-autotune-no-cudagraphs",
            module=module,
        )

    assert compile_kwargs == {
        "dynamic": None,
        "fullgraph": False,
        **expected,
    }
    get_compile_backend.assert_called_once_with("max-autotune-no-cudagraphs")
    get_compile_options.assert_called_once_with(module)


def test_ltx2_compile_conditions_match_only_direct_blocks():
    conditions = LTX2VideoTransformer3DModel._compile_conditions

    assert conditions
    assert any(condition("transformer_blocks.0", object()) for condition in conditions)
    assert not any(
        condition("transformer_blocks.0.attn1", object()) for condition in conditions
    )
    assert not any(
        condition("transformer_blocks", object()) for condition in conditions
    )


def test_compile_matching_submodules_matches_only_declared_regions():
    model = _RegionalModel()

    count = compile_matching_submodules(
        model,
        compile_kwargs={"mode": "default", "fullgraph": False},
    )

    assert count == 2
    assert [len(block.compile_calls) for block in model.transformer_blocks] == [1, 1]
    assert not model.transformer_blocks[0].inner.compile_calls
    assert not model.proj_out.compile_calls
    assert not model.compile_calls


def test_compile_matching_submodules_fails_when_no_region_matches():
    model = _RegionalModel()
    model._compile_conditions = [lambda _name, _module: False]

    with pytest.raises(ValueError, match="no matching submodules"):
        compile_matching_submodules(model, compile_kwargs={"mode": "default"})


def test_compiled_module_registry_installs_regions_once():
    model = _RegionalModel()
    registry = CompiledModuleRegistry()

    assert (
        registry.compile_regions_once(
            model,
            compile_kwargs={"mode": "default"},
        )
        == 2
    )
    assert (
        registry.compile_regions_once(
            model,
            compile_kwargs={"mode": "default"},
        )
        == 0
    )
    assert [len(block.compile_calls) for block in model.transformer_blocks] == [1, 1]
    assert registry.region_inventory(model) == (
        "transformer_blocks.0",
        "transformer_blocks.1",
    )
    assert registry.region_digest(model) == region_inventory_digest(
        ("transformer_blocks.0", "transformer_blocks.1")
    )
    registry.set_regions_active(model, False)
    assert all(block._compiled_call_impl is None for block in model.transformer_blocks)
    registry.set_regions_active(model, True)
    assert all(
        block._compiled_call_impl is not None for block in model.transformer_blocks
    )


def test_denoising_stage_selects_regional_compile():
    model = _RegionalModel()
    compile_kwargs = {"backend": "custom_backend"}
    stage = DenoisingStage.__new__(DenoisingStage)
    stage.server_args = SimpleNamespace(
        enable_breakable_cuda_graph=False,
        enable_torch_compile=True,
        regional_compile=True,
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(torch_compile_mode="default")
        ),
    )
    stage._cache_dit_enabled = False
    stage._torch_compile_registry = CompiledModuleRegistry()

    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
            "current_platform.is_npu",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
            "maybe_enable_inductor_compute_comm_overlap"
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
            "build_torch_compile_kwargs",
            return_value=compile_kwargs,
        ) as build_compile_kwargs,
    ):
        stage._maybe_torch_compile(model)

    build_compile_kwargs.assert_called_once_with(mode="default", module=model)
    assert [len(block.compile_calls) for block in model.transformer_blocks] == [1, 1]
    assert [block.compile_calls for block in model.transformer_blocks] == [
        [compile_kwargs],
        [compile_kwargs],
    ]
    assert not model.compile_calls


def test_denoising_stage_keeps_eager_path_for_uncovered_plan():
    model = _RegionalModel()
    stage = DenoisingStage.__new__(DenoisingStage)
    stage.server_args = SimpleNamespace(
        enable_breakable_cuda_graph=False,
        enable_torch_compile=True,
        regional_compile=True,
    )
    stage._cache_dit_enabled = False
    stage._torch_compile_registry = CompiledModuleRegistry()

    stage._maybe_torch_compile(
        model,
        CompilePlanResolution(None, "workload_signature_mismatch"),
    )

    assert not model.compile_calls


def test_request_signature_resolves_manifest_and_rejects_shape_drift():
    model = _RegionalModel()
    regions = ("transformer_blocks.0", "transformer_blocks.1")
    signature = CompileWorkloadSignature(
        model_revision="model@sha",
        dtype="float32",
        backend="inductor",
        parallel_signature="tp=1,sp=1,cfg=1",
        latent_shape_regime=(1, 4, 8, 8),
        num_inference_steps=4,
        cfg_mode="off",
        cache_mode="off",
        state_schema_version="stateless-v1",
    )
    manifest = CompiledPlanManifest(
        signature=signature,
        regions=regions,
        compile_options={"backend": "inductor"},
        gate_digest="gate-sha",
        status="validated",
        region_digest=region_inventory_digest(regions),
    )
    stage = DenoisingStage.__new__(DenoisingStage)
    stage.server_args = SimpleNamespace(
        compile_gate_digest="gate-sha",
        compile_model_revision="model@sha",
        compile_state_schema_version="stateless-v1",
        regional_compile=True,
        tp_size=1,
        sp_degree=1,
        cfg_parallel_degree=1,
    )
    stage._compile_plan_resolver = CompilePlanResolver([manifest])
    stage._torch_compile_registry = CompiledModuleRegistry()
    stage._cache_dit_enabled = False
    batch = SimpleNamespace(
        raw_latent_shape=(1, 4, 8, 8),
        latents=torch.zeros(1, 4, 8, 8),
        num_inference_steps=4,
        do_classifier_free_guidance=False,
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
        "current_platform.is_npu",
        return_value=False,
    ):
        assert stage._resolve_compile_plan(model, batch).use_compiled
        batch.raw_latent_shape = (1, 4, 16, 16)
        resolution = stage._resolve_compile_plan(model, batch)

    assert not resolution.use_compiled
    assert resolution.fallback_reason == "workload_signature_mismatch"
    assert all(not block.compile_calls for block in model.transformer_blocks)
