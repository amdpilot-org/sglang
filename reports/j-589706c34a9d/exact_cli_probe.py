"""Offline reproduction of the argument-routing portion of issue #33466."""

import argparse

from sglang.cli.serve import _create_backend_registry
from sglang.cli.serve_backends import ServeRequest
from sglang.cli.utils import get_is_diffusion_model
from sglang.multimodal_gen.registry import get_non_diffusers_pipeline_name
from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
    add_multimodal_gen_serve_args,
)


argv = [
    "--model-path",
    "/mnt/models/MiniMax-H3",
    "--num-gpus",
    "2",
    "--tp-size",
    "2",
    "--ulysses-degree",
    "1",
    "--performance-mode",
    "memory",
    "--layerwise-offload-components",
    "dit,text_encoder,vae",
    "--dit-offload-prefetch-size",
    "1",
    "--dit-layerwise-resident-layers",
    "20",
    "--enable-torch-compile",
    "false",
    "--host",
    "0.0.0.0",
    "--port",
    "30010",
    "--model-variant",
    "fl2va",
]

parser = argparse.ArgumentParser(prog="sglang serve")
add_multimodal_gen_serve_args(parser)
parsed, remaining = parser.parse_known_args(argv)
selected = _create_backend_registry().auto_detect(
    ServeRequest(tuple(argv), "/mnt/models/MiniMax-H3")
).name

print("selected_backend:", selected)
print("remaining_argv:", remaining)
for name in (
    "num_gpus",
    "tp_size",
    "ulysses_degree",
    "performance_mode",
    "layerwise_offload_components",
    "dit_offload_prefetch_size",
    "dit_layerwise_resident_layers",
    "enable_torch_compile",
    "model_variant",
):
    print(f"{name}:", getattr(parsed, name))
print("pipeline:", get_non_diffusers_pipeline_name("/mnt/models/MiniMax-H3"))
print("exact_detected:", get_is_diffusion_model("/mnt/models/MiniMax-H3"))

boundaries = (
    "/mnt/models/MiniMax-H3.5",
    "/mnt/models/MiniMax-H3-4B",
    "/mnt/models/not-MiniMax-H3",
)
for path in boundaries:
    print(
        "boundary:",
        path,
        get_is_diffusion_model(path),
        get_non_diffusers_pipeline_name(path),
    )

assert selected == "diffusion"
assert remaining == []
assert (parsed.num_gpus, parsed.tp_size, parsed.ulysses_degree) == (2, 2, 1)
assert parsed.performance_mode == "memory"
assert parsed.layerwise_offload_components == ["dit,text_encoder,vae"]
assert parsed.dit_offload_prefetch_size == 1
assert parsed.dit_layerwise_resident_layers == 20
assert parsed.enable_torch_compile is False
assert parsed.model_variant == "fl2va"
assert get_non_diffusers_pipeline_name("/mnt/models/MiniMax-H3") == "MiniMaxH3Pipeline"
for path in boundaries:
    assert get_non_diffusers_pipeline_name(path) is None
print("all_assertions_passed: true")
