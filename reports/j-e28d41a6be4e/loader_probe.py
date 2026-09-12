"""Deterministic probe for the loader/interface path in upstream issue 31236."""

from types import SimpleNamespace

from sglang.srt.model_loader.loader import (
    DefaultModelLoader,
    LoadConfig,
    ModelOptModelLoader,
    get_model_loader,
)
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration


config = SimpleNamespace(
    quantization="modelopt_fp4",
    modelopt_quant=None,
    _is_already_quantized=lambda: False,
)
loader = get_model_loader(LoadConfig(), config)
print("current_loader", type(loader).__name__)
print("preserves_sglang_initialization", isinstance(loader, DefaultModelLoader))
print(
    "gemma4_has_get_embed_and_head",
    hasattr(Gemma4ForConditionalGeneration, "get_embed_and_head"),
)

# Pre-4ad990ba7 loader selection admitted every modelopt_fp4 request here.
legacy_would_select_modelopt = config.quantization in {
    "modelopt_fp8",
    "modelopt_fp4",
    "modelopt_mixed",
    "modelopt",
}
print("legacy_would_select", ModelOptModelLoader.__name__, legacy_would_select_modelopt)

for option in (
    "modelopt_checkpoint_restore_path",
    "modelopt_checkpoint_save_path",
    "modelopt_export_path",
):
    selected = get_model_loader(LoadConfig(**{option: "/tmp/modelopt"}), config)
    print("explicit_workflow", option, type(selected).__name__)
