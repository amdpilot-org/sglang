"""Loading for a DSpark draft-model LoRA adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.runtime_context import get_lora, get_spec

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


_DSPARK_INTERNAL_LORA_NAME = "__dspark_draft__"
_DSPARK_LORA_POOL_CAPACITY = 2  # base slot plus the fixed draft adapter


def init_dspark_lora_manager(model_runner: ModelRunner) -> tuple[LoRAManager, str]:
    """Load and pin the DSpark-only adapter on the draft model runner."""

    lora_path = get_spec().speculative_dspark_lora_path
    if lora_path is None:
        raise ValueError("DSpark draft LoRA initialization requires an adapter path.")

    dspark_ref = LoRARef(
        lora_id=LoRARef.deterministic_id(_DSPARK_INTERNAL_LORA_NAME, lora_path),
        lora_name=_DSPARK_INTERNAL_LORA_NAME,
        lora_path=lora_path,
        pinned=True,
    )
    manager = LoRAManager(
        base_model=model_runner.model,
        base_hf_config=model_runner.model_config.hf_config,
        max_loras_per_batch=_DSPARK_LORA_POOL_CAPACITY,
        load_config=model_runner.load_config,
        dtype=model_runner.dtype,
        server_args=model_runner.server_args,
        lora_backend=get_lora().lora_backend,
        tp_size=model_runner.ps.tp_size,
        tp_rank=model_runner.ps.tp_rank,
        max_lora_rank=None,
        target_modules=None,
        lora_paths=[dspark_ref],
    )
    manager.fetch_new_loras({dspark_ref.lora_id})
    return manager, dspark_ref.lora_id
