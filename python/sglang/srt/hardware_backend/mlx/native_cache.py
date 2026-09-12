"""Pure-Python policy helpers for MLX model-native cache bridges."""

from typing import Any


def uses_model_native_cache(model: Any) -> bool:
    """Recognize mlx-lm's public Gemma 4 wrapper and text model."""
    candidates = (
        getattr(model, "model_type", None),
        getattr(getattr(model, "args", None), "model_type", None),
        getattr(getattr(model, "language_model", None), "model_type", None),
        getattr(
            getattr(getattr(model, "language_model", None), "args", None),
            "model_type",
            None,
        ),
    )
    return any(value in {"gemma4", "gemma4_text"} for value in candidates)


def validate_model_native_cache_config(
    *,
    disable_radix_cache: bool,
    disable_overlap_schedule: bool,
    chunked_prefill_size: int,
) -> None:
    """Reject scheduler modes that cannot preserve Gemma 4 cache semantics."""
    unsupported = []
    if not disable_radix_cache:
        unsupported.append("--disable-radix-cache")
    if not disable_overlap_schedule:
        unsupported.append("--disable-overlap-schedule")
    if chunked_prefill_size != -1:
        unsupported.append("--chunked-prefill-size=-1")
    if unsupported:
        raise ValueError(
            "Gemma 4 on the MLX backend currently uses model-native "
            "mlx-lm caches for correctness. Start it with "
            + ", ".join(unsupported)
            + ". Radix/prefix reuse, overlap scheduling, and chunked "
            "prefill are not supported by this MVP."
        )
