TRTLLM_MLA_MAX_BATCH_SIZE = 65535


def validate_trtllm_mla_batch_size(batch_size: int) -> None:
    """Reject TRT-LLM MLA launches that exceed CUDA's gridDim.z limit."""
    if batch_size > TRTLLM_MLA_MAX_BATCH_SIZE:
        raise ValueError(
            f"DSA trtllm backend got {batch_size} query rows in one forward, "
            f"exceeding the CUDA gridDim.z limit of {TRTLLM_MLA_MAX_BATCH_SIZE}. "
            "This path launches one row per token (tokens are flattened into the "
            "batch dimension); the launch would otherwise fail silently and leave "
            "the attention output uninitialized. Keep aggregate rows per forward "
            "at or below 65535, e.g. via chunked prefill."
        )
