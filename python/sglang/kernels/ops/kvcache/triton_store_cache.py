from typing import Literal

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.attention.dsa.utils import (
    INDEXER_K_CACHE_PRESHUFFLE_TILE,
    aiter_can_use_preshuffle_paged_mqa,
)

_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
_FP8_INFO = torch.finfo(_FP8_DTYPE)

# DeepSeek-V4 MLA paged FP8 cache layout
_MLA_HEAD_DIM = 512  # full MLA token dim (elements per input row)
_MLA_NOPE_DIM = 448  # nope sub-dim (elements)
_MLA_TILE_SIZE = 64  # FP8 tile width (also rope copy stride)
_MLA_SLOT_BYTES = 576  # bytes per slot in the paged FP8 cache
_MLA_BF16_SLOT_ELEMS = _MLA_SLOT_BYTES // 2  # bf16-view slot stride (elements)
_MLA_BF16_ROPE_OFFSET = _MLA_NOPE_DIM // 2  # bf16-view rope offset (elements)
_MLA_SCALES_PER_TOKEN = 8  # UE8M0 scales per token (7 nope tiles + 1 padding)
_MLA_NUM_TILES = 8  # 7 nope quant tiles + 1 rope copy tile
_MLA_ROPE_TILE_ID = 7  # tile id reserved for the rope copy

# C4 indexer paged FP8 cache layout
_INDEXER_HEAD_DIM = 128

_ALIGNMENT_BYTES = 16

_UE8M0_EXPONENT_BIAS = 127


@triton.jit(
    do_not_specialize=[
        "k_ptr",
        "v_ptr",
        "k_cache_ptr",
        "v_cache_ptr",
        "indices_ptr",
        "k_scale_ptr",
        "v_scale_ptr",
        "k_scale_value",
        "v_scale_value",
        "size_limit",
    ]
)
def _triton_store_cache_fp8_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    k_scale_ptr,
    v_scale_ptr,
    k_scale_value,
    v_scale_value,
    size_limit,
    K_ROW_DIM: tl.constexpr,
    V_ROW_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_K_SCALE: tl.constexpr,
    USE_V_SCALE: tl.constexpr,
    K_SCALE_IS_PTR: tl.constexpr,
    V_SCALE_IS_PTR: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    loc = tl.load(indices_ptr + token_id).to(tl.int64)
    tl.device_assert(loc >= 0 and loc < size_limit, "KV cache index out of bounds")
    if loc == 0:
        return

    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    k_mask = offsets < K_ROW_DIM
    v_mask = offsets < V_ROW_DIM

    k = tl.load(
        k_ptr + token_id * K_ROW_DIM + offsets, mask=k_mask, other=0.0
    ).to(tl.float32)
    v = tl.load(
        v_ptr + token_id * V_ROW_DIM + offsets, mask=v_mask, other=0.0
    ).to(tl.float32)

    if USE_K_SCALE:
        if K_SCALE_IS_PTR:
            k = tl.div_rn(k, tl.load(k_scale_ptr).to(tl.float32))
        else:
            k = tl.div_rn(k, k_scale_value)
    if USE_V_SCALE:
        if V_SCALE_IS_PTR:
            v = tl.div_rn(v, tl.load(v_scale_ptr).to(tl.float32))
        else:
            v = tl.div_rn(v, v_scale_value)

    k = tl.clamp(k, FP8_MIN, FP8_MAX).to(k_cache_ptr.dtype.element_ty)
    v = tl.clamp(v, FP8_MIN, FP8_MAX).to(v_cache_ptr.dtype.element_ty)
    tl.store(
        k_cache_ptr + loc * K_ROW_DIM + offsets, k, mask=k_mask
    )
    tl.store(
        v_cache_ptr + loc * V_ROW_DIM + offsets, v, mask=v_mask
    )


def _valid_fp8_scale(scale, device: torch.device) -> bool:
    return (
        scale is None
        or isinstance(scale, (float, int))
        or (
            isinstance(scale, torch.Tensor)
            and scale.numel() == 1
            and scale.dtype == torch.float32
            and scale.device == device
            and scale.is_contiguous()
        )
    )


def _is_16_byte_aligned(*tensors: torch.Tensor) -> bool:
    return all(tensor.data_ptr() % _ALIGNMENT_BYTES == 0 for tensor in tensors)


def try_triton_store_cache_fp8(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k_scale,
    v_scale,
) -> bool:
    """Try a one-launch BF16/FP16-to-FP8 scale/cast/scatter store.

    The caller must use the unfused path when this returns False. The conditions
    are deliberately conservative: NHD rows must be contiguous and 16-byte
    aligned, and device scales must be one-element float32 tensors.
    """
    if (
        k.dtype not in (torch.bfloat16, torch.float16)
        or _FP8_DTYPE != torch.float8_e4m3fnuz
        or k_cache.dtype != torch.uint8
        or v_cache.dtype != torch.uint8
        or k.ndim != 3
        or v.ndim != 3
        or k.shape[0] != v.shape[0]
        or k.shape[1] != v.shape[1]
        or k.device != v.device
        or k.device != k_cache.device
        or k.device != v_cache.device
        or k.device != indices.device
        or indices.ndim != 1
        or indices.shape[0] != k.shape[0]
        or indices.dtype not in (torch.int32, torch.int64)
        or not (
            k.is_contiguous()
            and v.is_contiguous()
            and k_cache.is_contiguous()
            and v_cache.is_contiguous()
            and indices.is_contiguous()
        )
        or not _is_16_byte_aligned(k, v, k_cache, v_cache, indices)
        or not _valid_fp8_scale(k_scale, k.device)
        or not _valid_fp8_scale(v_scale, v.device)
    ):
        return False

    k_row_dim = k.shape[1] * k.shape[2]
    v_row_dim = v.shape[1] * v.shape[2]
    if (
        k_row_dim == 0
        or v_row_dim == 0
        or k_cache.shape[1:] != k.shape[1:]
        or v_cache.shape[1:] != v.shape[1:]
        or k_row_dim * k.element_size() % _ALIGNMENT_BYTES != 0
        or v_row_dim * v.element_size() % _ALIGNMENT_BYTES != 0
        or k_row_dim * k_cache.element_size() % _ALIGNMENT_BYTES != 0
        or v_row_dim * v_cache.element_size() % _ALIGNMENT_BYTES != 0
    ):
        return False

    num_tokens = k.shape[0]
    if num_tokens == 0:
        return True

    k_cache_rows = k_cache.view(_FP8_DTYPE).view(-1, k_row_dim)
    v_cache_rows = v_cache.view(_FP8_DTYPE).view(-1, v_row_dim)
    block = triton.next_power_of_2(max(k_row_dim, v_row_dim))
    k_scale_ptr = k_scale if isinstance(k_scale, torch.Tensor) else k
    v_scale_ptr = v_scale if isinstance(v_scale, torch.Tensor) else v
    k_scale_value = (
        1.0
        if k_scale is None or isinstance(k_scale, torch.Tensor)
        else float(k_scale)
    )
    v_scale_value = (
        1.0
        if v_scale is None or isinstance(v_scale, torch.Tensor)
        else float(v_scale)
    )

    grid = (num_tokens, triton.cdiv(max(k_row_dim, v_row_dim), block))
    args = (
        k,
        v,
        k_cache_rows,
        v_cache_rows,
        indices,
        k_scale_ptr,
        v_scale_ptr,
        k_scale_value,
        v_scale_value,
        k_cache_rows.shape[0],
    )
    kwargs = {
        "K_ROW_DIM": k_row_dim,
        "V_ROW_DIM": v_row_dim,
        "BLOCK": block,
        "USE_K_SCALE": k_scale is not None,
        "USE_V_SCALE": v_scale is not None,
        "K_SCALE_IS_PTR": isinstance(k_scale, torch.Tensor),
        "V_SCALE_IS_PTR": isinstance(v_scale, torch.Tensor),
        "FP8_MIN": _FP8_INFO.min,
        "FP8_MAX": _FP8_INFO.max,
    }
    _triton_store_cache_fp8_kernel[grid](*args, **kwargs, num_warps=8)
    return True


@triton.jit
def _triton_fused_store_flashmla_kernel(
    input_ptr,
    cache_fp8_ptr,
    cache_bf16_ptr,
    cache_u8_ptr,
    indices_ptr,
    N,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    BYTES_PER_PAGE_BF16: tl.constexpr,
    S_OFFSET: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    SLOT_BYTES: tl.constexpr,
    BF16_SLOT_ELEMS: tl.constexpr,
    BF16_ROPE_OFFSET: tl.constexpr,
    SCALES_PER_TOKEN: tl.constexpr,
    ROPE_TILE_ID: tl.constexpr,
    UE8M0_BIAS: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    if token_id >= N:
        return

    loc = tl.load(indices_ptr + token_id).to(tl.int32)
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    if tile_id == ROPE_TILE_ID:
        rope_lane = tl.arange(0, TILE_SIZE)
        rope_vals = tl.load(input_ptr + token_id * HEAD_DIM + NOPE_DIM + rope_lane)
        rope_bf16_offset = (
            page * BYTES_PER_PAGE_BF16
            + slot * BF16_SLOT_ELEMS
            + BF16_ROPE_OFFSET
            + rope_lane
        )
        tl.store(cache_bf16_ptr + rope_bf16_offset, rope_vals)
    else:
        tile_lane = tl.arange(0, TILE_SIZE)
        x_bf16 = tl.load(
            input_ptr + token_id * HEAD_DIM + tile_id * TILE_SIZE + tile_lane
        )
        x_fp32 = x_bf16.to(tl.float32)

        abs_max = tl.max(tl.abs(x_fp32))
        scale = tl.maximum(abs_max, EPS) / FP8_MAX

        # cast scale to ue8m0 format
        log2_scale = tl.log2(scale)
        ceil_log2 = tl.math.ceil(log2_scale)
        inv_scale = tl.exp2(-ceil_log2)

        x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
            cache_fp8_ptr.dtype.element_ty
        )

        nope_offset = (
            page * BYTES_PER_PAGE + slot * SLOT_BYTES + tile_id * TILE_SIZE + tile_lane
        )
        tl.store(cache_fp8_ptr + nope_offset, x_fp8)

        ue8m0 = (ceil_log2.to(tl.int32) + UE8M0_BIAS).to(tl.uint8)
        scale_offset = (
            page * BYTES_PER_PAGE + S_OFFSET + slot * SCALES_PER_TOKEN + tile_id
        )
        tl.store(cache_u8_ptr + scale_offset, ue8m0)


def triton_fused_store_flashmla(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    """Fused FP8 quantise + paged scatter for the SWA (flashmla) KV cache."""
    N = input.shape[0]
    if N == 0:
        return

    bytes_per_page = cache.shape[1]
    cache_fp8 = cache.view(_FP8_DTYPE)
    cache_bf16 = cache.view(torch.bfloat16)
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices

    _triton_fused_store_flashmla_kernel[(N, _MLA_NUM_TILES)](
        input,
        cache_fp8,
        cache_bf16,
        cache,
        indices_i32,
        N,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        BYTES_PER_PAGE_BF16=bytes_per_page // 2,
        S_OFFSET=page_size * _MLA_SLOT_BYTES,
        TILE_SIZE=_MLA_TILE_SIZE,
        HEAD_DIM=_MLA_HEAD_DIM,
        NOPE_DIM=_MLA_NOPE_DIM,
        SLOT_BYTES=_MLA_SLOT_BYTES,
        BF16_SLOT_ELEMS=_MLA_BF16_SLOT_ELEMS,
        BF16_ROPE_OFFSET=_MLA_BF16_ROPE_OFFSET,
        SCALES_PER_TOKEN=_MLA_SCALES_PER_TOKEN,
        ROPE_TILE_ID=_MLA_ROPE_TILE_ID,
        UE8M0_BIAS=_UE8M0_EXPONENT_BIAS,
        FP8_MIN=_FP8_INFO.min,
        FP8_MAX=_FP8_INFO.max,
        EPS=1e-8,
    )


@triton.jit
def _triton_fused_store_indexer_kernel(
    input_ptr,
    cache_fp8_ptr,
    cache_f32_ptr,
    indices_ptr,
    N,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    BYTES_PER_PAGE_F32: tl.constexpr,
    SCALE_PAGE_OFFSET_F32: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    if token_id >= N:
        return

    loc = tl.load(indices_ptr + token_id).to(tl.int32)
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE

    lane = tl.arange(0, HEAD_DIM)
    x_fp32 = tl.load(input_ptr + token_id * HEAD_DIM + lane).to(tl.float32)

    abs_max = tl.max(tl.abs(x_fp32))
    scale = tl.maximum(abs_max, EPS) / FP8_MAX
    inv_scale = 1.0 / scale

    x_fp8 = tl.clamp(x_fp32 * inv_scale, FP8_MIN, FP8_MAX).to(
        cache_fp8_ptr.dtype.element_ty
    )

    if PRESHUFFLE_TILE:
        token_tile_id = slot // PRESHUFFLE_TILE
        token_in_tile = slot % PRESHUFFLE_TILE
        col_tile_id = lane // PRESHUFFLE_TILE
        col_in_tile = lane % PRESHUFFLE_TILE
        fp8_offset = (
            page * BYTES_PER_PAGE
            + token_tile_id * (PRESHUFFLE_TILE * HEAD_DIM)
            + col_tile_id * (PRESHUFFLE_TILE * PRESHUFFLE_TILE)
            + token_in_tile * PRESHUFFLE_TILE
            + col_in_tile
        )
    else:
        fp8_offset = page * BYTES_PER_PAGE + slot * HEAD_DIM + lane
    tl.store(cache_fp8_ptr + fp8_offset, x_fp8)

    f32_offset = page * BYTES_PER_PAGE_F32 + SCALE_PAGE_OFFSET_F32 + slot
    tl.store(cache_f32_ptr + f32_offset, scale)


def triton_fused_store_indexer(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    """Fused FP8 quantise + paged scatter for the C4 indexer KV cache."""
    N = input.shape[0]
    if N == 0:
        return

    bytes_per_page = cache.shape[1]
    bytes_per_page_f32 = bytes_per_page // 4
    scale_page_offset_f32 = (_INDEXER_HEAD_DIM * page_size) // 4

    cache_fp8 = cache.view(_FP8_DTYPE)
    cache_f32 = cache.view(torch.float32)
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices

    _triton_fused_store_indexer_kernel[(N,)](
        input,
        cache_fp8,
        cache_f32,
        indices_i32,
        N,
        PAGE_SIZE=page_size,
        BYTES_PER_PAGE=bytes_per_page,
        BYTES_PER_PAGE_F32=bytes_per_page_f32,
        SCALE_PAGE_OFFSET_F32=scale_page_offset_f32,
        HEAD_DIM=_INDEXER_HEAD_DIM,
        PRESHUFFLE_TILE=(
            INDEXER_K_CACHE_PRESHUFFLE_TILE
            if aiter_can_use_preshuffle_paged_mqa()
            else 0
        ),
        FP8_MIN=_FP8_INFO.min,
        FP8_MAX=_FP8_INFO.max,
        EPS=1e-8,
    )


def triton_fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    """ROCm dispatch for fused_store_cache()."""
    if type == "flashmla":
        triton_fused_store_flashmla(input, cache, indices, page_size)
    else:
        triton_fused_store_indexer(input, cache, indices, page_size)
