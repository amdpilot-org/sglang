from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    accept_greedy_triton,
    finalize_accept_lens_triton,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildOutTokens,
    build_unified_commit_inject_layout,
    scatter_compact_to_strided_into,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.models.dflash import DFlashAttention, DFlashDraftModel
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)


SEED = 0
WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 30
RING_STRIDE = 256
MAX_POSITION_EMBEDDINGS = 4096


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    stride: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int


CASES = (
    Case("small", 4, 4, 256, 2, 4, 2, 64, 64),
    Case("medium", 8, 4, 512, 2, 4, 2, 64, 128),
    Case("large", 16, 4, 1024, 2, 4, 2, 64, 256),
)


def rms_norm_reference(
    value: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    value = value.double()
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    normalized = value * torch.rsqrt(variance + epsilon)
    return normalized * weight.double()


def rope_reference(
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    key = key.double()
    cos_sin = cos_sin_cache.index_select(0, positions.long()).double()
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    if is_neox_style:
        first, second = torch.chunk(key, 2, dim=-1)
    else:
        first, second = key[..., ::2], key[..., 1::2]
    rotated_first = first * cos - second * sin
    rotated_second = second * cos + first * sin
    if is_neox_style:
        return torch.cat((rotated_first, rotated_second), dim=-1)
    return torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)


def scatter_reference(
    compact: torch.Tensor,
    verify_lens: torch.Tensor,
    stride: int,
    fill_value: float,
) -> torch.Tensor:
    batch_size = verify_lens.shape[0]
    dim = compact.shape[1]
    output = torch.full(
        (batch_size * stride, dim), fill_value, dtype=compact.dtype, device=compact.device
    )
    start = 0
    for row, length in enumerate(verify_lens.tolist()):
        length = int(length)
        output[row * stride : row * stride + length] = compact[start : start + length]
        start += length
    return output


def accept_reference(
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, stride = candidates.shape
    target_predict = torch.argmax(target_logits, dim=-1).view(batch_size, stride)
    correct_len = torch.zeros(batch_size, dtype=torch.int32, device=candidates.device)
    bonus = torch.zeros(batch_size, dtype=torch.int64, device=candidates.device)
    cap_trim_lens = torch.zeros_like(correct_len)
    for row in range(batch_size):
        uncapped = 0
        while (
            uncapped < stride - 1
            and candidates[row, uncapped + 1] == target_predict[row, uncapped]
        ):
            uncapped += 1
        capped = min(uncapped, int(verify_lens[row]) - 1)
        correct_len[row] = capped
        cap_trim_lens[row] = uncapped - capped
        bonus[row] = target_predict[row, capped]
    return correct_len, bonus, cap_trim_lens


def finalize_reference(
    correct_len: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    commit_lens = correct_len.to(torch.int32) + 1
    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
    return commit_lens, new_seq_lens, cap_trim_lens.to(torch.int32)


def out_tokens_reference(
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    stride: int,
    gamma: int,
) -> torch.Tensor:
    output = torch.zeros(
        (draft_tokens.shape[0], stride), dtype=torch.int64, device=draft_tokens.device
    )
    output[:, :gamma] = draft_tokens
    output.scatter_(1, correct_len.to(torch.int64)[:, None], bonus[:, None])
    return output


def commit_layout_reference(
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
    ring_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions_2d = prefix_lens[:, None] + block_pos_offsets[:stride][None, :]
    positions = positions_2d.reshape(-1).to(torch.int64)
    state_slot = req_pool_indices.to(torch.int64)[:, None].expand(-1, stride).reshape(-1)
    loc = state_slot * ring_stride + positions % ring_stride
    column = torch.arange(stride, device=loc.device)[None, :]
    committed = (column < commit_lens.to(torch.long)[:, None]).reshape(-1)
    swa_loc = torch.where(committed, loc, torch.full_like(loc, -1)).to(torch.int32)
    return swa_loc, positions


class SimpleKVPool:
    def __init__(
        self,
        *,
        device: torch.device,
        num_slots: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.num_slots = num_slots
        self.sink_index = num_slots
        shape = (num_slots + 1, num_kv_heads, head_dim)
        self.keys = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.values = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]

    def reset(self) -> None:
        for key in self.keys:
            key.zero_()
        for value in self.values:
            value.zero_()

    def set_kv_buffer(
        self,
        attention: Any,
        cache_loc: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_scale: Any = None,
        v_scale: Any = None,
    ) -> None:
        loc = cache_loc.to(torch.int64)
        self.keys[attention.layer_id].index_copy_(0, loc, key)
        self.values[attention.layer_id].index_copy_(0, loc, value)

    def set_kv_buffer_prefix_valid(
        self,
        attention: Any,
        cache_loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_scale: Any = None,
        v_scale: Any = None,
    ) -> None:
        batch_size, stride = cache_loc_2d.shape
        flat_loc = cache_loc_2d.reshape(-1).to(torch.int64)
        column = torch.arange(stride, device=flat_loc.device)[None, :]
        committed = (column < commit_lens.to(torch.long)[:, None]).reshape(-1)
        safe_loc = torch.where(
            committed, flat_loc, torch.full_like(flat_loc, self.sink_index)
        )
        self.keys[attention.layer_id].index_copy_(0, safe_loc, key)
        self.values[attention.layer_id].index_copy_(0, safe_loc, value)


class ReducedBlock:
    def __init__(self, case: Case, device: torch.device) -> None:
        self.case = case
        self.device = device
        self.gamma = case.stride - 1
        self.num_slots = case.batch_size * RING_STRIDE
        self.dtype = torch.float32

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        self.fc = torch.nn.Linear(
            case.hidden_size, case.hidden_size, bias=True, device=device, dtype=self.dtype
        )
        self.hidden_norm = RMSNorm(case.hidden_size, eps=1e-5).to(device=device, dtype=self.dtype)

        self.layers = []
        for layer_id in range(case.num_layers):
            q_size = case.num_heads * case.head_dim
            kv_size = case.num_kv_heads * case.head_dim
            qkv_proj = torch.nn.Linear(
                case.hidden_size,
                q_size + 2 * kv_size,
                bias=True,
                device=device,
                dtype=self.dtype,
            )
            qkv_proj.quant_method = UnquantizedLinearMethod()
            k_norm = RMSNorm(case.head_dim, eps=1e-5).to(device=device, dtype=self.dtype)
            rotary = RotaryEmbedding(
                head_size=case.head_dim,
                rotary_dim=case.head_dim,
                max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                base=10000,
                is_neox_style=True,
                dtype=self.dtype,
            ).to(device=device, dtype=self.dtype)
            attention = SimpleNamespace(
                qkv_proj=qkv_proj,
                q_size=q_size,
                kv_size=kv_size,
                head_dim=case.head_dim,
                num_kv_heads=case.num_kv_heads,
                rotary_emb=rotary,
                k_norm=k_norm,
                attn=SimpleNamespace(
                    layer_id=layer_id,
                    k_scale=None,
                    v_scale=None,
                ),
            )
            attention.kv_proj_only = MethodType(DFlashAttention.kv_proj_only, attention)
            attention.apply_k_norm = MethodType(DFlashAttention.apply_k_norm, attention)
            attention.apply_k_rope = MethodType(DFlashAttention.apply_k_rope, attention)
            self.layers.append(SimpleNamespace(self_attn=attention))

        self.pool = SimpleKVPool(
            device=device,
            num_slots=self.num_slots,
            num_layers=case.num_layers,
            num_kv_heads=case.num_kv_heads,
            head_dim=case.head_dim,
            dtype=self.dtype,
        )
        self.draft_model = SimpleNamespace(
            layers=self.layers,
            fc=self.fc,
            hidden_norm=self.hidden_norm,
            is_nemotron_35_draft=False,
            _fused_kv_write_cache=None,
        )
        self.draft_model.project_target_hidden = MethodType(
            DFlashDraftModel.project_target_hidden, self.draft_model
        )
        self.draft_model._fused_kv_write_bundle = MethodType(
            DSparkDraftMixin._fused_kv_write_bundle, self.draft_model
        )
        self.draft_model._build_fused_kv_write_bundle = MethodType(
            DSparkDraftMixin._build_fused_kv_write_bundle, self.draft_model
        )
        self.draft_model._stacked_ctx_kv_params = MethodType(
            DSparkDraftMixin._stacked_ctx_kv_params, self.draft_model
        )
        self.draft_model._project_ctx_kv_stacked = MethodType(
            DSparkDraftMixin._project_ctx_kv_stacked, self.draft_model
        )
        self.draft_model.write_target_hidden_kv = MethodType(
            DSparkDraftMixin.write_target_hidden_kv, self.draft_model
        )
        self.draft_model_runner = SimpleNamespace(token_to_kv_pool=self.pool)
        self.model_runner = SimpleNamespace(device=device)
        self.injector = TargetHiddenKvInjector(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            model_runner=self.model_runner,
            device=device,
            verify_num_draft_tokens=case.stride,
            block_pos_offsets=torch.arange(case.stride, device=device),
        )

        token_count = case.batch_size * case.stride
        self.compact_hidden = torch.randn(
            token_count, case.hidden_size, device=device, dtype=self.dtype
        )
        self.compact_logits = torch.randn(
            token_count, case.vocab_size, device=device, dtype=self.dtype
        )
        self.compact_candidates = torch.randint(
            0,
            case.vocab_size,
            (token_count, 1),
            device=device,
            dtype=torch.int64,
        )
        self.draft_tokens = torch.randint(
            0,
            case.vocab_size,
            (case.batch_size, self.gamma),
            device=device,
            dtype=torch.int64,
        )
        self.verify_lens = torch.randint(
            1, case.stride + 1, (case.batch_size,), device=device, dtype=torch.int32
        )
        self.prefix_lens = torch.randint(
            1, 1024, (case.batch_size,), device=device, dtype=torch.int64
        )
        self.req_pool_indices = torch.arange(
            case.batch_size, device=device, dtype=torch.int64
        )
        self.initial_positions = torch.arange(token_count, device=device, dtype=torch.int64)
        self.initial_cache_loc = torch.arange(token_count, device=device, dtype=torch.int64)

        self.projected_compact_hidden = torch.empty_like(self.compact_hidden)
        self.strided_hidden = torch.empty_like(self.compact_hidden)
        self.projected_strided_hidden = torch.empty_like(self.compact_hidden)
        self.strided_logits = torch.empty_like(self.compact_logits)
        self.strided_candidates = torch.empty_like(self.compact_candidates)
        self.correct_len = torch.empty(
            case.batch_size, device=device, dtype=torch.int32
        )
        self.bonus = torch.empty(case.batch_size, device=device, dtype=torch.int64)
        self.cap_trim_lens = torch.empty_like(self.correct_len)
        self.commit_lens = torch.empty_like(self.correct_len)
        self.new_seq_lens = torch.empty_like(self.prefix_lens)
        self.out_tokens = torch.empty(
            (case.batch_size, case.stride), device=device, dtype=torch.int64
        )
        self.commit_swa_loc = torch.empty(
            case.batch_size * case.stride, device=device, dtype=torch.int32
        )
        self.commit_positions = torch.empty(
            case.batch_size * case.stride, device=device, dtype=torch.int64
        )

    def run(self) -> None:
        with torch.inference_mode():
            case = self.case
            token_count = case.batch_size * case.stride
            self.projected_compact_hidden.copy_(self.draft_model.project_target_hidden(self.compact_hidden))
            self.injector.inject_target_hidden(
                target_hidden=self.projected_compact_hidden,
                cache_loc=self.initial_cache_loc,
                positions=self.initial_positions,
                target_hidden_is_projected=True,
            )
            scatter_compact_to_strided_into(
                compact=self.compact_hidden,
                verify_lens=self.verify_lens,
                out=self.strided_hidden,
                stride=case.stride,
                fill_value=0.0,
            )
            self.projected_strided_hidden.copy_(
                self.draft_model.project_target_hidden(self.strided_hidden)
            )
            scatter_compact_to_strided_into(
                compact=self.compact_logits,
                verify_lens=self.verify_lens,
                out=self.strided_logits,
                stride=case.stride,
                fill_value=0.0,
            )
            scatter_compact_to_strided_into(
                compact=self.compact_candidates,
                verify_lens=self.verify_lens,
                out=self.strided_candidates,
                stride=case.stride,
                fill_value=0,
            )
            self.correct_len, self.bonus, self.cap_trim_lens = accept_greedy_triton(
                candidates=self.strided_candidates.view(case.batch_size, case.stride),
                target_logits=self.strided_logits,
                verify_num_draft_tokens=case.stride,
                cutoff_verify_lens=self.verify_lens,
            )
            finalized = finalize_accept_lens_triton(
                correct_len=self.correct_len,
                cap_trim_lens=self.cap_trim_lens,
                prefix_lens=self.prefix_lens,
            )
            self.commit_lens.copy_(finalized.commit_lens)
            self.new_seq_lens.copy_(finalized.new_seq_lens)
            self.out_tokens.copy_(
                BuildOutTokens.execute(
                    draft_tokens=self.draft_tokens,
                    correct_len=self.correct_len,
                    bonus=self.bonus,
                    verify_num_draft_tokens=case.stride,
                    gamma=self.gamma,
                )
            )
            commit_layout = build_unified_commit_inject_layout(
                req_pool_indices=self.req_pool_indices,
                prefix_lens=self.prefix_lens,
                block_pos_offsets=self.injector._block_pos_offsets,
                commit_lens=self.commit_lens,
                stride=case.stride,
                ring_stride=RING_STRIDE,
            )
            self.commit_swa_loc.copy_(commit_layout.swa_loc)
            self.commit_positions.copy_(commit_layout.positions)
            self.draft_model.write_target_hidden_kv(
                target_hidden=self.projected_strided_hidden,
                pool=self.pool,
                positions=self.commit_positions,
                cache_loc=self.commit_swa_loc,
                cache_loc_2d=self.commit_swa_loc.view(case.batch_size, case.stride),
                commit_lens=self.commit_lens,
                target_hidden_is_projected=True,
            )


def reference_outputs(block: ReducedBlock) -> dict[str, torch.Tensor]:
    case = block.case
    device = block.device
    compact_hidden = block.compact_hidden.double().cpu()
    compact_logits = block.compact_logits.double().cpu()
    compact_candidates = block.compact_candidates.cpu()
    draft_tokens = block.draft_tokens.cpu()
    verify_lens = block.verify_lens.cpu()
    prefix_lens = block.prefix_lens.cpu()
    req_pool_indices = block.req_pool_indices.cpu()
    initial_positions = block.initial_positions.cpu()
    initial_cache_loc = block.initial_cache_loc.cpu()

    fc_weight = block.fc.weight.detach().double().cpu()
    fc_bias = block.fc.bias.detach().double().cpu()
    hidden_weight = block.hidden_norm.weight.detach().double().cpu()
    projected_compact = torch.nn.functional.linear(compact_hidden, fc_weight, fc_bias)
    projected_compact = rms_norm_reference(projected_compact, hidden_weight, 1e-5)

    reference_pool_keys = []
    reference_pool_values = []
    for layer in block.layers:
        attention = layer.self_attn
        q_size = attention.q_size
        kv_size = attention.kv_size
        weight = attention.qkv_proj.weight.detach().double().cpu()
        bias = attention.qkv_proj.bias.detach().double().cpu()
        kv_weight = weight[q_size : q_size + 2 * kv_size]
        kv_bias = bias[q_size : q_size + 2 * kv_size]
        kv = torch.nn.functional.linear(projected_compact, kv_weight, kv_bias)
        key, value = kv.split([kv_size, kv_size], dim=-1)
        key = key.reshape(-1, attention.num_kv_heads, attention.head_dim)
        value = value.reshape(-1, attention.num_kv_heads, attention.head_dim)
        key = rms_norm_reference(
            key.reshape(-1, attention.head_dim),
            attention.k_norm.weight.detach().double().cpu(),
            1e-5,
        ).reshape(-1, attention.num_kv_heads, attention.head_dim)
        key = rope_reference(
            key,
            initial_positions,
            attention.rotary_emb.cos_sin_cache.cpu(),
            True,
        )
        reference_pool_keys.append((attention.attn.layer_id, initial_cache_loc, key))
        reference_pool_values.append((attention.attn.layer_id, initial_cache_loc, value))

    strided_hidden = scatter_reference(
        compact_hidden, verify_lens, case.stride, 0.0
    )
    projected_strided = torch.nn.functional.linear(strided_hidden, fc_weight, fc_bias)
    projected_strided = rms_norm_reference(projected_strided, hidden_weight, 1e-5)
    strided_logits = scatter_reference(compact_logits, verify_lens, case.stride, 0.0)
    strided_candidates = scatter_reference(compact_candidates, verify_lens, case.stride, 0)
    candidates = strided_candidates.view(case.batch_size, case.stride)
    correct_len, bonus, cap_trim_lens = accept_reference(
        candidates, strided_logits, verify_lens
    )
    commit_lens, new_seq_lens, cap_trim_lens = finalize_reference(
        correct_len, cap_trim_lens, prefix_lens
    )
    out_tokens = out_tokens_reference(
        draft_tokens, correct_len, bonus, case.stride, case.stride - 1
    )
    block_pos_offsets = torch.arange(case.stride, dtype=torch.int64)
    commit_swa_loc, commit_positions = commit_layout_reference(
        req_pool_indices,
        prefix_lens,
        block_pos_offsets,
        commit_lens,
        case.stride,
        RING_STRIDE,
    )

    for layer in block.layers:
        attention = layer.self_attn
        q_size = attention.q_size
        kv_size = attention.kv_size
        weight = attention.qkv_proj.weight.detach().double().cpu()
        bias = attention.qkv_proj.bias.detach().double().cpu()
        kv_weight = weight[q_size : q_size + 2 * kv_size]
        kv_bias = bias[q_size : q_size + 2 * kv_size]
        kv = torch.nn.functional.linear(projected_strided, kv_weight, kv_bias)
        key, value = kv.split([kv_size, kv_size], dim=-1)
        key = key.reshape(-1, attention.num_kv_heads, attention.head_dim)
        value = value.reshape(-1, attention.num_kv_heads, attention.head_dim)
        key = rms_norm_reference(
            key.reshape(-1, attention.head_dim),
            attention.k_norm.weight.detach().double().cpu(),
            1e-5,
        ).reshape(-1, attention.num_kv_heads, attention.head_dim)
        key = rope_reference(
            key,
            commit_positions,
            attention.rotary_emb.cos_sin_cache.cpu(),
            True,
        )
        committed = (
            torch.arange(case.stride)[None, :]
            < commit_lens.to(torch.long)[:, None]
        ).reshape(-1)
        valid_loc = commit_swa_loc[committed].to(torch.int64)
        valid_key = key.reshape(-1, attention.num_kv_heads, attention.head_dim)[committed]
        valid_value = value.reshape(-1, attention.num_kv_heads, attention.head_dim)[committed]
        reference_pool_keys.append((attention.attn.layer_id, valid_loc, valid_key))
        reference_pool_values.append((attention.attn.layer_id, valid_loc, valid_value))

    cumulative_pool_keys = [
        torch.zeros_like(block.pool.keys[layer_id].detach().cpu(), dtype=torch.float64)
        for layer_id in range(case.num_layers)
    ]
    cumulative_pool_values = [
        torch.zeros_like(block.pool.values[layer_id].detach().cpu(), dtype=torch.float64)
        for layer_id in range(case.num_layers)
    ]
    for layer_id, loc, expected in reference_pool_keys:
        cumulative_pool_keys[layer_id].index_copy_(
            0, loc.to(torch.int64), expected
        )
    for layer_id, loc, expected in reference_pool_values:
        cumulative_pool_values[layer_id].index_copy_(
            0, loc.to(torch.int64), expected
        )

    return {
        "projected_compact_hidden": projected_compact,
        "strided_hidden": strided_hidden,
        "projected_strided_hidden": projected_strided,
        "strided_logits": strided_logits,
        "strided_candidates": strided_candidates,
        "correct_len": correct_len,
        "bonus": bonus,
        "cap_trim_lens": cap_trim_lens,
        "commit_lens": commit_lens,
        "new_seq_lens": new_seq_lens,
        "out_tokens": out_tokens,
        "commit_swa_loc": commit_swa_loc,
        "commit_positions": commit_positions,
        "pool_keys": cumulative_pool_keys,
        "pool_values": cumulative_pool_values,
    }


def tensor_difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual = actual.detach()
    expected = expected.detach()
    if actual.is_floating_point():
        difference = (actual.double() - expected.double()).abs()
        maximum = float(difference.max()) if difference.numel() else 0.0
        denominator = expected.double().abs().clamp_min(1e-12)
        relative = float((difference / denominator).max()) if difference.numel() else 0.0
        return {
            "match": torch.allclose(
                actual.double(), expected.double(), atol=2e-5, rtol=2e-4
            ),
            "max_abs_diff": maximum,
            "max_relative_diff": relative,
        }
    if not torch.equal(actual, expected):
        if actual.is_floating_point():
            difference = (actual.double() - expected.double()).abs()
            maximum = float(difference.max()) if difference.numel() else 0.0
            denominator = expected.double().abs().clamp_min(1e-12)
            relative = float((difference / denominator).max()) if difference.numel() else 0.0
            return {
                "match": False,
                "max_abs_diff": maximum,
                "max_relative_diff": relative,
            }
        return {"match": False, "max_abs_diff": None, "max_relative_diff": None}
    return {"match": True, "max_abs_diff": 0.0, "max_relative_diff": 0.0}


def compare_outputs(block: ReducedBlock, reference: dict[str, torch.Tensor]) -> dict[str, Any]:
    comparisons = {}
    for name in (
        "projected_compact_hidden",
        "strided_hidden",
        "projected_strided_hidden",
        "strided_logits",
        "strided_candidates",
        "correct_len",
        "bonus",
        "cap_trim_lens",
        "commit_lens",
        "new_seq_lens",
        "out_tokens",
        "commit_swa_loc",
        "commit_positions",
    ):
        comparisons[name] = tensor_difference(
            getattr(block, name).detach().cpu(), reference[name]
        )

    for output_name, reference_name in (
        ("keys", "pool_keys"),
        ("values", "pool_values"),
    ):
        actual_layers = getattr(block.pool, output_name)
        reference_entries = reference[reference_name]
        max_abs_diff = 0.0
        max_relative_diff = 0.0
        matched = True
        for layer_id, loc, expected in reference_entries:
            actual = actual_layers[layer_id].detach().cpu().double()
            expected_full = torch.zeros_like(actual)
            expected_full.index_copy_(0, loc.to(torch.int64), expected)
            result = tensor_difference(actual, expected_full)
            matched = matched and result["match"]
            max_abs_diff = max(max_abs_diff, result["max_abs_diff"] or 0.0)
            max_relative_diff = max(max_relative_diff, result["max_relative_diff"] or 0.0)
        comparisons[f"pool_{output_name}"] = {
            "match": matched,
            "max_abs_diff": max_abs_diff,
            "max_relative_diff": max_relative_diff,
        }
    return comparisons


def snapshot_outputs(block: ReducedBlock) -> dict[str, Any]:
    snapshot = {}
    for name in (
        "projected_compact_hidden",
        "strided_hidden",
        "projected_strided_hidden",
        "strided_logits",
        "strided_candidates",
        "correct_len",
        "bonus",
        "cap_trim_lens",
        "commit_lens",
        "new_seq_lens",
        "out_tokens",
        "commit_swa_loc",
        "commit_positions",
    ):
        snapshot[name] = getattr(block, name).detach().cpu().clone()
    snapshot["pool_keys"] = [
        tensor.detach().cpu().clone() for tensor in block.pool.keys
    ]
    snapshot["pool_values"] = [
        tensor.detach().cpu().clone() for tensor in block.pool.values
    ]
    return snapshot


def compare_snapshots(
    snapshot: dict[str, Any], reference: dict[str, torch.Tensor]
) -> dict[str, Any]:
    comparisons = {}
    for name in (
        "projected_compact_hidden",
        "strided_hidden",
        "projected_strided_hidden",
        "strided_logits",
        "strided_candidates",
        "correct_len",
        "bonus",
        "cap_trim_lens",
        "commit_lens",
        "new_seq_lens",
        "out_tokens",
        "commit_swa_loc",
        "commit_positions",
    ):
        comparisons[name] = tensor_difference(snapshot[name], reference[name])

    for output_name, reference_name in (
        ("pool_keys", "pool_keys"),
        ("pool_values", "pool_values"),
    ):
        actual_layers = snapshot[output_name]
        reference_layers = reference[reference_name]
        max_abs_diff = 0.0
        max_relative_diff = 0.0
        matched = True
        for actual, expected in zip(actual_layers, reference_layers):
            result = tensor_difference(actual[:-1], expected[:-1])
            matched = matched and result["match"]
            max_abs_diff = max(max_abs_diff, result["max_abs_diff"] or 0.0)
            max_relative_diff = max(max_relative_diff, result["max_relative_diff"] or 0.0)
        comparisons[f"pool_{output_name}"] = {
            "match": matched,
            "max_abs_diff": max_abs_diff,
            "max_relative_diff": max_relative_diff,
        }
    return comparisons


def compare_eager_to_graph(
    eager: dict[str, Any], graph: dict[str, Any]
) -> dict[str, Any]:
    comparisons = {}
    for name in eager:
        if name.startswith("pool_"):
            max_abs_diff = 0.0
            max_relative_diff = 0.0
            matched = True
            for actual, expected in zip(eager[name], graph[name]):
                result = tensor_difference(actual[:-1], expected[:-1])
                matched = matched and result["match"]
                max_abs_diff = max(max_abs_diff, result["max_abs_diff"] or 0.0)
                max_relative_diff = max(
                    max_relative_diff, result["max_relative_diff"] or 0.0
                )
            comparisons[name] = {
                "match": matched,
                "max_abs_diff": max_abs_diff,
                "max_relative_diff": max_relative_diff,
            }
        else:
            comparisons[name] = tensor_difference(eager[name], graph[name])
    return comparisons


def set_fresh_inputs(block: ReducedBlock, seed: int) -> None:
    torch.manual_seed(seed)
    block.compact_hidden.copy_(
        torch.randn(
            block.compact_hidden.shape,
            device=block.device,
            dtype=block.dtype,
        )
    )
    block.compact_logits.copy_(
        torch.randn(
            block.compact_logits.shape,
            device=block.device,
            dtype=block.dtype,
        )
    )
    block.compact_candidates.copy_(
        torch.randint(
            0,
            block.case.vocab_size,
            block.compact_candidates.shape,
            device=block.device,
            dtype=torch.int64,
        )
    )
    block.draft_tokens.copy_(
        torch.randint(
            0,
            block.case.vocab_size,
            block.draft_tokens.shape,
            device=block.device,
            dtype=torch.int64,
        )
    )
    block.verify_lens.copy_(
        torch.randint(
            1,
            block.case.stride + 1,
            block.verify_lens.shape,
            device=block.device,
            dtype=torch.int32,
        )
    )
    block.prefix_lens.copy_(
        torch.randint(
            1,
            1024,
            block.prefix_lens.shape,
            device=block.device,
            dtype=torch.int64,
        )
    )


def time_block(
    *,
    execute: Any,
    prepare: Any,
    warmup: int,
    measured: int,
) -> dict[str, float]:
    for iteration in range(warmup):
        prepare(iteration)
        execute()
    torch.cuda.synchronize()
    elapsed = []
    for iteration in range(measured):
        prepare(warmup + iteration)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        execute()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(elapsed),
        "mean_ms": statistics.fmean(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "iterations": measured,
    }


def time_execution(
    execute: Any,
    *,
    warmup: int,
    measured: int,
) -> dict[str, float]:
    for _ in range(warmup):
        execute()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(measured):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        execute()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(elapsed),
        "mean_ms": statistics.fmean(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "iterations": measured,
    }


def native_paths() -> dict[str, Any]:
    import sglang
    import sgl_kernel
    import triton

    return {
        "python": sys.executable,
        "torch": torch.__file__,
        "torch_version": torch.__version__,
        "torch_hip": torch.version.hip,
        "sglang": sglang.__file__,
        "sgl_kernel": sgl_kernel.__file__,
        "triton": triton.__file__,
        "cuda_graph": f"{torch.cuda.CUDAGraph.__module__}.{torch.cuda.CUDAGraph.__name__}",
        "rotary_embedding": f"{RotaryEmbedding.__module__}.{RotaryEmbedding.__name__}",
        "dflash_projection": f"{DFlashDraftModel.__module__}.{DFlashDraftModel.__name__}.project_target_hidden",
        "dspark_write": f"{DSparkDraftMixin.__module__}.{DSparkDraftMixin.__name__}.write_target_hidden_kv",
        "target_injector": f"{TargetHiddenKvInjector.__module__}.{TargetHiddenKvInjector.__name__}",
        "scatter": f"{scatter_compact_to_strided_into.__module__}.{scatter_compact_to_strided_into.__name__}",
        "accept": f"{accept_greedy_triton.__module__}.{accept_greedy_triton.__name__}",
        "finalize": f"{finalize_accept_lens_triton.__module__}.{finalize_accept_lens_triton.__name__}",
        "out_tokens": f"{BuildOutTokens.__module__}.{BuildOutTokens.__name__}",
        "commit_layout": f"{build_unified_commit_inject_layout.__module__}.{build_unified_commit_inject_layout.__name__}",
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This reduced pipeline requires one available HIP GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one assigned GPU, found {torch.cuda.device_count()}"
        )

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    results = {
        "label": "reduced DSpark handoff pipeline; synthetic weights and states only",
        "scope": {
            "gpu": properties.name,
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_memory_bytes": properties.total_memory,
            "device_count": torch.cuda.device_count(),
            "image": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "expected_local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
            "seed": SEED,
            "warmup_iterations": WARMUP_ITERATIONS,
            "measured_iterations": MEASURED_ITERATIONS,
            "workload_cases": len(CASES),
            "weights_bytes": 0,
            "distributed_claim": False,
            "checkpoint_download": False,
        },
        "native_paths": native_paths(),
        "cases": [],
    }

    total_weight_bytes = 0
    for case in CASES:
        started = time.perf_counter()
        block = ReducedBlock(case, device)

        validation_seed = 10_000 + len(results["cases"])
        set_fresh_inputs(block, validation_seed)
        block.pool.reset()
        with torch.inference_mode():
            block.run()
        eager_snapshot = snapshot_outputs(block)
        reference = reference_outputs(block)
        eager_comparisons = compare_snapshots(eager_snapshot, reference)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(WARMUP_ITERATIONS):
                set_fresh_inputs(block, validation_seed + 100 + _)
                block.pool.reset()
                block.run()
        torch.cuda.current_stream().wait_stream(stream)

        set_fresh_inputs(block, validation_seed)
        block.pool.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            block.run()

        block.pool.reset()
        graph.replay()
        torch.cuda.synchronize()
        graph_snapshot = snapshot_outputs(block)
        graph_comparisons = compare_snapshots(graph_snapshot, reference)
        eager_vs_graph = compare_eager_to_graph(eager_snapshot, graph_snapshot)

        def prepare_eager(iteration: int) -> None:
            set_fresh_inputs(block, 20_000 + iteration)
            block.pool.reset()

        eager_timing = time_block(
            execute=block.run,
            prepare=prepare_eager,
            warmup=WARMUP_ITERATIONS,
            measured=MEASURED_ITERATIONS,
        )

        def replay() -> None:
            graph.replay()

        def prepare_graph(iteration: int) -> None:
            set_fresh_inputs(block, 20_000 + iteration)
            block.pool.reset()

        graph_timing = time_block(
            execute=replay,
            prepare=prepare_graph,
            warmup=WARMUP_ITERATIONS,
            measured=MEASURED_ITERATIONS,
        )

        case_weight_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in block.fc.parameters()
        ) + sum(
            parameter.numel() * parameter.element_size()
            for layer in block.layers
            for parameter in layer.self_attn.qkv_proj.parameters()
        ) + sum(
            parameter.numel() * parameter.element_size()
            for parameter in block.hidden_norm.parameters()
        ) + sum(
            parameter.numel() * parameter.element_size()
            for layer in block.layers
            for parameter in layer.self_attn.k_norm.parameters()
        )
        total_weight_bytes += case_weight_bytes

        case_result = {
            "name": case.name,
            "dimensions": {
                "batch_size": case.batch_size,
                "verify_num_draft_tokens": case.stride,
                "gamma": case.stride - 1,
                "hidden_size": case.hidden_size,
                "num_layers": case.num_layers,
                "num_heads": case.num_heads,
                "num_kv_heads": case.num_kv_heads,
                "head_dim": case.head_dim,
                "vocab_size": case.vocab_size,
                "dtype": str(block.dtype).replace("torch.", ""),
            },
            "weights_bytes": case_weight_bytes,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "eager_reference_comparisons": eager_comparisons,
            "graph_reference_comparisons": graph_comparisons,
            "eager_vs_graph_comparisons": eager_vs_graph,
            "eager_timing": eager_timing,
            "graph_timing": graph_timing,
            "graph_replay_speedup_median": (
                eager_timing["median_ms"] / graph_timing["median_ms"]
                if graph_timing["median_ms"] > 0
                else None
            ),
        }
        results["cases"].append(case_result)

    failed_comparisons = {
        case_result["name"]: [
            comparison_name
            for comparison_group in (
                case_result["eager_reference_comparisons"],
                case_result["graph_reference_comparisons"],
                case_result["eager_vs_graph_comparisons"],
            )
            for comparison_name, comparison in comparison_group.items()
            if not comparison["match"]
        ]
        for case_result in results["cases"]
    }
    failed_comparisons = {
        case_name: comparison_names
        for case_name, comparison_names in failed_comparisons.items()
        if comparison_names
    }
    if failed_comparisons:
        raise RuntimeError(f"Validation failed: {failed_comparisons}")

    results["scope"]["weights_bytes"] = total_weight_bytes
    output_path = os.path.join(
        os.path.dirname(__file__), "results.json"
    )
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
