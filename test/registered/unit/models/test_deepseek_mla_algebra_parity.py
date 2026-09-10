"""Numerical parity between DeepSeek MLA normal and absorbed representations.

The test uses the existing tiny MLA fixture, analytically equivalent small
projection weights, and the same latent KV state. It compares the selected
Triton forward, a local absorbed representation, a local normal representation,
and an independent Torch SDPA reference under causal and sequence-tail controls.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLA_ATOL,
    MLA_RTOL,
    MLAAttentionCase,
    build_mla_attention_fixture,
    run_mla_fixture_eager,
)
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")


def _manual_attention(query, key, value, scale, query_positions):
    scores = torch.einsum("thd,shd->ths", query.float(), key.float()) * scale
    key_positions = torch.arange(key.shape[0], device=query.device)
    allowed = key_positions[None, None, :] <= query_positions[:, None, None]
    scores = scores.masked_fill(~allowed, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("ths,shv->thv", probabilities, value.float()).to(query.dtype)


def _independent_sdpa(query, key, value, scale, query_positions):
    key_positions = torch.arange(key.shape[0], device=query.device)
    mask = (
        key_positions[None, None, None, :]
        <= query_positions[None, None, :, None]
    )
    output = F.scaled_dot_product_attention(
        query.permute(1, 0, 2).float().unsqueeze(0),
        key.permute(1, 0, 2).float().unsqueeze(0),
        value.permute(1, 0, 2).float().unsqueeze(0),
        attn_mask=mask,
        scale=scale,
    )
    return output.squeeze(0).permute(1, 0, 2).to(query.dtype)


def _representations(fixture):
    module = fixture.actual_module
    case = fixture.case
    scale = module.attn_mqa.scaling
    outputs = {}

    with torch.no_grad():
        for representation in ("absorbed", "normal"):
            dense_rows = []
            independent_rows = []
            for request_index, (prefix, current) in enumerate(
                zip(fixture.prefix_hidden, fixture.input_hidden.split(case.input_lens))
            ):
                hidden = torch.cat([prefix, current], dim=0)
                query_start = case.prefix_lens[request_index]
                query_positions = torch.arange(
                    query_start,
                    query_start + current.shape[0],
                    device=hidden.device,
                )
                q_nope = module.q_proj(hidden).view(
                    -1, module.num_heads, module.kv_lora_rank
                )
                latent = module._rms_norm(module.kv_a_proj(hidden)).unsqueeze(1)

                if representation == "absorbed":
                    query = torch.einsum(
                        "thd,hdr->thr", q_nope.float(), module.w_kc.float()
                    )[query_start:].to(hidden.dtype)
                    attention = _manual_attention(
                        query, latent, latent, scale, query_positions
                    )
                    dense = torch.einsum(
                        "thr,hrv->thv", attention.float(), module.w_vc.float()
                    ).to(hidden.dtype)
                else:
                    query = q_nope[query_start:]
                    key = torch.einsum(
                        "tr,hdr->thd", latent.squeeze(1).float(), module.w_kc.float()
                    ).to(hidden.dtype)
                    value = torch.einsum(
                        "tr,hrv->thv", latent.squeeze(1).float(), module.w_vc.float()
                    ).to(hidden.dtype)
                    attention = _manual_attention(
                        query, key, value, scale, query_positions
                    )
                    dense = attention
                    independent_rows.append(
                        _independent_sdpa(
                            query, key, value, scale, query_positions
                        )
                    )

                dense_rows.append(dense)

            outputs[representation] = module.o_proj(
                torch.cat(dense_rows, dim=0).flatten(1, 2)
            )
            if representation == "normal":
                outputs["independent"] = module.o_proj(
                    torch.cat(independent_rows, dim=0).flatten(1, 2)
                )

    return outputs


def _record_selected_paths(fixture):
    selected_paths = []
    original_attention_forward = fixture.actual_module.attn_mqa.forward
    original_extend = fixture.backend.forward_extend
    original_decode = fixture.backend.forward_decode

    def attention_forward(*args, **kwargs):
        selected_paths.append("RadixAttention.forward")
        return original_attention_forward(*args, **kwargs)

    def forward_extend(*args, **kwargs):
        selected_paths.append("TritonAttnBackend.forward_extend")
        return original_extend(*args, **kwargs)

    def forward_decode(*args, **kwargs):
        selected_paths.append("TritonAttnBackend.forward_decode")
        return original_decode(*args, **kwargs)

    fixture.actual_module.attn_mqa.forward = attention_forward
    fixture.backend.forward_extend = forward_extend
    fixture.backend.forward_decode = forward_decode
    return selected_paths


class TestDeepseekMLAAlgebraParity(CustomTestCase):
    CASES = {
        "causal_cross_page_extend": MLAAttentionCase(
            name="algebra_parity_causal_cross_page_extend",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(2,),
        ),
        "decode_sequence_tail": MLAAttentionCase(
            name="algebra_parity_decode_sequence_tail",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    }

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_normal_and_absorbed_representations_match_independent_torch(self):
        for name, case in self.CASES.items():
            with self.subTest(case=name):
                fixture = build_mla_attention_fixture(self, case)
                selected_paths = _record_selected_paths(fixture)
                actual = run_mla_fixture_eager(fixture)
                local = _representations(fixture)

                expected_paths = (
                    ["RadixAttention.forward", "TritonAttnBackend.forward_extend"]
                    if case.forward_mode.is_extend()
                    else ["RadixAttention.forward", "TritonAttnBackend.forward_decode"]
                )
                self.assertEqual(selected_paths, expected_paths)

                torch.testing.assert_close(
                    actual,
                    local["independent"],
                    atol=MLA_ATOL,
                    rtol=MLA_RTOL,
                )
                torch.testing.assert_close(
                    local["absorbed"],
                    local["independent"],
                    atol=MLA_ATOL,
                    rtol=MLA_RTOL,
                )
                torch.testing.assert_close(
                    local["normal"],
                    local["independent"],
                    atol=MLA_ATOL,
                    rtol=MLA_RTOL,
                )
                torch.testing.assert_close(
                    actual,
                    local["absorbed"],
                    atol=MLA_ATOL,
                    rtol=MLA_RTOL,
                )
                torch.testing.assert_close(
                    actual,
                    local["normal"],
                    atol=MLA_ATOL,
                    rtol=MLA_RTOL,
                )
