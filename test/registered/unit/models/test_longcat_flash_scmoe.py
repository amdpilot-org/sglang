from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.models import longcat_flash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _Backend:
    def __init__(self, is_none):
        self._is_none = is_none

    def is_none(self):
        return self._is_none


class _Attention:
    def __init__(self, expected_tokens):
        self.expected_tokens = expected_tokens

    def __call__(self, *, positions, hidden_states, **kwargs):
        if hidden_states.shape[0] != positions.shape[0]:
            raise RuntimeError(
                f"RoPE token mismatch: hidden={hidden_states.shape[0]}, "
                f"positions={positions.shape[0]}"
            )
        assert hidden_states.shape[0] == self.expected_tokens
        return hidden_states


class _Communicator:
    def prepare_attn(self, hidden_states, residual, forward_batch):
        return hidden_states, residual

    def prepare_mlp(self, hidden_states, residual, forward_batch):
        return hidden_states, residual

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        return hidden_states, residual


def _fake_layer(attn_tp_size, expected_tokens):
    return SimpleNamespace(
        attn_tp_size=attn_tp_size,
        mlps=[lambda x: x, lambda x: x],
        self_attn=[None, _Attention(expected_tokens)],
        mlp_layer_communicator=[None, _Communicator()],
    )


def _repeat_rows(tensor, target):
    if tensor is None or tensor.shape[0] == target:
        return tensor
    assert target % tensor.shape[0] == 0
    return tensor.repeat(target // tensor.shape[0], 1)


def _run_forward_mlp(*, backend_is_none, attn_tp_size, hidden_tokens, positions):
    layer = _fake_layer(attn_tp_size, positions)
    hidden = torch.arange(hidden_tokens * 3, dtype=torch.float32).view(hidden_tokens, 3)
    residual = hidden + 100
    with (
        patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=_Backend(backend_is_none),
        ),
        patch.object(longcat_flash, "_scmoe_align_rows", side_effect=_repeat_rows),
        patch.object(
            longcat_flash, "tensor_model_parallel_all_reduce", side_effect=lambda x: x
        ),
    ):
        return longcat_flash.LongcatFlashDecoderLayer.forward_mlp(
            layer,
            hidden,
            torch.arange(positions),
            residual,
            SimpleNamespace(),
            None,
            None,
        )


def test_scmoe_second_attention_gathers_scattered_ep_input():
    hidden, residual, _ = _run_forward_mlp(
        backend_is_none=False, attn_tp_size=2, hidden_tokens=2, positions=4
    )

    assert hidden.shape == (4, 3)
    assert residual.shape == (4, 3)


def test_scmoe_legacy_path_reproduces_rope_token_mismatch():
    layer = _fake_layer(attn_tp_size=2, expected_tokens=4)
    hidden = torch.zeros(2, 3)
    with (
        patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=_Backend(False),
        ),
        patch.object(longcat_flash, "_scmoe_align_rows", side_effect=lambda t, _: t),
        patch.object(
            longcat_flash, "tensor_model_parallel_all_reduce", side_effect=lambda x: x
        ),
        pytest.raises(RuntimeError, match="hidden=2, positions=4"),
    ):
        longcat_flash.LongcatFlashDecoderLayer.forward_mlp(
            layer,
            hidden,
            torch.arange(4),
            hidden.clone(),
            SimpleNamespace(),
            None,
            None,
        )


@pytest.mark.parametrize(("backend_is_none", "attn_tp_size"), [(True, 2), (False, 1)])
def test_scmoe_does_not_gather_without_real_ep_and_attn_tp(
    backend_is_none, attn_tp_size
):
    hidden, residual, _ = _run_forward_mlp(
        backend_is_none=backend_is_none,
        attn_tp_size=attn_tp_size,
        hidden_tokens=4,
        positions=4,
    )

    assert hidden.shape == (4, 3)
    assert residual.shape == (4, 3)


def test_scmoe_align_rows_gathers_and_slices_the_local_rank():
    local = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def fake_all_gather(output, tensor):
        output.copy_(torch.cat((tensor, tensor + 10)))

    with (
        patch(
            "sglang.srt.layers.dp_attention.attn_tp_all_gather_into_tensor",
            side_effect=fake_all_gather,
        ),
        patch.object(
            longcat_flash,
            "_gp",
            return_value=SimpleNamespace(attn_tp_rank=1),
        ),
    ):
        gathered = longcat_flash._scmoe_align_rows(local, 4)
        sliced = longcat_flash._scmoe_align_rows(gathered, 2)

    torch.testing.assert_close(gathered, torch.cat((local, local + 10)))
    torch.testing.assert_close(sliced, local + 10)
    assert longcat_flash._scmoe_align_rows(local, 2) is local
    assert longcat_flash._scmoe_align_rows(None, 2) is None
