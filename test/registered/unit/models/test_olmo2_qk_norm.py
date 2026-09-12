from unittest.mock import Mock, patch

import torch

from sglang.srt.models.olmo2 import Olmo2Attention


def test_qk_norm_uses_dispatched_norm_outside_capture():
    attention = Mock(spec=Olmo2Attention)
    attention.tp_size = 1
    attention.alt_stream = None
    attention.q_norm = Mock(side_effect=lambda x: x)
    attention.k_norm = Mock(side_effect=lambda x: x)

    q = torch.randn(3, 16)
    k = torch.randn(3, 8)
    output_q, output_k = Olmo2Attention._apply_qk_norm(attention, q, k)

    attention.q_norm.assert_called_once()
    attention.k_norm.assert_called_once()
    assert attention.q_norm.call_args.args[0].data_ptr() == q.data_ptr()
    assert attention.k_norm.call_args.args[0].data_ptr() == k.data_ptr()
    attention.q_norm.forward_native.assert_not_called()
    attention.k_norm.forward_native.assert_not_called()
    torch.testing.assert_close(output_q, q)
    torch.testing.assert_close(output_k, k)


def test_qk_norm_uses_dual_stream_only_during_capture():
    attention = Mock(spec=Olmo2Attention)
    attention.tp_size = 1
    attention.alt_stream = Mock()
    attention.q_norm = Mock(side_effect=lambda x: x)
    attention.k_norm = Mock(side_effect=lambda x: x)
    q = torch.randn(3, 16)
    k = torch.randn(3, 8)

    current_stream = Mock()
    stream_context = Mock()
    stream_context.__enter__ = Mock()
    stream_context.__exit__ = Mock(return_value=False)
    with (
        patch("sglang.srt.models.olmo2.get_is_capture_mode", return_value=True),
        patch(
            "sglang.srt.models.olmo2.torch.cuda.current_stream",
            return_value=current_stream,
        ),
        patch("sglang.srt.models.olmo2.torch.cuda.stream", return_value=stream_context),
    ):
        output_q, output_k = Olmo2Attention._apply_qk_norm(attention, q, k)

    attention.alt_stream.wait_stream.assert_called_once_with(current_stream)
    current_stream.wait_stream.assert_called_once_with(attention.alt_stream)
    attention.q_norm.assert_called_once()
    attention.k_norm.assert_called_once()
    assert attention.q_norm.call_args.args[0].data_ptr() == q.data_ptr()
    assert attention.k_norm.call_args.args[0].data_ptr() == k.data_ptr()
    torch.testing.assert_close(output_q, q)
    torch.testing.assert_close(output_k, k)
