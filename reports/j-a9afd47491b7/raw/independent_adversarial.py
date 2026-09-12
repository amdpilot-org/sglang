import inspect
import types
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention import flashinfer_backend as backend


def check_prefix_and_policy_edges():
    rows = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 9, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
        ],
        device="cuda",
    )
    assert backend.common_prefix_length(rows) == 2
    assert not backend.should_use_cascade_attention(4, 511, 512)
    assert backend.should_use_cascade_attention(4, 512, 640)
    assert not backend.should_use_cascade_attention(15, 600, 1000)
    assert backend.should_use_cascade_attention(16, 600, 1000)


def check_zero_suffix_plan_shape():
    table = torch.tensor([[10, 11, 12]] * 4, dtype=torch.int32, device="cuda")

    class Translator:
        def fill_packed_read_stream(self, **kwargs):
            out = kwargs["out"]
            offset = 0
            starts = kwargs["kv_start_idx"]
            for i, length in enumerate(kwargs["seq_lens"].tolist()):
                start = 0 if starts is None else int(starts[i])
                out[offset : offset + length] = table[i, start : start + length]
                offset += length
            return False

    updater = backend.FlashInferIndicesUpdaterCascadeDecode.__new__(
        backend.FlashInferIndicesUpdaterCascadeDecode
    )
    updater.num_qo_heads = 4
    updater.num_kv_heads = 4
    updater.head_dim = 64
    updater.data_type = torch.float16
    updater.q_data_type = torch.float16
    updater.attn_backend = types.SimpleNamespace(kv_index_translator=Translator())
    wrappers = [Mock(), Mock()]
    updater.update(
        torch.arange(4, dtype=torch.int32, device="cuda"),
        torch.full((4,), 3, dtype=torch.int32, device="cuda"),
        [3, 3, 3, 3],
        3,
        wrappers,
    )
    unique_args = wrappers[1].begin_forward.call_args.args
    assert unique_args[2].numel() == 0
    assert unique_args[1].tolist() == [0, 0, 0, 0, 0]


def check_explicit_scope_exclusions():
    init_source = inspect.getsource(backend.FlashInferAttnBackend.__init__)
    metadata_source = inspect.getsource(backend.FlashInferAttnBackend.init_forward_metadata)
    assert "self.dispatch_reason is None" in init_source
    assert "not self.decode_uses_dequant_workspace" in init_source
    assert "forward_batch.spec_info is None" in metadata_source
    assert "self.indices_updater_decode.update" in metadata_source


if __name__ == "__main__":
    check_prefix_and_policy_edges()
    check_zero_suffix_plan_shape()
    check_explicit_scope_exclusions()
    print("independent adversarial checks passed on", torch.cuda.get_device_name(0))
