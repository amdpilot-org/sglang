import inspect
import types
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention import flashinfer_backend as backend


def check_graph_replay_uses_ordinary_decode():
    source = inspect.getsource(backend.FlashInferAttnBackend.init_forward_metadata_out_graph)
    assert "self.indices_updater_decode.update(" in source
    assert "indices_updater_cascade_decode" not in source
    assert "cascade_wrappers" not in source


def check_mode_exclusions_are_real():
    source = inspect.getsource(backend.FlashInferAttnBackend.init_forward_metadata)
    assert "forward_batch.spec_info is None" in source
    init_source = inspect.getsource(backend.FlashInferAttnBackend.__init__)
    assert "self.dispatch_reason is None and not self.decode_uses_dequant_workspace" in init_source
    assert "WrapperDispatch.SLIDING_WINDOW" in init_source
    assert "WrapperDispatch.CROSS_ATTENTION" in init_source


def check_prefix_adversaries():
    rows = torch.tensor([[1, 2, 3], [1, 9, 3], [1, 2, 3]], dtype=torch.int32)
    assert backend.common_prefix_length(rows) == 1
    assert backend.common_prefix_length(torch.empty((4, 0), dtype=torch.int32)) == 0
    assert backend.should_use_cascade_attention(4, 512, 512)
    assert not backend.should_use_cascade_attention(4, 512, 1025)


def check_identical_requests_produce_empty_unique_table():
    token_table = torch.arange(600, dtype=torch.int32).repeat(4, 1)

    class Translator:
        def fill_packed_read_stream(self, *, req_pool_indices, seq_lens, out, kv_start_idx, **_):
            offset = 0
            for i, (row, length) in enumerate(zip(req_pool_indices.tolist(), seq_lens.tolist())):
                start = 0 if kv_start_idx is None else int(kv_start_idx[i])
                out[offset : offset + length] = token_table[row, start : start + length]
                offset += length

    updater = backend.FlashInferIndicesUpdaterCascadeDecode.__new__(backend.FlashInferIndicesUpdaterCascadeDecode)
    updater.num_qo_heads = 8
    updater.num_kv_heads = 2
    updater.head_dim = 64
    updater.data_type = torch.float16
    updater.q_data_type = torch.float16
    updater.attn_backend = types.SimpleNamespace(kv_index_translator=Translator())
    wrappers = [Mock(), Mock()]
    seq_lens = torch.full((4,), 600, dtype=torch.int32)
    updater.update(torch.arange(4, dtype=torch.int32), seq_lens, seq_lens, 600, wrappers)
    unique_args = wrappers[1].begin_forward.call_args.args
    assert unique_args[1].tolist() == [0, 0, 0, 0, 0]
    assert unique_args[2].numel() == 0
    print("identical-prefix layout: candidate invokes unique FlashInfer wrapper with four zero-length KV segments")


if __name__ == "__main__":
    check_graph_replay_uses_ordinary_decode()
    check_mode_exclusions_are_real()
    check_prefix_adversaries()
    check_identical_requests_produce_empty_unique_table()
    print("all independent adversarial checks passed")
