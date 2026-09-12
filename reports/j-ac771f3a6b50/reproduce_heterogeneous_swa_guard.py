"""Reproduce issue #38552 against the real NIXL dispatch implementation.

This intentionally avoids constructing a NIXL agent: the reported failure occurs
in ``maybe_send_extra`` before any transport call.  The accepted boundary cases
replace only the eventual transport operation with a recorder.
"""

from types import SimpleNamespace

from sglang.srt.disaggregation.nixl.conn import NixlKVManager, StateType


def make_manager(*, prefill_tp: int, is_mla: bool):
    manager = NixlKVManager.__new__(NixlKVManager)
    manager.attn_tp_size = prefill_tp
    manager.is_mla_backend = is_mla
    manager.pp_size = 1
    manager.kv_args = SimpleNamespace(
        state_types=[StateType.SWA],
        state_data_ptrs=[[0x1000, 0x2000]],
        state_item_lens=[[64, 64]],
        state_dim_per_tensor=[[]],
        state_conv_shard_groups=[[]],
        state_slice_outer_counts=[[]],
        state_layer_ids=[[]],
        gpu_id=0,
    )
    calls = []

    def record_transfer(**kwargs):
        calls.append(kwargs)
        return "transfer-handle"

    manager._send_kvcache_generic = record_transfer
    return manager, calls


def dispatch(manager, *, decode_tp: int):
    return manager.maybe_send_extra(
        peer_name="decode-rank",
        prefill_state_indices=[[3]],
        dst_state_data_ptrs=[[0x3000, 0x4000]],
        dst_state_indices=[[7]],
        dst_gpu_id=0,
        notif="room_state",
        decode_tp_size=decode_tp,
        decode_tp_rank=0,
        dst_state_item_lens=[[64, 64]],
        dst_state_dim_per_tensor=[[]],
        dst_state_layer_ids=[[]],
    )


def main():
    manager, calls = make_manager(prefill_tp=2, is_mla=False)
    try:
        dispatch(manager, decode_tp=4)
    except RuntimeError as exc:
        print(f"REPRODUCED hetero_non_mla: {exc}")
        assert "different TP sizes for non-MLA SWA" in str(exc)
        assert calls == []
    else:
        raise AssertionError("heterogeneous non-MLA SWA unexpectedly dispatched")

    manager, calls = make_manager(prefill_tp=2, is_mla=False)
    handles = dispatch(manager, decode_tp=2)
    assert handles == ["transfer-handle"] and len(calls) == 1
    print("BOUNDARY accepted_equal_tp_non_mla: one whole-item transfer")

    manager, calls = make_manager(prefill_tp=2, is_mla=True)
    handles = dispatch(manager, decode_tp=4)
    assert handles == ["transfer-handle"] and len(calls) == 1
    print("BOUNDARY accepted_hetero_tp_mla: one whole-item transfer")


if __name__ == "__main__":
    main()
