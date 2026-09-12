import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler


def make_handler():
    handler = object.__new__(DecodeStagingHandler)
    handler._wm_subscribers = {}
    handler._wm_subscribers_lock = threading.Lock()
    handler._wm_next_generation = 0
    handler.staging_allocator = SimpleNamespace(
        get_watermark=lambda: (0, 0), free=lambda alloc_id: None
    )
    return handler


def make_manager(connection_pool, handler):
    manager = object.__new__(CommonKVManager)
    manager.connection_lock = threading.Lock()
    manager.connection_pool = connection_pool
    manager.prefill_info_table = {"10.0.0.1:8998": object()}
    manager.addr_to_rooms_tracker = {"10.0.0.1:8998": []}
    manager.request_status = {}
    manager._staging_handler = handler
    return manager


def main():
    # CommonKVReceiver._setup_bootstrap_infos concatenates one cached group per
    # target CP rank. The subscriber is therefore keyed by the concatenation,
    # but _handle_node_failure snapshots each connection_pool group separately.
    cp0 = [{"rank_ip": "10.0.0.1", "rank_port": 12000, "cp": 0}]
    cp1 = [{"rank_ip": "10.0.0.1", "rank_port": 12001, "cp": 1}]
    handler = make_handler()
    receiver = SimpleNamespace(bootstrap_infos=cp0 + cp1)
    handler.register_wm_subscriber(receiver, "multi-cp-session")
    manager = make_manager(
        {
            "10.0.0.1:8998_0_0_0": cp0,
            "10.0.0.1:8998_0_1_0": cp1,
        },
        handler,
    )
    with patch.object(CommonKVManager, "check_status", return_value=KVPoll.Failed), patch(
        "sglang.srt.disaggregation.common.conn.CommonKVReceiver.disconnect_endpoint"
    ):
        manager._handle_node_failure("10.0.0.1:8998")
    print("multi_cp_remaining", len(handler._wm_subscribers))
    assert len(handler._wm_subscribers) == 1, handler._wm_subscribers

    # Broadcast still targets the stale receiver after failure cleanup.
    handler._send_watermark = MagicMock()
    handler._free_and_send_watermark(7, SimpleNamespace())
    print("stale_broadcast_calls", handler._send_watermark.call_count)
    assert handler._send_watermark.call_count == 1

    # A single cached group remains fixed, showing the counterexample is
    # specifically the aggregation mismatch rather than a broken harness.
    single_handler = make_handler()
    single_receiver = SimpleNamespace(bootstrap_infos=cp0)
    single_handler.register_wm_subscriber(single_receiver, "single-cp-session")
    single_manager = make_manager({"10.0.0.1:8998_0_0_0": cp0}, single_handler)
    with patch.object(CommonKVManager, "check_status", return_value=KVPoll.Failed), patch(
        "sglang.srt.disaggregation.common.conn.CommonKVReceiver.disconnect_endpoint"
    ):
        single_manager._handle_node_failure("10.0.0.1:8998")
    print("single_cp_remaining", len(single_handler._wm_subscribers))
    assert len(single_handler._wm_subscribers) == 0


if __name__ == "__main__":
    main()
