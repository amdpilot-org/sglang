from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)


def make_req(l1, l2, l3, tree):
    req = SimpleNamespace(
        rid="adversarial",
        origin_input_ids=list(range(l1 + l2 + l3)),
        last_node=tree.root_node,
    )
    return SimpleNamespace(
        req=req,
        prefix_match=DecodePrefixMatch(
            prefix_indices=torch.arange(l1, dtype=torch.int64),
            l2_host_hit_length=l2,
            l3_storage_hit_length=l3,
            last_device_node=tree.root_node,
        ),
        hicache_restore_status=HiCacheRestoreResult.PENDING,
    )


tree = MagicMock()
tree.check_prefetch_progress.return_value = True
tree.inc_lock_ref.return_value.to_dec_params.return_value = "receipt"
harness = SimpleNamespace(tree_cache=tree)

# One token short must fail rather than exposing transport success.
short = make_req(0, 0, 64, tree)
tree.init_load_back.return_value = (torch.arange(63), object())
with patch(
    "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
    return_value=SimpleNamespace(
        device_indices=torch.empty(0, dtype=torch.int64),
        best_match_node=tree.root_node,
        host_hit_length=63,
    ),
):
    assert not DecodeHiCacheTransferMixin._try_hicache_queue_load_back(harness, short)
assert short.hicache_restore_status is HiCacheRestoreResult.FAILED

# Mixed L1/L2/L3 coverage must retain exactly the non-L1 restored suffix.
mixed = make_req(8, 16, 40, tree)
restored = torch.arange(100, 156, dtype=torch.int64)
tree.init_load_back.return_value = (restored, object())
with patch(
    "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
    return_value=SimpleNamespace(
        device_indices=torch.arange(8, dtype=torch.int64),
        best_match_node=tree.root_node,
        host_hit_length=56,
    ),
):
    assert DecodeHiCacheTransferMixin._try_hicache_queue_load_back(harness, mixed)
torch.testing.assert_close(mixed.hicache_restored_kv_indices, restored)
assert len(mixed.hicache_restored_kv_indices) == 56

# Gating must preserve transport failures while hiding only premature success.
receiver = MagicMock()
failed_req = SimpleNamespace(
    kv_receiver=receiver, hicache_restore_status=HiCacheRestoreResult.PENDING
)
receiver.poll.return_value = KVPoll.Failed
assert HiCacheRestoreGatedKVReceiver(failed_req).poll() == KVPoll.Failed

print("adversarial HiCache contract checks passed")
