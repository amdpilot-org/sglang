import os
import runpy
import tempfile
import types
from unittest.mock import patch

fixture = runpy.run_path("/job/review-evidence-j-5a69937f0025/independent_contract.py")
helper = fixture["_direct_linker_storage_config"]
make_store = fixture["make_store"]
config = fixture["config"]


def direct_config(base, *, attn_dp_rank, attn_cp_rank):
    params = types.SimpleNamespace(
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=2 if attn_cp_rank else 1,
    )
    with (
        patch(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.get_parallel"
        ) as get_parallel,
        patch(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.get_model"
        ) as get_model,
    ):
        get_parallel.return_value.attn_dp_rank = attn_dp_rank
        get_model.return_value.model_path = "adversarial-review"
        return helper(
            params=params,
            tp_rank=0,
            tp_size=1,
            rank_replicated=True,
            extra_config=config(base).extra_config,
        )


failures = []
with tempfile.TemporaryDirectory() as base:
    # Pure DP has independent caches but attn_dp_rank == 0 for every replica.
    pure_dp_paths = [
        make_store(direct_config(base, attn_dp_rank=0, attn_cp_rank=0)).setup_kwargs[
            "ssd_offload_path"
        ]
        for _dp_rank in (0, 1)
    ]
    print("pure_dp_replica_paths=", pure_dp_paths)
    if len(set(pure_dp_paths)) != 2:
        failures.append("pure-DP replicas collide because only attn_dp_rank is used")

with tempfile.TemporaryDirectory() as base:
    cp_paths = [
        make_store(direct_config(base, attn_dp_rank=0, attn_cp_rank=cp)).setup_kwargs[
            "ssd_offload_path"
        ]
        for cp in (0, 1)
    ]
    print("attention_cp_paths=", cp_paths)
    if len(set(cp_paths)) != 2:
        failures.append("attention-CP workers collide because attn_cp_rank is omitted")

assert not failures, "; ".join(failures)
