"""Tests for the NCCL one-sided RMA support.

Tier 1 (no GPU) guards the ctypes binding layer. Tier 2 (2 GPUs + NCCL)
asserts the warm-up contract on a real communicator: init must not raise
when NCCL returns a NULL window handle (no NVLink / vGPU), and on NVLink it
must prove the put/wait round-trip.

Run::

    pytest test/registered/e2e/distributed/test_pynccl_rma.py -q
    python test/registered/e2e/distributed/test_pynccl_rma.py --num-gpu 2
"""

from __future__ import annotations

import ctypes

import pytest
import torch

from sglang.srt.distributed.device_communicators import pynccl as pynccl_mod
from sglang.srt.distributed.device_communicators import pynccl_wrapper as W
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
    NCCL_API_MAGIC,
    NCCL_CONFIG_UNDEF_INT,
    NCCL_WIN_COLL_SYMMETRIC,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")


def test_rma_function_table_is_complete():
    """Guards against dropping a bound RMA symbol (would silently break the
    warm-up path that needs it)."""
    names = {f.name for f in W.NCCLLibrary.exported_functions_rma}
    assert names == {
        "ncclMemAlloc",
        "ncclMemFree",
        "ncclPutSignal",
        "ncclSignal",
        "ncclWaitSignal",
        "ncclWinGetUserPtr",
        "ncclGetPeerDevicePointer",
        "ncclCommInitRankConfig",
    }


def test_rma_functions_have_argtypes():
    """Empty argtypes makes ctypes reject positional calls."""
    for f in W.NCCLLibrary.exported_functions_rma:
        assert len(f.argtypes) > 0, f"{f.name} has empty argtypes"


def test_optional_function_discovery_does_not_mutate_base_table(monkeypatch):
    class _Fn:
        def __call__(self, *args):
            return 0

    class _Lib:
        pass

    base_names = [f.name for f in W.NCCLLibrary.exported_functions]

    def make_lib(rma_names):
        lib = _Lib()
        for name in base_names:
            setattr(lib, name, _Fn())
        for name in rma_names:
            setattr(lib, name, _Fn())
        return lib

    all_rma = {f.name for f in W.NCCLLibrary.exported_functions_rma}
    libs = {"full": make_lib(all_rma), "partial": make_lib({"ncclPutSignal"})}
    monkeypatch.setattr(W.ctypes, "CDLL", lambda path: libs[path])
    W.NCCLLibrary.path_to_library_cache.clear()
    W.NCCLLibrary.path_to_dict_mapping.clear()

    assert W.NCCLLibrary("full").has_rma is True
    assert W.NCCLLibrary("partial").has_rma is False
    assert [f.name for f in W.NCCLLibrary.exported_functions] == base_names
    W.NCCLLibrary.path_to_library_cache.clear()
    W.NCCLLibrary.path_to_dict_mapping.clear()


def test_window_flag_matches_nccl_source():
    assert NCCL_WIN_COLL_SYMMETRIC == 0x01


def test_config_initializer_constants_match_nccl_source():
    assert NCCL_API_MAGIC == 0xCAFEBEEF
    assert NCCL_CONFIG_UNDEF_INT == -2147483648  # INT_MIN


def test_rma_min_version_matches_nccl_2_30():
    """ncclPutSignal shipped in 2.30 (raw 23007); the gate must not drift."""
    assert pynccl_mod._NCCL_RMA_MIN_VERSION == 23000


def test_nccl_config_t_field_count_is_21_with_trailing_graph_stream_ordering():
    """nccl4py lists 20 fields but NCCL_CONFIG_INITIALIZER at v2.30.7-1 has 21
    (trailing graphStreamOrdering); a 20-field binding makes init reject the
    config."""
    fields = [n for n, _ in W.ncclConfig_t._fields_]
    assert len(fields) == 21
    assert fields[-1] == "graphStreamOrdering"


def test_nccl_config_create_preserves_legacy_initializer_contract():
    """The RMA layout extension must not shadow the initializer used by the
    established configured-init and symmetric-memory paths."""
    cfg = W.ncclConfig_t.create()
    assert cfg.size == ctypes.sizeof(W.ncclConfig_t)
    assert cfg.magic == NCCL_API_MAGIC
    assert cfg.version == W.NCCL_CONFIG_VERSION
    for name, ctype in W.ncclConfig_t._fields_[3:]:
        expected = None if ctype is ctypes.c_char_p else NCCL_CONFIG_UNDEF_INT
        assert getattr(cfg, name) == expected, f"{name} not initialized"


def test_nccl_comm_init_rank_config_default_uses_initializer():
    """Exercise NCCLLibrary's pre-existing default-config call site."""
    seen = {}

    def _init(comm, world_size, unique_id, rank, config):
        seen["config"] = ctypes.cast(config, ctypes.POINTER(W.ncclConfig_t)).contents
        return 0

    lib = object.__new__(W.NCCLLibrary)
    lib._funcs = {"ncclCommInitRankConfig": _init}
    lib.ncclCommInitRankConfig(1, W.ncclUniqueId(), 0)
    assert seen["config"].magic == NCCL_API_MAGIC
    assert seen["config"].graphStreamOrdering == NCCL_CONFIG_UNDEF_INT


def test_symmetric_memory_config_call_site_uses_initializer(monkeypatch):
    """Exercise communicator construction's symmetric-memory call site."""
    seen = {}

    class _Nccl:
        def __init__(self, path):
            pass

        def ncclGetRawVersion(self):
            return 23007

        def ncclGetVersion(self):
            return "2.30.7"

        def ncclGetUniqueId(self):
            return W.ncclUniqueId()

        def ncclCommInitRankConfig(self, world_size, unique_id, rank, config):
            seen["config"] = config
            return W.ncclComm_t()

    class _Group:
        rank = 0
        world_size = 2

        def broadcast_obj(self, value, src):
            return value

    class _Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def synchronize(self):
            pass

    monkeypatch.setattr(pynccl_mod, "StatelessProcessGroup", _Group)
    monkeypatch.setattr(pynccl_mod, "NCCLLibrary", _Nccl)
    monkeypatch.setattr(pynccl_mod.torch.cuda, "device", lambda device: _Context())
    monkeypatch.setattr(pynccl_mod.torch.cuda, "Stream", _Context)
    monkeypatch.setattr(pynccl_mod.torch.cuda, "stream", lambda stream: _Context())
    monkeypatch.setattr(pynccl_mod.torch, "zeros", lambda *args, **kwargs: object())
    monkeypatch.setattr(PyNcclCommunicator, "all_reduce", lambda self, data: None)

    PyNcclCommunicator(
        _Group(), torch.device("cuda:0"), is_symmetric_memory_enabled=True
    )
    assert seen["config"].magic == NCCL_API_MAGIC
    assert seen["config"].graphUsageMode == 1


def test_wait_signal_desc_layout_matches_nccl():
    """Field order matters: ncclWaitSignal reads op_cnt from the first int32."""
    fields = [n for n, _ in W.ncclWaitSignalDesc_t._fields_]
    assert fields == ["op_cnt", "peer", "sig_idx", "ctx"]
    assert ctypes.sizeof(W.ncclWaitSignalDesc_t) == 16


def test_make_wait_descs_writes_each_peer_into_its_slot():
    """A peer/op_cnt column swap would wait on the wrong rank."""
    arr = PyNcclCommunicator.make_wait_descs([(1, 3), (0, 1)])
    assert len(arr) == 2
    assert (arr[0].peer, arr[0].op_cnt) == (1, 3)
    assert (arr[1].peer, arr[1].op_cnt) == (0, 1)


def test_make_wait_descs_keeps_storage_alive_after_allocator_churn():
    arr = PyNcclCommunicator.make_wait_descs([(7, 11)])
    for _ in range(1000):
        ctypes.create_string_buffer(16)
    assert (arr[0].peer, arr[0].op_cnt) == (7, 11)


def test_make_nccl_config_header_matches_initializer():
    """version must be seeded from the library; a wrong value is rejected."""

    class _Standin(PyNcclCommunicator):
        def __init__(self):
            self.nccl_version = 23007

    cfg = _Standin().make_nccl_config(num_rma_ctx=2)
    assert cfg.size == ctypes.sizeof(W.ncclConfig_t)
    assert cfg.magic == NCCL_API_MAGIC
    assert cfg.version == 23007
    assert cfg.numRmaCtx == 2


def test_make_nccl_config_fills_every_tunable_with_undef():
    """A new tunable added to the struct but unset here would hold 0, not the
    NCCL UNDEF sentinel (INT_MIN)."""

    class _Standin(PyNcclCommunicator):
        def __init__(self):
            self.nccl_version = 23007

    cfg = _Standin().make_nccl_config()
    for fld in (
        "blocking",
        "cgaClusterSize",
        "minCTAs",
        "maxCTAs",
        "splitShare",
        "trafficClass",
        "collnetEnable",
        "CTAPolicy",
        "shrinkShare",
        "nvlsCTAs",
        "nChannelsPerNetPeer",
        "nvlinkCentricSched",
        "graphUsageMode",
        "maxP2pPeers",
        "graphStreamOrdering",
    ):
        assert getattr(cfg, fld) == NCCL_CONFIG_UNDEF_INT, f"{fld} not UNDEF"


def _supports_rma(has_rma, version):
    class _Nccl:
        pass

    class _Probe(PyNcclCommunicator):
        def __init__(self, h, v):
            self.nccl = _Nccl()
            self.nccl.has_rma = h
            self.nccl_version = v

    return _Probe(has_rma, version).supports_rma()


def test_supports_rma_false_when_symbol_missing():
    """A lib without ncclPutSignal (e.g. torch's bundled 2.27.3) must be gated
    out, else AttributeError at runtime."""
    assert _supports_rma(has_rma=False, version=23007) is False


def test_supports_rma_false_below_min_version():
    """An old NCCL that happens to expose the symbol must still be gated out;
    the ABI stabilized at 2.30."""
    assert _supports_rma(has_rma=True, version=21903) is False


def test_supports_rma_true_at_version_boundary():
    """Pins the threshold; bumping it to 23001 would turn this red."""
    assert _supports_rma(has_rma=True, version=23000) is True


# --- Tier 2: E2E warm-up on a real 2-rank NCCL communicator ---

_GPU_2 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2),
    reason="needs >=2 CUDA GPUs",
)


def _group_has_nvlink() -> bool:
    """True iff the group's GPUs are all NVLink-connected. RMA requires NVLink
    symmetric memory; PCIe-only groups get a NULL window handle. Requires
    torch.distributed initialized."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() < 2:
        return False
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        is_full_nvlink,
    )

    local_gpu = torch.cuda.current_device()
    gpu_ids = [None] * dist.get_world_size()
    dist.all_gather_object(gpu_ids, local_gpu)
    gpu_ids = [int(x) for x in gpu_ids]
    return is_full_nvlink(gpu_ids, len(gpu_ids))


def _init_gloo_and_build_rma():
    import torch.distributed as dist

    from sglang.srt.distributed.device_communicators.rma_communicator import (
        NcclRmaCommunicator,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    if dist.get_world_size() < 2:
        pytest.skip("needs world_size>=2")
    device = torch.device("cuda", rank)
    pynccl_comm = PyNcclCommunicator(group=dist.group.WORLD, device=device)
    return NcclRmaCommunicator(pynccl_comm)


@_GPU_2
def test_rma_warmup_init_does_not_raise_on_unavailable_hw():
    """On RMA-unavailable HW NCCL returns success + a NULL window handle; the
    warm-up must skip without raising (reproduced on 2x RTX 4080 SUPER, no
    NVLink). Run under torchrun --nproc_per_node=2."""
    import torch.distributed as dist

    try:
        rma = _init_gloo_and_build_rma()
        assert rma.enabled is True
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@_GPU_2
def test_rma_warmup_success_path_on_nvlink():
    """On NVLink the ring put/wait handshake must verify, setting
    rma_available=True. Guards the success-path logic the graceful-skip test
    cannot exercise. Skipped on PCIe-only groups."""
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    torch.cuda.set_device(dist.get_rank())
    if not _group_has_nvlink():
        pytest.skip("needs NVLink-connected GPUs for the RMA success path")
    try:
        rma = _init_gloo_and_build_rma()
        assert rma.rma_available is True
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    from sglang.jit_kernel.tests.utils import multigpu_pytest_main

    multigpu_pytest_main(__name__, __file__, num_gpus=(2,))
