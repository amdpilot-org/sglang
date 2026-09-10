from __future__ import annotations

import json
import os
import sys
import traceback
from contextlib import contextmanager
from types import SimpleNamespace

os.environ.setdefault("SGLANG_DSA_TOPK_BROADCAST", "0")

import torch

import sglang.srt.layers.attention.dsa.dsa_indexer as dsa_indexer_module


from sglang.srt.layers.attention.dsa_backend import (
    DSAMetadata,
    DeepseekSparseAttnBackend,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)


dsa_indexer_module.maybe_capture_indexer_topk = lambda layer_id, topk_indices: topk_indices
dsa_indexer_module.logits_head_gate_graph = (
    lambda x, weight, n_heads_inv_sqrt, softmax_scale, q_scale: x
)
dsa_indexer_module.scale_head_gate_graph = (
    lambda weights_raw, n_heads_inv_sqrt, softmax_scale, q_scale: weights_raw
)


DEVICE = torch.device("cuda:0")
BATCH = 2
TOKENS_PER_SEQUENCE = 130
BLOCKS_PER_SEQUENCE = (TOKENS_PER_SEQUENCE + 63) // 64
PAGE_SIZE = 64


def independent_page_ids(token_table: torch.Tensor) -> list[list[int]]:
    expected: list[list[int]] = []
    for row in range(token_table.shape[0]):
        row_ids: list[int] = []
        for block in range(BLOCKS_PER_SEQUENCE):
            token_id = int(token_table[row, block * PAGE_SIZE].item())
            row_ids.append(token_id // PAGE_SIZE)
        expected.append(row_ids)
    return expected


def make_token_table(base_block: int) -> torch.Tensor:
    table = torch.empty(BATCH, TOKENS_PER_SEQUENCE, dtype=torch.int32, device=DEVICE)
    for row in range(BATCH):
        for offset in range(TOKENS_PER_SEQUENCE):
            block = base_block + row * BLOCKS_PER_SEQUENCE + offset // PAGE_SIZE
            table[row, offset] = block * PAGE_SIZE + offset % PAGE_SIZE
    return table


class ReducedBackend:
    def __init__(self, token_table: torch.Tensor):
        self.real_page_size = PAGE_SIZE
        self.token_table = token_table
        self.real_page_table = torch.empty(
            BATCH, BLOCKS_PER_SEQUENCE, dtype=torch.int32, device=DEVICE
        )
        self.metadata_seen = 0

    def get_indexer_metadata(self, layer_id: int, forward_batch) -> DSAIndexerMetadata:
        transformed = DeepseekSparseAttnBackend._transform_table_1_to_real(
            self, self.token_table
        )
        self.real_page_table.copy_(transformed)
        self.metadata_seen += 1
        return DSAIndexerMetadata(
            attn_metadata=DSAMetadata(
                page_size=PAGE_SIZE,
                cache_seqlens_int32=torch.full(
                    (BATCH,), TOKENS_PER_SEQUENCE, dtype=torch.int32, device=DEVICE
                ),
                max_seq_len_q=BATCH,
                max_seq_len_k=TOKENS_PER_SEQUENCE,
                cu_seqlens_q=torch.arange(
                    0, BATCH + 1, dtype=torch.int32, device=DEVICE
                ),
                cu_seqlens_k=torch.arange(
                    0, (BATCH + 1) * TOKENS_PER_SEQUENCE,
                    TOKENS_PER_SEQUENCE,
                    dtype=torch.int32,
                    device=DEVICE,
                ),
                page_table_1=self.token_table,
                real_page_table=self.real_page_table,
                dsa_cache_seqlens_int32=torch.full(
                    (BATCH,), TOKENS_PER_SEQUENCE, dtype=torch.int32, device=DEVICE
                ),
                dsa_cu_seqlens_q=torch.arange(
                    0, BATCH + 1, dtype=torch.int32, device=DEVICE
                ),
                dsa_cu_seqlens_k=torch.arange(
                    0, (BATCH + 1) * TOKENS_PER_SEQUENCE,
                    TOKENS_PER_SEQUENCE,
                    dtype=torch.int32,
                    device=DEVICE,
                ),
                dsa_extend_seq_lens_list=[BATCH] * BATCH,
                dsa_seqlens_expanded=torch.arange(
                    1, BATCH + 1, dtype=torch.int32, device=DEVICE
                ),
            ),
            topk_transform_method=TopkTransformMethod.PAGED,
            topk_backend=DSATopKBackend.TORCH,
        )


class ReducedIndexer(Indexer):
    def __init__(self, backend: ReducedBackend):
        self.backend = backend
        self.index_topk = BLOCKS_PER_SEQUENCE
        self.use_dsa_indexer_fusion = False
        self.dsa_enable_prefill_cp = False
        self.weights_proj = SimpleNamespace(
            set_lora=False,
            weight=torch.zeros(16, 16, dtype=torch.bfloat16, device=DEVICE),
        )
        self.alt_stream = None
        self.n_heads = 16
        self.softmax_scale = 1.0
        self.block_size = 16
        self.scale_fmt = None

    def _should_skip_logits_computation(self, forward_batch) -> bool:
        return True

    def _forward_cuda_k_only(
        self,
        x,
        positions,
        forward_batch,
        layer_id,
        act_quant,
        metadata,
        return_indices,
        num_tokens=None,
        topk_result=None,
    ):
        assert metadata is not None, "DSA paging object is None before get_page_table_64"
        page_table = metadata.get_page_table_64()
        assert page_table is not None, "get_page_table_64 returned None"
        if topk_result is None:
            topk_result = torch.empty(
                x.shape[0], self.index_topk, dtype=torch.int32, device=DEVICE
            )
        topk_result.view(-1).copy_(page_table.reshape(-1))
        return topk_result

    def _get_q_k_bf16(self, q_lora, x, positions, enable_dual_stream, *, forward_batch=None):
        return x, x, x

    def _get_logits_head_gate(self, x, q_scale):
        return x

    def _store_index_k_cache(self, **kwargs):
        return None

    def _get_topk_ragged(
        self,
        enable_dual_stream,
        forward_batch,
        layer_id,
        q_fp8,
        weights,
        metadata,
        topk_result=None,
    ):
        assert metadata is not None, "DSA paging object is None before get_page_table_64"
        page_table = metadata.get_page_table_64()
        if topk_result is None:
            return page_table.reshape(-1)
        topk_result.view(-1).copy_(page_table.reshape(-1))
        return topk_result


@contextmanager
def contexts(
    backend: ReducedBackend,
    indexer: ReducedIndexer,
    forward_batch,
    *,
    full_graph: bool = False,
):
    with (
        forward_context(ForwardContext(attn_backend=backend)),
        set_tc_piecewise_forward_context(
            forward_batch,
            [],
            None,
            [],
            [],
            dsa_indexers=[indexer],
            num_tokens=BATCH,
            raw_num_tokens=BATCH,
            full_graph=full_graph,
        ),
    ):
        yield


def run_eager(backend: ReducedBackend, indexer: ReducedIndexer, forward_batch):
    with contexts(backend, indexer, forward_batch):
        result = indexer.forward_cuda(
            torch.zeros(BATCH, 16, dtype=torch.bfloat16, device=DEVICE),
            torch.zeros(BATCH, 16, dtype=torch.bfloat16, device=DEVICE),
            torch.zeros(BATCH, dtype=torch.int64, device=DEVICE),
            forward_batch,
            0,
        )
    torch.cuda.synchronize()
    return result


def run_full_graph(backend: ReducedBackend, indexer: ReducedIndexer, forward_batch):
    graph = torch.cuda.CUDAGraph()
    with contexts(backend, indexer, forward_batch, full_graph=True):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            result = indexer.forward_cuda(
                torch.zeros(BATCH, 16, dtype=torch.bfloat16, device=DEVICE),
                torch.zeros(BATCH, 16, dtype=torch.bfloat16, device=DEVICE),
                torch.zeros(BATCH, dtype=torch.int64, device=DEVICE),
                forward_batch,
                0,
            )
    return graph, result


def run_breakable(backend: ReducedBackend, indexer: ReducedIndexer, forward_batch):
    graph = BreakableCUDAGraph()
    stream = torch.cuda.Stream(DEVICE)
    with contexts(backend, indexer, forward_batch), enable_breakable_cuda_graph():
        torch.cuda.synchronize()
        with BreakableCUDAGraphCapture(graph, stream=stream):
            pre = torch.zeros(BATCH, device=DEVICE)
            pre.copy_(torch.ones(BATCH, device=DEVICE))
            result = indexer.forward_cuda(
                torch.zeros(BATCH, 16, dtype=torch.bfloat16, device=DEVICE),
                torch.zeros(BATCH, 16, dtype=torch.bfloat16, device=DEVICE),
                torch.zeros(BATCH, dtype=torch.int64, device=DEVICE),
                forward_batch,
                0,
            )
    return graph, result


def check_readback(result: torch.Tensor, expected: list[list[int]]) -> None:
    actual = result.reshape(-1).tolist()
    flattened = [value for row in expected for value in row]
    assert actual == flattened, (actual, flattened)


def main() -> int:
    source = os.environ.get("SGLANG_SOURCE", os.getcwd())
    report = {
        "source": source,
        "python": sys.executable,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "modes": {},
    }

    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_num_tokens=BATCH,
        attn_cp_metadata=None,
    )

    for mode in ("eager", "full_graph", "breakable"):
        entry = {"status": "not_run"}
        report["modes"][mode] = entry
        try:
            token_table = make_token_table(7)
            backend = ReducedBackend(token_table)
            indexer = ReducedIndexer(backend)
            expected = independent_page_ids(token_table)

            if mode == "eager":
                result = run_eager(backend, indexer, forward_batch)
            elif mode == "full_graph":
                graph, result = run_full_graph(backend, indexer, forward_batch)
                graph.replay()
            else:
                graph, result = run_breakable(backend, indexer, forward_batch)

            check_readback(result, expected)
            initial_expected = expected
            token_table = make_token_table(31)
            backend.token_table.copy_(token_table)
            expected = independent_page_ids(token_table)
            if mode == "full_graph":
                graph.replay()
            else:
                if mode == "breakable":
                    with contexts(
                        backend, indexer, forward_batch
                    ), enable_breakable_cuda_graph():
                        graph.replay()
                else:
                    result = run_eager(backend, indexer, forward_batch)
            torch.cuda.synchronize()
            check_readback(result, expected)
            entry.update(
                status="pass",
                metadata_calls=backend.metadata_seen,
                initial_expected=initial_expected,
                replay_expected=expected,
                readback=result.reshape(-1).tolist(),
            )
        except Exception as error:
            entry.update(
                status="fail",
                error_type=type(error).__name__,
                error=str(error),
                trace=traceback.format_exc(),
            )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(v["status"] == "pass" for v in report["modes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
