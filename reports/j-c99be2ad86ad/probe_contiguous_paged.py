from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    _jit_topk_v2_module,
    plan_topk_v2,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)


PAGE_SIZE = 64
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1
BATCH = 4
SEQ = 65537
K = 512
WIDTH = (SEQ + 3) & ~3


def make_scores() -> torch.Tensor:
    """Finite, unique scores with winners on page boundaries and row end."""
    scores = torch.zeros(BATCH, WIDTH, dtype=torch.float32, device="cuda")[:, :SEQ]
    page_starts = torch.arange(0, SEQ, PAGE_SIZE, dtype=torch.int64, device="cuda")[:256]
    page_ends = torch.arange(PAGE_SIZE - 1, SEQ, PAGE_SIZE, dtype=torch.int64, device="cuda")[:128]
    tail = torch.arange(SEQ - 128, SEQ - 1, dtype=torch.int64, device="cuda")
    winners = torch.cat((page_starts, page_ends, tail, torch.tensor([SEQ - 1], device="cuda")))
    assert winners.numel() == K and torch.unique(winners).numel() == K
    top_values = torch.arange(SEQ - K, SEQ, dtype=torch.float32, device="cuda")
    losers = torch.arange(0, SEQ - K, dtype=torch.float32, device="cuda")
    loser_positions = torch.tensor(
        sorted(set(range(SEQ)) - set(winners.cpu().tolist())),
        dtype=torch.int64,
        device="cuda",
    )
    for row in range(BATCH):
        scores[row, winners] = top_values
        scores[row, loser_positions] = losers
    return scores


def make_page_tables() -> tuple[torch.Tensor, torch.Tensor]:
    num_pages = (SEQ + PAGE_SIZE - 1) // PAGE_SIZE
    torch.manual_seed(0xC0FFEE)
    tables = torch.stack(
        [torch.randperm(num_pages, device="cuda") for _ in range(BATCH)]
    ).to(torch.int32)
    inverse = torch.empty_like(tables, device="cpu")
    arange = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    for row in range(BATCH):
        inverse[row] = arange.cpu()
        inverse[row, tables[row].long().cpu()] = arange.cpu()
    return tables, inverse


def invert_page_indices(indices: torch.Tensor, inverse: torch.Tensor) -> list[list[int]]:
    result = []
    for row, values in enumerate(indices.cpu().tolist()):
        result.append(
            [
                (int(inverse[row, value >> PAGE_BITS]) << PAGE_BITS) | (value & PAGE_MASK)
                for value in values
                if value != -1
            ]
        )
    return result


def independent_reference(scores: torch.Tensor) -> list[list[int]]:
    reference = []
    for row in range(scores.shape[0]):
        order = sorted(range(SEQ), key=lambda index: (-float(scores[row, index]), index))
        reference.append(order[:K])
    return reference


def main() -> None:
    started = time.perf_counter()
    scores_ragged = make_scores()
    scores_paged = make_scores()
    assert torch.equal(scores_ragged, scores_paged)
    seq_lens = torch.full((BATCH,), SEQ, dtype=torch.int32, device="cuda")
    starts = torch.zeros(BATCH, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(BATCH, dtype=torch.int32, device="cuda")
    page_tables, inverse = make_page_tables()

    real_module = _jit_topk_v2_module()
    recorder = MagicMock(wraps=real_module)
    with patch(
        "sglang.kernels.ops.attention.dsv4.topk._jit_topk_v2_module",
        return_value=recorder,
    ):
        metadata = plan_topk_v2(seq_lens)
        torch.cuda.synchronize()

        contiguous_out = torch.full((BATCH, K), -1, dtype=torch.int32, device="cuda")
        topk_transform_ragged_v2(
            scores_ragged,
            seq_lens,
            out_offsets=offsets,
            out_indices=contiguous_out,
            row_starts=starts,
        )
        torch.cuda.synchronize()

        raw_out = torch.full((BATCH, K), -1, dtype=torch.int32, device="cuda")
        topk_transform_paged_v2(scores_paged, seq_lens, None, raw_out, PAGE_SIZE, metadata)
        torch.cuda.synchronize()

        paged_out = torch.full((BATCH, K), -1, dtype=torch.int32, device="cuda")
        topk_transform_paged_v2(
            scores_paged, seq_lens, page_tables, paged_out, PAGE_SIZE, metadata
        )
        torch.cuda.synchronize()

    contiguous = [sorted(row) for row in contiguous_out.cpu().tolist()]
    raw = [sorted(row) for row in raw_out.cpu().tolist()]
    paged_raw = invert_page_indices(paged_out, inverse)
    paged_raw = [sorted(row) for row in paged_raw]
    reference = independent_reference(scores_paged)

    exact_reference = all(
        set(contiguous[row]) == set(reference[row])
        and set(raw[row]) == set(reference[row])
        and set(paged_raw[row]) == set(reference[row])
        for row in range(BATCH)
    )
    selected_values = {
        "contiguous": [
            sorted(float(scores_paged[row, index]) for index in contiguous[row])
            for row in range(BATCH)
        ],
        "raw_paged": [
            sorted(float(scores_paged[row, index]) for index in raw[row])
            for row in range(BATCH)
        ],
        "paged_transformed": [
            sorted(float(scores_paged[row, index]) for index in paged_raw[row])
            for row in range(BATCH)
        ],
    }
    value_preserved = all(
        selected_values["contiguous"][row]
        == selected_values["raw_paged"][row]
        == selected_values["paged_transformed"][row]
        for row in range(BATCH)
    )
    dispatch_calls = sorted(call[0] for call in recorder.method_calls)
    mismatches = []
    for row in range(BATCH):
        expected = set(reference[row])
        for label, selected in (
            ("contiguous", contiguous[row]),
            ("raw_paged", raw[row]),
            ("paged_transformed", paged_raw[row]),
        ):
            extra = sorted(set(selected) - expected)
            missing = sorted(expected - set(selected))
            if extra or missing:
                mismatches.append(
                    {
                        "row": row,
                        "mode": label,
                        "extra_indices": extra[:16],
                        "extra_values": [float(scores_paged[row, index]) for index in extra[:16]],
                        "missing_indices": missing[:16],
                        "missing_values": [float(scores_paged[row, index]) for index in missing[:16]],
                    }
                )

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "architecture": torch.cuda.get_device_properties(0).gcnArchName,
        "batch": BATCH,
        "seq": SEQ,
        "k": K,
        "page_size": PAGE_SIZE,
        "finite_unique_scores": True,
        "winner_placement": "page starts, page ends, tail, and final position",
        "independent_reference": "CPU stable descending score, ascending index",
        "contiguous_matches_reference": exact_reference,
        "raw_paged_matches_reference": exact_reference,
        "paged_transformed_matches_reference": exact_reference,
        "selected_value_multisets_preserved": value_preserved,
        "reference_mismatches": mismatches[:12],
        "dispatch_calls": dispatch_calls,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path("/job/evidence/probe_contiguous_paged_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    assert exact_reference, "same finite scores selected different top-k indices"
    assert value_preserved, "same finite scores selected different top-k values"
    assert "topk_transform_ragged" in dispatch_calls
    assert "topk_transform_paged" in dispatch_calls


if __name__ == "__main__":
    main()
