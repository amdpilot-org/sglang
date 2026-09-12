from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    topk_transform_paged_v2_with_optional_raw,
)


def test_v2_with_raw_indices_avoids_v1_and_translates_pages():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores = torch.empty((2, 8), device=device)
    seq_lens = torch.tensor([8, 5], dtype=torch.int32, device=device)
    page_tables = torch.tensor(
        [[7, 3], [11, 5]], dtype=torch.int32, device=device
    )
    out_pages = torch.empty((2, 4), dtype=torch.int32, device=device)
    out_raw = torch.empty_like(out_pages)
    metadata = torch.empty((3, 2), dtype=torch.int32, device=device)
    selected = torch.tensor(
        [[0, 3, 4, 7], [4, -1, 0, 3]], dtype=torch.int32, device=device
    )

    def fake_v2(scores, seq_lens, page_tables, output, page_size, metadata):
        assert page_tables is None
        output.copy_(selected)

    with (
        patch(
            "sglang.srt.layers.attention.dsv4.indexer.topk_transform_paged_v2",
            side_effect=fake_v2,
        ) as v2,
        patch(
            "sglang.srt.layers.attention.dsv4.indexer.topk_transform_paged"
        ) as v1,
    ):
        topk_transform_paged_v2_with_optional_raw(
            scores,
            seq_lens,
            page_tables,
            out_pages,
            page_size=4,
            metadata=metadata,
            out_raw_indices=out_raw,
        )

    v2.assert_called_once()
    v1.assert_not_called()
    torch.testing.assert_close(out_raw, selected)
    torch.testing.assert_close(
        out_pages,
        torch.tensor(
            [[28, 31, 12, 15], [20, -1, 44, 47]],
            dtype=torch.int32,
            device=device,
        ),
    )


def test_v2_without_raw_indices_keeps_fused_page_transform():
    scores = torch.empty((1, 8))
    seq_lens = torch.tensor([8], dtype=torch.int32)
    page_tables = torch.tensor([[7, 3]], dtype=torch.int32)
    out_pages = torch.empty((1, 4), dtype=torch.int32)
    metadata = torch.empty((2, 2), dtype=torch.int32)

    with patch(
        "sglang.srt.layers.attention.dsv4.indexer.topk_transform_paged_v2"
    ) as v2:
        topk_transform_paged_v2_with_optional_raw(
            scores,
            seq_lens,
            page_tables,
            out_pages,
            page_size=4,
            metadata=metadata,
        )

    v2.assert_called_once_with(
        scores, seq_lens, page_tables, out_pages, 4, metadata
    )


def test_v2_raw_translation_handles_page_size_one_and_padding():
    scores = torch.empty((1, 3))
    seq_lens = torch.tensor([3], dtype=torch.int32)
    page_tables = torch.tensor([[9, 2, 6]], dtype=torch.int32)
    out_pages = torch.empty((1, 4), dtype=torch.int32)
    out_raw = torch.empty_like(out_pages)
    metadata = torch.empty((2, 2), dtype=torch.int32)

    def fake_v2(scores, seq_lens, page_tables, output, page_size, metadata):
        output.copy_(torch.tensor([[2, 0, -1, 1]], dtype=torch.int32))

    with patch(
        "sglang.srt.layers.attention.dsv4.indexer.topk_transform_paged_v2",
        side_effect=fake_v2,
    ):
        topk_transform_paged_v2_with_optional_raw(
            scores,
            seq_lens,
            page_tables,
            out_pages,
            page_size=1,
            metadata=metadata,
            out_raw_indices=out_raw,
        )

    torch.testing.assert_close(
        out_pages, torch.tensor([[6, 9, -1, 2]], dtype=torch.int32)
    )
