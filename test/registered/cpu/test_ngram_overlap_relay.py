from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang.srt.speculative.ngram_worker import NGRAMWorker


def _worker(*, enable_overlap: bool):
    worker = NGRAMWorker.__new__(NGRAMWorker)
    worker.enable_overlap = enable_overlap
    worker.draft_token_num = 4
    worker.max_trie_depth = 5
    worker.ngram_corpus = Mock()
    worker.ngram_corpus.batch_get.return_value = (
        np.arange(8, dtype=np.int32),
        np.zeros(8 * 4, dtype=np.bool_),
    )
    return worker


def _batch(*, grammar_needs_sync: bool):
    reqs = [
        SimpleNamespace(rid="a", origin_input_ids=[1, 2, 3], output_ids=[4, 5]),
        SimpleNamespace(rid="b", origin_input_ids=[10, 11], output_ids=[12]),
    ]
    return SimpleNamespace(
        reqs=reqs,
        spec_info=SimpleNamespace(
            # Fixed-width rows: request a accepted two tokens, request b one.
            accept_tokens=torch.tensor([6, 7, 90, 91, 13, 92, 93, 94]),
            accept_lens=torch.tensor([2, 1]),
        ),
        grammar_needs_sync=lambda: grammar_needs_sync,
    )


@pytest.mark.parametrize(
    ("enable_overlap", "grammar_needs_sync", "expected_tokens", "expected_lens"),
    [
        (True, False, [[3, 4, 5, 6, 7], [10, 11, 12, 13]], [7, 4]),
        (False, False, [[1, 2, 3, 4, 5], [10, 11, 12]], [5, 3]),
        # Grammar batches are synchronized before draft prep, so req.output_ids
        # already contains the accepted tail and the relay must not be appended.
        (True, True, [[1, 2, 3, 4, 5], [10, 11, 12]], [5, 3]),
    ],
)
def test_prepare_draft_tokens_relays_only_unprocessed_overlap_tail(
    enable_overlap, grammar_needs_sync, expected_tokens, expected_lens
):
    worker = _worker(enable_overlap=enable_overlap)

    worker._prepare_draft_tokens(
        _batch(grammar_needs_sync=grammar_needs_sync)
    )

    worker.ngram_corpus.synchronize.assert_called_once_with()
    worker.ngram_corpus.batch_get.assert_called_once_with(
        ["a", "b"], expected_tokens, expected_lens
    )


def test_prepare_draft_tokens_rejects_malformed_corpus_width():
    worker = _worker(enable_overlap=True)
    worker.ngram_corpus.batch_get.return_value = (
        np.arange(7, dtype=np.int32),
        np.zeros(7 * 4, dtype=np.bool_),
    )

    with pytest.raises(AssertionError, match="total_draft_token_num=7, bs=2"):
        worker._prepare_draft_tokens(_batch(grammar_needs_sync=False))
