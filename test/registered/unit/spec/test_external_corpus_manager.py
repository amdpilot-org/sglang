import threading
import unittest

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    RemoveExternalCorpusReqInput,
)
from sglang.srt.speculative.external_corpus_manager import ExternalCorpusManager


class _BlockingWorker:
    def __init__(self):
        self.corpora = {"already_loaded": 3}
        self.load_started = threading.Event()
        self.release_load = threading.Event()
        self.removed = []

    def add_external_corpus(self, corpus_id, token_chunks):
        self.load_started.set()
        if not self.release_load.wait(timeout=5):
            raise TimeoutError("test did not release the pending corpus load")
        return sum(len(chunk) for chunk in token_chunks)

    def commit_corpus_load(self, corpus_id, loaded_token_count):
        self.corpora[corpus_id] = loaded_token_count

    def remove_external_corpus(self, corpus_id):
        self.removed.append(corpus_id)
        self.corpora.pop(corpus_id, None)

    def list_external_corpora(self):
        return dict(self.corpora)


class TestExternalCorpusManagerRemove(unittest.TestCase):
    def setUp(self):
        self.worker = _BlockingWorker()
        self.responses = []
        self.manager = ExternalCorpusManager(
            self.worker,
            lambda output, request: self.responses.append((output, request)),
        )

    def tearDown(self):
        self.worker.release_load.set()
        pending = self.manager._pending_load
        if pending is not None:
            pending[1].join(timeout=5)

    def start_blocked_load(self, corpus_id="loading"):
        result = self.manager.add(
            AddExternalCorpusReqInput(
                corpus_id=corpus_id,
                token_chunks=[[1, 2, 3], [4, 5]],
            )
        )
        self.assertIsNone(result)
        self.assertTrue(self.worker.load_started.wait(timeout=5))

    def finish_load(self):
        self.worker.release_load.set()
        pending = self.manager._pending_load
        self.assertIsNotNone(pending)
        pending[1].join(timeout=5)
        self.assertFalse(pending[1].is_alive())
        self.manager.check_pending_load()

    def test_remove_rejects_same_corpus_while_load_is_pending(self):
        self.start_blocked_load()

        result = self.manager.remove(RemoveExternalCorpusReqInput(corpus_id="loading"))

        self.assertFalse(result.success)
        self.assertIn("still being loaded", result.message)
        self.assertEqual(self.worker.removed, [])

        self.finish_load()
        self.assertEqual(self.worker.corpora["loading"], 5)

    def test_remove_allows_different_corpus_while_load_is_pending(self):
        self.start_blocked_load()

        result = self.manager.remove(
            RemoveExternalCorpusReqInput(corpus_id="already_loaded")
        )

        self.assertTrue(result.success)
        self.assertNotIn("already_loaded", self.worker.corpora)
        self.assertEqual(self.worker.removed, ["already_loaded"])

    def test_remove_allows_same_corpus_after_load_finishes(self):
        self.start_blocked_load()
        self.finish_load()

        result = self.manager.remove(RemoveExternalCorpusReqInput(corpus_id="loading"))

        self.assertTrue(result.success)
        self.assertNotIn("loading", self.worker.corpora)
        self.assertEqual(self.worker.removed, ["loading"])


if __name__ == "__main__":
    unittest.main()
