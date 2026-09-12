import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PauseGenerationReqInput


source = Path("/job/repo/test/registered/unit/managers/test_scheduler_pause_generation.py")
spec = importlib.util.spec_from_file_location("candidate_pause_tests", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Base = module.TestSchedulerPauseGeneration


class TestAdversarialCandidate(Base):
    def test_live_chunk_already_in_running_batch_still_retires_sender(self):
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler._add_request_to_queue = MagicMock()
        scheduler.clear_pending_chunk_send = MagicMock()
        scheduler.last_batch = None

        req = self._make_req("chunked-in-running")
        req.disagg_kv_sender = MagicMock()
        req.metadata_buffer_index = 7
        scheduler.chunked_req = req
        scheduler.running_batch.reqs = [req]

        with patch("sglang.srt.managers.scheduler.retract_all"):
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.clear_pending_chunk_send.assert_called_once_with(req)
        req.disagg_kv_sender.abort.assert_called_once_with()
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)

    def test_sender_abort_failure_does_not_skip_cache_boundary(self):
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler._add_request_to_queue = MagicMock()
        scheduler.clear_pending_chunk_send = MagicMock()
        scheduler.last_batch = None

        req = self._make_req("abort-fails")
        req.disagg_kv_sender = MagicMock()
        req.disagg_kv_sender.abort.side_effect = RuntimeError("transport unavailable")
        req.metadata_buffer_index = 7
        scheduler.chunked_req = req

        with patch("sglang.srt.managers.scheduler.retract_all"):
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.tree_cache.reset.assert_called_once_with()
        scheduler.token_to_kv_pool_allocator.clear.assert_called_once_with()
