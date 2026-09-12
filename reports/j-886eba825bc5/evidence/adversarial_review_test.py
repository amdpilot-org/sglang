import asyncio
import importlib.util
from pathlib import Path
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock


HELPERS = Path("/job/repo/test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py")
spec = importlib.util.spec_from_file_location("candidate_helpers", HELPERS)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
case = TestCase()


def test_abort_then_duplicate_abort_and_late_output_release_exactly_once():
    async def run():
        tm = helpers._make_tokenizer_manager(case)
        tm.enable_lora = True
        tm.lora_registry = MagicMock()
        tm.lora_registry.release = AsyncMock()
        rid = "adversarial-rid"
        state = helpers._make_req_state(rid)
        state.obj.lora_path = "/adapter"
        state.obj.lora_id = "adapter-id"
        tm.rid_to_state[rid] = state

        abort = helpers._make_abort_req(rid)
        tm._handle_abort_req(abort)
        tm._handle_abort_req(abort)
        await asyncio.sleep(0)

        # A scheduler result arriving after direct abort has no state to own.
        assert rid not in tm.rid_to_state
        tm.lora_registry.release.assert_awaited_once_with("adapter-id")

    asyncio.run(run())


def test_reused_rid_does_not_let_old_error_consumer_release_new_state():
    async def run():
        tm = helpers._make_tokenizer_manager(case)
        tm.enable_lora = True
        tm.lora_registry = MagicMock()
        tm.lora_registry.release = AsyncMock()
        rid = "reused-rid"
        old = helpers._make_req_state(rid)
        old.obj.lora_path = "/old"
        old.obj.lora_id = "old-id"
        new = helpers._make_req_state(rid)
        new.obj.lora_path = "/new"
        new.obj.lora_id = "new-id"
        tm.rid_to_state[rid] = new
        out = {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "status_code": 500,
                    "message": "old scheduler error",
                }
            }
        }

        assert await tm._handle_abort_finish_reason(out, old, True) is out
        assert tm.rid_to_state[rid] is new
        tm.lora_registry.release.assert_not_awaited()

    asyncio.run(run())
