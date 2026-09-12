import asyncio
from unittest.mock import patch

from sglang.srt.disaggregation.encoder.runtime import EncoderScheduler, PendingRequest
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.schedule_batch import Modality


def request(req_id, modality="image"):
    return {
        "req_id": req_id,
        "modality": modality,
        "mm_items": [object()],
        "num_parts": 1,
        "part_idx": 0,
    }


def test_post_lock_mixed_cancelled_completed_and_live_payload():
    class Encoder:
        def __init__(self):
            self.encode_dispatch_lock = asyncio.Lock()
            self.seen = []

        async def batch_encode(self, requests, modality):
            self.seen.append(([r["req_id"] for r in requests], modality))
            return [(1, 2, 3, None, None) for _ in requests]

    async def run():
        encoder = Encoder()
        scheduler = EncoderScheduler(encoder, [object(), object()], 4)
        loop = asyncio.get_running_loop()
        cancelled = PendingRequest(request("cancelled"), loop)
        completed = PendingRequest(request("completed"), loop)
        live = PendingRequest(request("live"), loop)
        sent = []

        await encoder.encode_dispatch_lock.acquire()
        with patch(
            "sglang.srt.disaggregation.encoder.runtime.sock_send",
            side_effect=lambda sock, payload: sent.append(unwrap_from_pickle(payload)),
        ):
            task = asyncio.create_task(
                scheduler._dispatch_group(
                    [cancelled, completed, live], Modality.IMAGE
                )
            )
            await asyncio.sleep(0)
            cancelled.future.cancel()
            completed.future.set_result((9, 9, 9, None, None))
            encoder.encode_dispatch_lock.release()
            await task

        assert encoder.seen == [(["live"], Modality.IMAGE)]
        assert len(sent) == 2
        assert all([r["req_id"] for r in payload["requests"]] == ["live"] for payload in sent)
        assert await live.future == (1, 2, 3, None, None)

    asyncio.run(run())


def test_post_lock_all_abandoned_has_no_side_effects():
    class Encoder:
        def __init__(self):
            self.encode_dispatch_lock = asyncio.Lock()
            self.calls = 0

        async def batch_encode(self, requests, modality):
            self.calls += 1
            raise AssertionError("abandoned batch reached encoder")

    async def run():
        encoder = Encoder()
        scheduler = EncoderScheduler(encoder, [object()], 2)
        loop = asyncio.get_running_loop()
        first = PendingRequest(request("first"), loop)
        second = PendingRequest(request("second"), loop)
        sent = []

        await encoder.encode_dispatch_lock.acquire()
        with patch(
            "sglang.srt.disaggregation.encoder.runtime.sock_send",
            side_effect=lambda sock, payload: sent.append(payload),
        ):
            task = asyncio.create_task(
                scheduler._dispatch_group([first, second], Modality.IMAGE)
            )
            await asyncio.sleep(0)
            first.future.cancel()
            second.future.cancel()
            encoder.encode_dispatch_lock.release()
            await task

        assert encoder.calls == 0
        assert sent == []

    asyncio.run(run())


def test_per_request_cancelled_while_prior_encode_blocks():
    class Encoder:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.seen = []

        async def encode(self, **kwargs):
            self.seen.append(kwargs["req_id"])
            if kwargs["req_id"] == "blocked":
                self.started.set()
                await self.release.wait()
            return (1, 2, 3, None, None)

    async def run():
        encoder = Encoder()
        scheduler = EncoderScheduler(encoder, [object()], 3)
        loop = asyncio.get_running_loop()
        blocked = PendingRequest(request("blocked", "video"), loop)
        cancelled = PendingRequest(request("cancelled", "video"), loop)
        live = PendingRequest(request("live", "video"), loop)
        sent = []

        with patch(
            "sglang.srt.disaggregation.encoder.runtime.sock_send",
            side_effect=lambda sock, payload: sent.append(payload),
        ):
            task = asyncio.create_task(
                scheduler._dispatch_group(
                    [blocked, cancelled, live], Modality.VIDEO
                )
            )
            await encoder.started.wait()
            cancelled.future.cancel()
            encoder.release.set()
            await task

        assert encoder.seen == ["blocked", "live"]
        assert len(sent) == 2
        assert await blocked.future == (1, 2, 3, None, None)
        assert await live.future == (1, 2, 3, None, None)

    asyncio.run(run())
