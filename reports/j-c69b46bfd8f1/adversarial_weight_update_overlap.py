import asyncio
import json
import threading
import time
from types import SimpleNamespace

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.managers.tokenizer_manager import ServerStatus


class FakeTokenizerManager:
    def __init__(self, loop):
        self.event_loop = loop
        self.server_status = ServerStatus.Up
        self.is_pause = False
        self.gracefully_exit = False
        self.started = 0
        self.finish = [asyncio.Event(), asyncio.Event()]

    async def update_weights_from_disk(self, obj, request=None):
        index = self.started
        self.started += 1
        await self.finish[index].wait()
        return True, f"done-{index}", 0


class Callback:
    def __init__(self):
        self.calls = []

    def __call__(self, payload, **kwargs):
        self.calls.append((payload, kwargs))


loop = asyncio.new_event_loop()
thread = threading.Thread(target=loop.run_forever, daemon=True)
thread.start()
tm = FakeTokenizerManager(loop)
handle = RuntimeHandle.__new__(RuntimeHandle)
handle.tokenizer_manager = tm
handle._event_loop = loop
handle._grpc_weight_update_in_progress = False

first = Callback()
second = Callback()
handle.update_weights_from_disk("model-a", None, first)
handle.update_weights_from_disk("model-b", None, second)

deadline = time.time() + 5
while tm.started != 2 and time.time() < deadline:
    time.sleep(0.01)
assert tm.started == 2, tm.started
print("both_running", handle.health_check(), handle.get_operational_state())

loop.call_soon_threadsafe(tm.finish[0].set)
deadline = time.time() + 5
while not first.calls and time.time() < deadline:
    time.sleep(0.01)
assert first.calls
state = json.loads(handle.get_operational_state())
print("one_still_running", handle.health_check(), json.dumps(state, sort_keys=True))
assert not state["ready_to_serve"], state
assert state["weight_update_in_progress"], state

loop.call_soon_threadsafe(tm.finish[1].set)
deadline = time.time() + 5
while not second.calls and time.time() < deadline:
    time.sleep(0.01)
assert second.calls
loop.call_soon_threadsafe(loop.stop)
thread.join(timeout=5)
