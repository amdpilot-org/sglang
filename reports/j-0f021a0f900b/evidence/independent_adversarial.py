from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler_pp_mixin import PPBootstrapDecision, SchedulerPPMixin


def scheduler(rank=3):
    value = SchedulerPPMixin()
    value.ps = SimpleNamespace(pp_rank=rank, pp_size=8)
    value.pp_group = SimpleNamespace(is_first_rank=rank == 0, is_last_rank=rank == 7)
    value.disagg_prefill_bootstrap_queue = SimpleNamespace(
        queue=[], pop_bootstrapped=MagicMock(return_value=([], []))
    )
    value.waiting_queue = []
    return value


# Reproduce the precise prior-review counterexample with independently chosen RID.
s = scheduler()
s.process_bootstrapped_queue(PPBootstrapDecision(0, ("independent-rid",), ()))
try:
    s.process_bootstrapped_queue(PPBootstrapDecision(0, (), ("independent-rid",)))
except RuntimeError as exc:
    assert "payload divergence" in str(exc)
else:
    raise AssertionError("divergent duplicate was silently accepted")
assert s.disagg_prefill_bootstrap_queue.pop_bootstrapped.call_count == 1

# A downstream stage must reject a mutable/unsequenced reduction payload.
s2 = scheduler()
try:
    s2.process_bootstrapped_queue([["independent-rid"], []])
except RuntimeError as exc:
    assert "expected a sequenced bootstrap decision" in str(exc)
else:
    raise AssertionError("downstream accepted an unsequenced payload")

# Abort after decision 0 is held until decision 0 crosses this stage.
s3 = scheduler()
s3.abort_request = MagicMock()
abort = AbortReq(rid="independent-rid", pp_bootstrap_abort_after_sequence=0)
assert s3._pp_order_or_defer_abort_request(abort) is True
assert not s3.abort_request.called
s3.process_bootstrapped_queue(PPBootstrapDecision(0, (), ()))
s3.abort_request.assert_called_once_with(abort)

print("independent adversarial protocol cases passed")
