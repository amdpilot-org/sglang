from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer


def batch(*rids):
    return SimpleNamespace(reqs=[SimpleNamespace(rid=rid) for rid in rids])


def inquirer_with(batches):
    obj = object.__new__(SchedulerLoadInquirer)
    object.__setattr__(obj, "get_running_batches", lambda: batches)
    return obj


def test_phase_independent_across_four_pp_slots():
    slots = [batch("r0"), batch("r1", "r2"), batch(), batch("r3", "r4")]
    assert inquirer_with(slots)._get_num_running_reqs() == 5


def test_same_identity_in_multiple_slots_is_counted_once():
    assert inquirer_with([batch("shared", "a"), batch("shared", "b")])._get_num_running_reqs() == 3


def test_distinct_objects_with_same_request_id_are_counted_once():
    first = SimpleNamespace(rid="same-rid")
    second = SimpleNamespace(rid="same-rid")
    batches = [SimpleNamespace(reqs=[first]), SimpleNamespace(reqs=[second])]
    assert first is not second
    assert inquirer_with(batches)._get_num_running_reqs() == 1


def test_non_pp_singleton_and_all_empty_boundaries():
    assert inquirer_with((batch("only"),))._get_num_running_reqs() == 1
    assert inquirer_with([batch(), batch(), batch(), batch()])._get_num_running_reqs() == 0
