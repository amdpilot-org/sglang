import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager


def manager(session="decode:1234"):
    return SimpleNamespace(
        engine=SimpleNamespace(send_probe=MagicMock(return_value=0)),
        failed_sessions=set(),
        session_failures=defaultdict(int),
        session_generations=defaultdict(int),
        session_lock=threading.Lock(),
        decode_kv_args_table={},
    )


def test_registration_bookkeeping_can_erase_failure_after_publication():
    """Exercise the source ordering: publish args, concurrent transfer fails, mark registration."""
    session = "decode:1234"
    m = manager(session)
    m.decode_kv_args_table[session] = object()
    with m.session_lock:
        m.session_failures[session] = m.session_failures.get(session, 0) + 1
        m.failed_sessions.add(session)
    MooncakeKVManager._mark_session_registered(m, session)
    assert session in m.failed_sessions
    assert m.session_failures[session] == 1


def test_two_generations_with_equal_counts_reject_old_probe():
    session = "decode:1234"
    started = threading.Event()
    finish = threading.Event()

    def probe(_):
        started.set()
        assert finish.wait(5)
        return 0

    m = manager(session)
    m.engine.send_probe.side_effect = probe
    m.failed_sessions.add(session)
    m.session_failures[session] = 1
    m.session_generations[session] = 0
    thread = threading.Thread(target=MooncakeKVManager._run_one_probe_pass, args=(m,))
    thread.start()
    assert started.wait(5)
    MooncakeKVManager._mark_session_registered(m, session)
    with m.session_lock:
        m.session_failures[session] = 1
        m.failed_sessions.add(session)
    finish.set()
    thread.join(5)
    assert not thread.is_alive()
    assert session in m.failed_sessions
    assert m.session_failures[session] == 1
