from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.constrained.base_grammar_backend import BaseGrammarBackend
from sglang.srt.constrained.grammar_manager import GrammarManager


def make_request():
    req = MagicMock()
    req.sampling_params = SimpleNamespace(
        json_schema='{"type":"object"}',
        regex=None,
        ebnf=None,
        structural_tag=None,
        custom_params=None,
    )
    req.require_reasoning = False
    req.grammar = None
    req.grammar_key = None
    req.grammar_wait_ct = 0
    req.finished.return_value = False
    return req


def run_scenario(name: str, entry_only: bool):
    manager = GrammarManager.__new__(GrammarManager)
    manager.grammar_backend = MagicMock(spec=BaseGrammarBackend)
    manager.grammar_backend.get_cached_or_future_value.return_value = (
        Future(),
        False,
    )
    manager.tp_grammar_entry_only = entry_only
    manager.is_grammar_sync_entry = False
    manager.grammar_queue = []
    manager._enable_strict_thinking = False

    req = make_request()
    queued = manager.process_req_with_grammar(req)
    calls = manager.grammar_backend.get_cached_or_future_value.call_count
    print(
        name,
        {
            "queued": queued,
            "compile_lookup_calls": calls,
            "grammar_type": type(req.grammar).__name__,
        },
    )
    return calls


if __name__ == "__main__":
    # Candidate initialization selects entry_only=False for both configurations.
    speculative_calls = run_scenario("speculative_tp", entry_only=False)
    context_parallel_calls = run_scenario("context_parallel_tp", entry_only=False)
    ordinary_calls = run_scenario("ordinary_tp", entry_only=True)
    assert speculative_calls == 1
    assert context_parallel_calls == 1
    assert ordinary_calls == 0
