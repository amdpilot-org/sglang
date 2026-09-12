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


def run(label, entry_only=None):
    manager = GrammarManager.__new__(GrammarManager)
    manager.grammar_backend = MagicMock(spec=BaseGrammarBackend)
    manager.grammar_backend.get_cached_or_future_value.return_value = (Future(), False)
    manager.grammar_queue = []
    manager._enable_strict_thinking = False
    if entry_only is not None:
        manager.tp_grammar_entry_only = entry_only
        manager.is_grammar_sync_entry = False
    req = make_request()
    queued = manager.process_req_with_grammar(req)
    print(label, {
        "queued": queued,
        "compile_lookup_calls": manager.grammar_backend.get_cached_or_future_value.call_count,
        "grammar_type": type(req.grammar).__name__,
    })


if __name__ == "__main__":
    import sys
    mode = sys.argv[1]
    if mode == "base":
        run("base_non_entry_tp")
    elif mode == "candidate-ordinary":
        run("candidate_ordinary_tp", True)
    elif mode == "candidate-speculative":
        run("candidate_speculative_tp", False)
    elif mode == "candidate-cp":
        run("candidate_context_parallel_tp", False)
    else:
        raise SystemExit(mode)
