from unittest.mock import Mock

import pytest

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar


@pytest.mark.parametrize(
    ("rollback_count", "expected_tokens"),
    [
        (1, [10, 20, 30]),
        (2, [10, 20]),
        (4, []),
    ],
)
def test_rollback_removes_tokens_in_place(rollback_count, expected_tokens):
    grammar = XGrammarGrammar.__new__(XGrammarGrammar)
    grammar.matcher = Mock()
    grammar.accepted_tokens = [10, 20, 30, 40]
    accepted_tokens = grammar.accepted_tokens

    grammar.rollback(rollback_count)

    grammar.matcher.rollback.assert_called_once_with(rollback_count)
    assert grammar.accepted_tokens is accepted_tokens
    assert grammar.accepted_tokens == expected_tokens


def test_rollback_zero_keeps_accepted_tokens():
    grammar = XGrammarGrammar.__new__(XGrammarGrammar)
    grammar.matcher = Mock()
    grammar.accepted_tokens = [10, 20, 30]
    accepted_tokens = grammar.accepted_tokens

    grammar.rollback(0)

    grammar.matcher.rollback.assert_called_once_with(0)
    assert grammar.accepted_tokens is accepted_tokens
    assert grammar.accepted_tokens == [10, 20, 30]
