from sglang.benchmark.serving import format_metric_scope_note


def test_metric_scope_note_distinguishes_client_and_engine_windows():
    note = format_metric_scope_note()

    assert "per-request" in note
    assert "after the first token" in note
    assert "stream events" in note
    assert "may omit the terminal response tail" in note
    assert "engine log" in note
    assert "recent reporting window" in note
    assert "whole-run" in note
    assert "not expected to match exactly" in note


def test_metric_scope_note_does_not_claim_tpot_is_aggregate_throughput():
    note = format_metric_scope_note().lower()

    assert "tpot × concurrency" not in note
    assert "not directly comparable" in note
    assert "whole request lifetime" not in note
