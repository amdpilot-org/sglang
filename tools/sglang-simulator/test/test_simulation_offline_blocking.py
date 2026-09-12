import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_simulation_sglang_serving import (
    SIM_CONFIGS,
    SGLangServingRunner,
    assert_decode_metrics,
)

REQUEST_RATE = 1
SEED = 123
ARRIVAL_TOLERANCE_S = 0.05
DURATION_TOLERANCE = 0.05
LATENCY_TOLERANCE = 0.25
RELATIVE_TOLERANCES = {
    "duration": DURATION_TOLERANCE,
    "request_throughput": DURATION_TOLERANCE,
    "input_throughput": DURATION_TOLERANCE,
    "output_throughput": DURATION_TOLERANCE,
    "mean_e2e_latency_ms": LATENCY_TOLERANCE,
    "mean_ttft_ms": LATENCY_TOLERANCE,
    "mean_tpot_ms": LATENCY_TOLERANCE,
    "mean_itl_ms": LATENCY_TOLERANCE,
}


def _relative_error(actual, expected):
    return abs(actual - expected) / abs(expected)


def _run_mode(mode, tmp_path):
    case_dir = tmp_path / mode
    case_dir.mkdir()
    runner = SGLangServingRunner(SIM_CONFIGS["aic_sol"], case_dir, mode=mode)
    try:
        metrics = runner.benchmark(
            case_dir / "benchmark.json", request_rate=REQUEST_RATE, seed=SEED
        )
    finally:
        runner.shutdown()

    requests = [
        json.loads(line)
        for line in (runner.output_dir / "request.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    requests.sort(key=lambda request: request["created_time"])
    return metrics, requests


def test_request_rate_offline_matches_blocking(tmp_path):
    offline_metrics, offline_requests = _run_mode("offline", tmp_path)
    blocking_metrics, blocking_requests = _run_mode("blocking", tmp_path)

    for metrics in (offline_metrics, blocking_metrics):
        assert_decode_metrics(metrics)

    assert len(offline_requests) == len(blocking_requests) == 3

    offline_arrivals = [request["created_time"] for request in offline_requests]
    blocking_arrivals = [request["created_time"] for request in blocking_requests]
    assert offline_arrivals[1] > 0.5
    assert blocking_arrivals[1] > 0.5
    assert offline_arrivals == pytest.approx(blocking_arrivals, abs=ARRIVAL_TOLERANCE_S)
    assert (
        offline_metrics["max_concurrent_requests"]
        == blocking_metrics["max_concurrent_requests"]
        == 1
    )

    for key in ("completed", "total_input", "total_output"):
        assert offline_metrics[key] == blocking_metrics[key]

    for key, tolerance in RELATIVE_TOLERANCES.items():
        error = _relative_error(offline_metrics[key], blocking_metrics[key])
        assert error <= tolerance, (
            key,
            offline_metrics[key],
            blocking_metrics[key],
            error,
            tolerance,
        )


@pytest.mark.parametrize("mode", ["OFFLINE", "BLOCKING"])
@pytest.mark.parametrize(
    ("predictor_wall_delays", "predictor_perf_delays"),
    [
        ((0.25, 0.001), (0.25, 0.001)),
        ((0.001, 0.25), (0.001, 0.25)),
        ((0.125, 0.125), (0.25, 0.25)),
    ],
)
def test_predictor_query_wall_time_does_not_change_offline_ttft(
    monkeypatch, mode, predictor_wall_delays, predictor_perf_delays
):
    """Offline latency ignores predictor time even when clock domains diverge."""
    from sglang_simulator.simulation.sglang import scheduler
    from sglang_simulator.simulation.types import SimulationMode

    hook = scheduler.C_SchedulerHook
    state = scheduler.StateManager
    stats_manager = scheduler.request_stats_manager
    clock = SimpleNamespace(wall=100.0, perf=500.0)

    def advance(duration):
        clock.wall += duration
        clock.perf += duration

    monkeypatch.setattr(
        scheduler,
        "time",
        SimpleNamespace(
            time=lambda: clock.wall,
            perf_counter=lambda: clock.perf,
            sleep=advance,
        ),
    )
    monkeypatch.setattr(hook, "SIM_MODE", SimulationMode(mode))
    monkeypatch.setattr(hook, "OVERLAP_SCHEDULE", False)
    monkeypatch.setattr(hook, "ITERATION_STATS", [])
    monkeypatch.setattr(hook, "TOTAL_PREDICTOR_TIME_COST", 0.0)
    monkeypatch.setattr(hook, "SIMULATION_BATCH", None)

    wall_delays = iter(predictor_wall_delays)
    perf_delays = iter(predictor_perf_delays)
    predicted_latencies = iter([0.1, 0.05])

    def predict(_batch):
        clock.wall += next(wall_delays)
        clock.perf += next(perf_delays)
        return next(predicted_latencies)

    monkeypatch.setattr(
        hook,
        "INFERENCE_PREDICTOR",
        SimpleNamespace(predict_infer_time=predict),
    )
    req = SimpleNamespace(
        rid="r1", extend_input_len=4, prefix_indices=[], output_ids=[]
    )
    batch = SimpleNamespace(
        reqs=[req],
        forward_mode=SimpleNamespace(
            is_extend=lambda: not req.output_ids,
            is_decode=lambda: bool(req.output_ids),
        ),
    )
    monkeypatch.setattr(scheduler, "get_obj_from_args", lambda _type_name, obj: obj)

    class GenerationBatchResult:
        pass

    def run_batch(_self, _batch):
        advance(0.03)
        return GenerationBatchResult()

    def process_batch_result(_self, _batch):
        advance(0.04)
        req.output_ids.append(1)

    target = SimpleNamespace(
        _prefetch_kvcache=Mock(),
        get_new_batch_prefill=Mock(),
        run_batch=run_batch,
        process_batch_result=process_batch_result,
        event_loop_normal=Mock(),
        init_request_dispatcher=Mock(),
    )
    hook.hook(target)
    state.reset()
    stats_manager.reset()
    state.set_last_real_time_ts(clock.wall)
    req_stats = stats_manager.get_req_stats(req.rid)
    req_stats.created_time = 0.0
    req_stats.input_length = 4
    req_stats.output_length = 2
    req_stats.queue_start = req_stats.queue_end = 0.0

    try:
        for _ in predictor_wall_delays:
            advance(0.02)
            target.run_batch(None, batch)
            target.process_batch_result(None, batch)

        expected_latencies = [0.19, 0.14]
        if mode == "BLOCKING":
            expected_latencies = [
                latency + delay
                for latency, delay in zip(expected_latencies, predictor_wall_delays)
            ]
        assert req_stats.gen_token_latencies == pytest.approx(expected_latencies)
        metrics = scheduler.calc_metrics([req_stats])
        assert metrics["mean_ttft_ms"] == pytest.approx(expected_latencies[0] * 1000)
        assert metrics["mean_itl_ms"] == pytest.approx(expected_latencies[1] * 1000)
        assert hook.TOTAL_PREDICTOR_TIME_COST == pytest.approx(
            sum(predictor_perf_delays)
        )
    finally:
        state.reset()
        stats_manager.reset()
