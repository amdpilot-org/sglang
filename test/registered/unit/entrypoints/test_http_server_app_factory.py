import asyncio
from types import SimpleNamespace

import httpx

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import build_app, init_app_state
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _TokenizerManager:
    def __init__(self, name):
        self.model_path = name
        self.served_model_name = name
        self.is_generation = True
        self.model_config = SimpleNamespace(
            context_len=128,
            is_image_understandable_model=False,
            is_audio_understandable_model=False,
            hf_config=SimpleNamespace(model_type="test", architectures=["Test"]),
        )
        self.lora_registry = {}

    def config_value(self, _name):
        return None


def _state(name):
    return http_server._GlobalState(
        tokenizer_manager=_TokenizerManager(name),
        template_manager=object(),
        scheduler_info={"name": name},
    )


async def _get(app, path):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get(path)


async def _get_both(first, second):
    return await asyncio.gather(_get(first, "/model_info"), _get(second, "/model_info"))


def test_two_apps_dispatch_native_routes_to_their_own_engine_state(monkeypatch):
    args = ServerArgs(model_path="unused")
    monkeypatch.setattr(
        http_server,
        "get_serving",
        lambda: SimpleNamespace(
            tokenizer_path="unused", preferred_sampling_params=None
        ),
    )
    first = build_app(args)
    second = build_app(args)
    first.state.global_state = _state("first")
    second.state.global_state = _state("second")

    first_response, second_response = asyncio.run(_get_both(first, second))

    assert first_response.json()["model_path"] == "first"
    assert second_response.json()["model_path"] == "second"
    assert http_server.get_global_state() is None


def test_init_app_state_does_not_mutate_compatibility_global():
    args = ServerArgs(model_path="unused")
    app = build_app(args)
    original = _state("legacy")
    http_server.set_global_state(original)
    engine = SimpleNamespace(
        server_args=args,
        tokenizer_manager=_TokenizerManager("engine"),
        template_manager=object(),
        _scheduler_init_result=SimpleNamespace(scheduler_infos=[{"engine": True}]),
    )

    init_app_state(engine, app.state, args)

    assert http_server.get_global_state() is original
    assert app.state.global_state.tokenizer_manager is engine.tokenizer_manager
    http_server.set_global_state(None)
