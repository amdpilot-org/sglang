"""Behavioral reproduction for unsupported caller-ID video/audio paths."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.multimodal.processors.whisper import WhisperProcessor


class ExpectedStop(Exception):
    pass


async def reproduce_video():
    calls = []

    async def load_mm_data(**kwargs):
        calls.append(kwargs)
        raise ExpectedStop

    fake = SimpleNamespace(load_mm_data=load_mm_data, mm_tokens=object())
    request = SimpleNamespace(
        video_data=["unavailable.mp4"],
        audio_data=None,
        mm_cache_ids=["stable-video-id"],
        rid="r",
    )
    for _ in range(2):
        try:
            await QwenVLImageProcessor.process_mm_data_async(
                fake, [], "prompt", request
            )
        except ExpectedStop:
            pass
    assert len(calls) == 2
    assert all(call["video_data"] == ["unavailable.mp4"] for call in calls)
    print("video load attempts with repeated cache_id:", len(calls))


async def reproduce_audio():
    calls = []

    def load_audio(source):
        calls.append(source)
        raise ExpectedStop

    fake = SimpleNamespace(
        _pop_sampling_param=lambda request, key: request.sampling_params.pop(
            key, None
        )
    )
    request = SimpleNamespace(
        mm_cache_ids=["stable-audio-id"], sampling_params={}
    )
    with patch(
        "sglang.srt.multimodal.processors.whisper.load_audio", load_audio
    ):
        for _ in range(2):
            try:
                await WhisperProcessor.process_mm_data_async(
                    fake, [], ["unavailable.wav"], "prompt", request
                )
            except ExpectedStop:
                pass
    assert calls == ["unavailable.wav", "unavailable.wav"]
    print("audio load attempts with repeated cache_id:", len(calls))


async def main():
    await reproduce_video()
    await reproduce_audio()


asyncio.run(main())
