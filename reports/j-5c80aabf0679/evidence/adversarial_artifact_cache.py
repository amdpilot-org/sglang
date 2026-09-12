import asyncio
import sys
from unittest.mock import patch

sys.path.insert(0, "/job/repo/test/registered/unit/multimodal")
from test_media_artifact_processor import _Processor


async def main():
    processor = _Processor()
    processor.trust_mm_content_hashes = True
    try:
        first = await processor.prepare_media_artifacts(
            [b"first payload"], cache_ids=["stable-id"]
        )
        with patch(
            "sglang.srt.multimodal.media_artifacts.base.snapshot_media",
            side_effect=AssertionError("hot caller-ID hit performed media I/O"),
        ):
            second = await processor.prepare_media_artifacts(
                ["unreadable-second-source"], cache_ids=["stable-id"]
            )
        assert first[0] is second[0]
        assert len(processor.batches) == 1
        print("generic artifact cache hot hit skipped snapshot and preprocessing")
    finally:
        processor.close()


asyncio.run(main())
