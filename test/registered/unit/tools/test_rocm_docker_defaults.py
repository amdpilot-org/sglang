from pathlib import Path


DOCKERFILE = Path(__file__).parents[4] / "docker" / "rocm.Dockerfile"


def _stage(name: str) -> str:
    text = DOCKERFILE.read_text()
    marker = f" AS {name}\n"
    start = text.index(marker) + len(marker)
    end = text.find("\nFROM ", start)
    return text[start:] if end == -1 else text[start:end]


def test_rocm700a_workaround_default_is_scoped_to_the_reported_image():
    assert 'ENV SGLANG_USE_ROCM700A="0"' in _stage("gfx950-rocm720")

    for stage in (
        "gfx942",
        "gfx942-rocm720",
        "gfx942-rocm724",
        "gfx950",
        "gfx950-rocm724",
        "gfx942-rocm1000",
        "gfx950-rocm1000",
        "gfx1250-rocm1000",
    ):
        assert 'ENV SGLANG_USE_ROCM700A="1"' in _stage(stage)

    final_stage = DOCKERFILE.read_text().split("FROM ${GPU_ARCH}", 1)[1]
    assert "ENV SGLANG_USE_ROCM700A=" not in final_stage
