import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"


def _copy_then_delete_pairs(dockerfile: str) -> set[tuple[str, str]]:
    """Return cross-instruction COPY destinations later removed by RUN."""
    instructions = re.findall(
        r"(?ms)^(COPY|RUN)\s+(.+?)(?=^[A-Z][A-Z0-9_]*\s|\Z)", dockerfile
    )
    copies: list[str] = []
    pairs: set[tuple[str, str]] = set()
    for kind, body in instructions:
        if kind == "COPY":
            destination = body.strip().split()[-1]
            copies.append(destination.rstrip("/"))
        else:
            for destination in copies:
                if re.search(rf"\brm\s+-rf\s+{re.escape(destination)}(?:\s|$)", body):
                    pairs.add((destination, body.strip()))
    return pairs


def test_detector_ignores_content_that_remains_in_the_image():
    dockerfile = "COPY --from=build /tool /usr/local/bin/tool\nRUN chmod +x /usr/local/bin/tool\n"
    assert _copy_then_delete_pairs(dockerfile) == set()


def test_detector_finds_deletion_in_a_later_layer_only():
    dockerfile = "COPY --from=build /out /tmp/out\nRUN consume /tmp/out && rm -rf /tmp/out\n"
    assert {path for path, _ in _copy_then_delete_pairs(dockerfile)} == {"/tmp/out"}


def test_transient_builder_artifacts_are_not_committed_to_final_image_layers():
    dockerfile = DOCKERFILE.read_text()
    transient_paths = {
        "/tmp/wheels/hpc-ops",
        "/tmp/local_src",
        "/tmp/gateway_wheels",
    }

    offending_paths = {path for path, _ in _copy_then_delete_pairs(dockerfile)}
    assert offending_paths.isdisjoint(transient_paths)

    expected_mounts = {
        "from=hpc_ops_builder,source=/wheels,target=/tmp/wheels/hpc-ops",
        "from=local_src,source=/src,target=/tmp/local_src",
        "from=gateway_builder,source=/build/gateway_wheels,target=/tmp/gateway_wheels",
    }
    assert all(mount in dockerfile for mount in expected_mounts)
