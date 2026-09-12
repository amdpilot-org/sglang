#!/usr/bin/env python3
"""Deterministic checks for sgl-project/sglang#35785."""

import argparse
import os
import runpy
import sys
import types
import warnings
from pathlib import Path


def warning_for(source: Path, version: str, use_system: bool = True) -> list[str]:
    old = os.environ.get("AITER_USE_SYSTEM_TRITON")
    try:
        os.environ["AITER_USE_SYSTEM_TRITON"] = "1" if use_system else "0"
        sys.modules["triton"] = types.SimpleNamespace(__version__=version)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runpy.run_path(str(source), run_name=f"aiter_gluon_{version}_{use_system}")
        return [str(item.message) for item in caught]
    finally:
        sys.modules.pop("triton", None)
        if old is None:
            os.environ.pop("AITER_USE_SYSTEM_TRITON", None)
        else:
            os.environ["AITER_USE_SYSTEM_TRITON"] = old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aiter-source", type=Path, required=True)
    args = parser.parse_args()

    warning_34 = warning_for(args.aiter_source, "3.4.0")
    assert len(warning_34) == 1 and "require triton>=3.6.0, found 3.4.0" in warning_34[0]
    assert warning_for(args.aiter_source, "3.6.0") == []
    assert warning_for(args.aiter_source, "3.7.0+amd.rocm7.2.0.git89002410") == []

    root = Path(__file__).resolve().parents[2]
    general = (root / "docker/rocm.Dockerfile").read_text()
    rdna = (root / "docker/rocm-gfx1151.Dockerfile").read_text()

    # The reported gfx1150 is not silently mapped to a CDNA build or to gfx1151.
    assert " AS gfx1150" not in general
    assert "ARG GPU_ARCH=gfx1150" not in general
    assert "gfx1150" not in rdna

    # Current source has a distinct, deliberately constrained gfx1151 path.
    assert "ARG GPU_ARCH=gfx1151" in rdna
    assert "rocm/pytorch@sha256:" in rdna
    assert "ENV SGLANG_USE_AITER=0" in rdna
    assert "--attention-backend triton" in rdna

    # Supported ROCm 7.2 flavors replace old base Triton with a compatible pin.
    assert 'ARG TRITON_VERSION="3.7.0+amd.rocm7.2.0.git89002410"' in general
    assert '"triton==${TRITON_VERSION}"' in general

    print("reproduced warning:", warning_34[0])
    print("boundary triton 3.6.0: no warning")
    print("current pinned triton 3.7.0: no warning")
    print("gfx1150 stage: absent (unsupported; not aliased)")
    print("gfx1151 path: separate pinned image, Triton attention, AITER dispatch disabled")


if __name__ == "__main__":
    main()
