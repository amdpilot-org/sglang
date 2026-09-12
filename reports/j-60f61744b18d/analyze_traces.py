#!/usr/bin/env python3
"""Summarize decode-marker/kernel overlap in the retained Chrome traces."""

import gzip
import json
import sys
from pathlib import Path


result = {}
for arg in sys.argv[1:]:
    path = Path(arg)
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    steps = [
        event for event in events
        if event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("step[DECODE")
    ]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    launches = [event for event in events if event.get("name") == "hipGraphLaunch"]
    overlap = [
        sum(
            kernel["ts"] < step["ts"] + step.get("dur", 0)
            and kernel["ts"] + kernel.get("dur", 0) > step["ts"]
            for kernel in kernels
        )
        for step in steps
    ]
    result[path.parent.parent.name] = {
        "trace": str(path),
        "decode_cpu_marker_count": len(steps),
        "gpu_kernel_count": len(kernels),
        "hip_graph_launch_count": len(launches),
        "gpu_kernel_overlap_count_per_decode_marker": overlap,
        "decode_markers_with_any_overlapping_kernel": sum(value > 0 for value in overlap),
        "last_decode_marker_end_us": max(
            step["ts"] + step.get("dur", 0) for step in steps
        ),
        "last_kernel_end_us": max(
            kernel["ts"] + kernel.get("dur", 0) for kernel in kernels
        ),
    }
print(json.dumps(result, indent=2, sort_keys=True))
