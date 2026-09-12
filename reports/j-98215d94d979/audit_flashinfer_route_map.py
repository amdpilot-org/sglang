#!/usr/bin/env python3
"""Audit the FlashInfer workaround associated with sglang#29160.

This is a source audit, not a CUDA reproduction.  Pass a downloaded
``csrc/trtllm_fused_moe_kernel_launcher.cu`` from a FlashInfer tag.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--expect", choices=("fixed", "vulnerable"), required=True)
    args = parser.parse_args()

    text = args.source.read_text()
    padded = len(re.findall(r"max_num_padded_tokens\s*\+\s*1", text))
    unpadded_route_allocations = len(
        re.findall(
            r"permuted_idx_to_token_idx\s*=\s*\n?\s*"
            r"alloc_tensor\(\{max_num_padded_tokens\}",
            text,
        )
    )

    # The current implementation has three independently reachable padded
    # sites: the common launcher workspace and the logits/precomputed routing
    # metadata entry points.  Requiring all three guards against a partial fix.
    fixed = padded >= 3 and unpadded_route_allocations == 0
    print(
        f"source={args.source} padded_occurrences={padded} "
        f"unpadded_route_allocations={unpadded_route_allocations} fixed={fixed}"
    )
    expected = args.expect == "fixed"
    return 0 if fixed == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
