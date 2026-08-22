"""Config fields for batch-wise adaptive pruning (BWAP)."""

from __future__ import annotations

import msgspec

from sglang.srt.arg_groups.arg_utils import A


class Bwap(msgspec.Struct):
    """Namespace ``bwap``."""

    _NS_PATH = "bwap"

    enable_bwap: A[
        bool,
        "Enable batch-wise adaptive pruning (BWAP), an opt-in, training-free FFN neuron pruning mode for gated-MLP models.",
    ] = False
    bwap_sparsity: A[
        float,
        "Target FFN neuron sparsity; sparse steps retain the top round((1 - sparsity) * D_FF) neurons per layer.",
    ] = 0.5
    bwap_t_init: A[
        int,
        "Number of initial dense exploration decode steps before the first BWAP mask is built.",
    ] = 8
    bwap_t_explore: A[
        int,
        "Number of dense exploration decode steps per BWAP refresh cycle.",
    ] = 4
    bwap_t_prune: A[
        int,
        "Number of sparse decode steps per BWAP refresh cycle.",
    ] = 16
    bwap_fused: A[
        bool,
        "Use gathered reduced-width GEMMs on eligible all-prune decode steps; unsupported layers fall back to activation masking.",
    ] = False
    bwap_probe: A[
        bool,
        "Force a frozen dummy BWAP mask into the decode graph for throughput measurement only; output is intentionally invalid.",
    ] = False
