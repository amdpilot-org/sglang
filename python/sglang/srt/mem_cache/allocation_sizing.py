from __future__ import annotations

from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
    get_spec,
    max_speculative_num_draft_tokens,
)


def get_alloc_page_size() -> int:
    # Mirrors _build_token_to_kv_pool_allocator's DCP branch; the platform
    # allocators that skip it page smaller, so this is an upper bound for them.
    return get_schedule().page_size * get_parallel().attn_dcp_size


def get_alloc_len_per_decode() -> int:
    """KV length one request may allocate in a single decode step.

    Reads the bags: adaptive speculative decoding moves the step count and the
    draft-token bound after publish, and this runs per decode batch.
    """
    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
    spec_steps = spec.speculative_num_steps or 1
    spec_topk = spec.speculative_eagle_topk or 1
    spec_tokens = max_speculative_num_draft_tokens()
    page_size = get_alloc_page_size()

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(spec.speculative_algorithm)
    if spec_algo.is_uno():
        if spec_tokens is None:
            raise RuntimeError("UNO requires speculative_num_draft_tokens")
        # UNO retains an additional clean-root position beside Q/F draft slots.
        return spec_tokens + 1
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # spec v2 tree (page>1, topk>1): worst-case page-aligned footprint per
        # topk branch is ceil((page_size-1 + num_steps) / page) pages, each branch
        # duplicated -- reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode() -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_utils.eagle_prepare_for_decode.
    """
    return 2 * get_alloc_len_per_decode()


def decode_seq_lens_from_reqs(reqs, *, is_encoder_decoder: bool):
    """Return the scheduler's exact host-side target sequence lengths."""
    seq_lens = [r.seqlen for r in reqs]
    if not is_encoder_decoder:
        return seq_lens
    return [
        seq_len
        - (
            r.multimodal_inputs.num_image_tokens
            if r.multimodal_inputs is not None
            else 0
        )
        for r, seq_len in zip(reqs, seq_lens, strict=True)
    ]


def page_aligned_decode_alloc_lens(
    reqs,
    *,
    reserve: int,
    page_size: int,
    base_kv_lens=None,
):
    """Whole-page decode alloc lens: nxt rounds committed up to page so allocated
    == recorded (unaligned tails leak at ps>1).

    ``base_kv_lens`` lets callers with a synchronous sequence-length mirror size
    from that mirror instead of a lagging per-request committed watermark.
    """
    if base_kv_lens is None:
        base_kv_lens = [r.kv.kv_committed_len for r in reqs]
    if len(base_kv_lens) != len(reqs):
        raise ValueError("base_kv_lens must have one entry per request")

    cur_kv_lens = [0] * len(reqs)
    nxt_kv_lens = [0] * len(reqs)
    num_needed_tokens = 0
    for i, (r, base_kv_len) in enumerate(zip(reqs, base_kv_lens, strict=True)):
        cur = r.kv.kv_allocated_len
        nxt = max(
            cur,
            (int(base_kv_len) + reserve + page_size - 1) // page_size * page_size,
        )
        cur_kv_lens[i] = cur
        nxt_kv_lens[i] = nxt
        num_needed_tokens += nxt - cur
    return cur_kv_lens, nxt_kv_lens, num_needed_tokens


def get_req_to_token_extra_context_len() -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation; the spec v2 page>1 topk>1 holey
    draft footprint can outgrow the default num_draft_tokens headroom. The row
    headroom and the pools it sits next to derive from the same bag leaves, so
    they cannot disagree after a post-publish override.
    """
    # FIXME(lsyin): temporary fix for the context length issue under spec decoding
    extra = 4 + (max_speculative_num_draft_tokens() or 0)
    page_size = get_alloc_page_size()
    spec_algorithm = get_spec().speculative_algorithm
    if spec_algorithm is not None:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        spec_algo = SpeculativeAlgorithm.from_string(spec_algorithm)
        if page_size > 1 or spec_algo.is_uno():
            # UNO's double-buffer reserve applies at every page size. Larger
            # pages may additionally round the allocation up by page_size - 1.
            extra = max(extra, get_alloc_reserve_per_decode() + page_size - 1)
    return extra
