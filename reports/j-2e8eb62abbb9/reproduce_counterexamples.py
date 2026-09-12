import math


def configured_slots(budget_in_slot_bytes, slots_per_request, drafts):
    """Mirror kv_cache_configurator.py's plain-spec joint solve."""
    return math.floor(
        (budget_in_slot_bytes - (1 + drafts))
        / (1 + drafts / slots_per_request)
    )


def candidate_kimi_budget(concurrency, slots_per_request, drafts):
    return (
        concurrency * slots_per_request
        + 2
        + (concurrency + 1) * drafts
    )


def candidate_qwen_budget(concurrency, slots_per_request, drafts):
    ssm_bytes = 48 * 48 * 128 * 128 * 4
    conv_slot_bytes = 48 * 10240 * 3 * 2
    state_bytes = ssm_bytes + conv_slot_bytes
    conv_intermediate_bytes = 48 * 10240 * (drafts + 2) * 2
    physical_scratch = (concurrency + 1) * (
        drafts * ssm_bytes + conv_intermediate_bytes
    )
    return concurrency * slots_per_request + 2 + physical_scratch / state_bytes


def corrected_budget(concurrency, slots_per_request, drafts):
    safe_slots = concurrency * slots_per_request + 1
    return safe_slots * (1 + drafts / slots_per_request) + 1 + drafts


cases = [
    ("kimi", 6, 5, 6, candidate_kimi_budget),
    ("kimi", 64, 5, 8, candidate_kimi_budget),
    ("qwen", 64, 5, 8, candidate_qwen_budget),
]

for model, concurrency, slots, drafts, candidate_budget_fn in cases:
    before_budget = candidate_budget_fn(concurrency, slots, drafts)
    after_budget = corrected_budget(concurrency, slots, drafts)
    before_slots = configured_slots(before_budget, slots, drafts)
    after_slots = configured_slots(after_budget, slots, drafts)
    required = concurrency * slots + 1
    print(
        f"{model} C={concurrency} S={slots} D={drafts}: "
        f"candidate_budget={before_budget:.6f} candidate_K={before_slots} "
        f"corrected_budget={after_budget:.6f} corrected_K={after_slots} "
        f"required_safe_K={required}"
    )
    assert before_slots < required
    assert after_slots == required
