from sglang.srt.model_executor.pool_configurator import HybridSWAPoolConfigurator
from sglang.srt.utils.common import ceil_align


def allocation(c, pages, available, page_size, fixed_swa_tokens=None):
    full_tokens = pages * page_size
    swa_tokens = fixed_swa_tokens if fixed_swa_tokens is not None else int(full_tokens * c._swa_full_tokens_ratio) // page_size * page_size
    target_full = c._full_per_token * c._full_layers_num
    target_swa = c._swa_per_token * (c._swa_layers_num + c._draft_swa_layers_num)
    target_bytes = full_tokens * target_full + swa_tokens * target_swa
    virtual_span = max(target_bytes // target_full - 1, 0)
    draft_tokens = ceil_align(virtual_span, page_size) + page_size
    draft_bpt = c._full_capacity_draft_pool_bytes_per_token()
    return target_bytes + draft_tokens * draft_bpt


cases = 0
for page_size in (1, 7, 128, 256):
    for ratio in (0.0, 0.1, 0.25, 1.0, 1.75):
        for full_layers, swa_layers in ((1, 1), (16, 16), (3, 29)):
            for draft_full, draft_swa, draft_swa_full in ((8, 0, 0), (3, 2, 1), (0, 2, 4)):
                c = object.__new__(HybridSWAPoolConfigurator)
                c._full_per_token = 2112
                c._swa_per_token = 528
                c._full_layers_num = full_layers
                c._swa_layers_num = swa_layers
                c._draft_full_layers_num = draft_full
                c._draft_swa_layers_num = draft_swa
                c._draft_swa_full_layers_num = draft_swa_full
                c._draft_cell_size = 37
                c._swa_full_tokens_ratio = ratio
                for fixed in (None, page_size * 3):
                    for budget_pages in range(1, 80):
                        budget = budget_pages * page_size * 2112 * full_layers + (budget_pages % 5) * 997
                        if allocation(c, 0, budget, page_size, fixed) > budget:
                            continue
                        tokens = c._max_full_tokens_with_draft_pools(budget, page_size, fixed)
                        pages = tokens // page_size
                        used = allocation(c, pages, budget, page_size, fixed)
                        next_used = allocation(c, pages + 1, budget, page_size, fixed)
                        assert used <= budget, (page_size, ratio, full_layers, swa_layers, draft_full, draft_swa, draft_swa_full, fixed, budget, tokens, used)
                        assert next_used > budget, (page_size, ratio, full_layers, swa_layers, draft_full, draft_swa, draft_swa_full, fixed, budget, tokens, next_used)
                        cases += 1
print(f"validated_cases={cases}")
