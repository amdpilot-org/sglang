import importlib.util
import math
from pathlib import Path

import pytest

P = Path('/tmp/amdpilot-repo-j-533a411ed246/review-evidence/test_hybrid_verify_group_padding.py')
spec = importlib.util.spec_from_file_location('candidate_group_tests', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('tp,width', [(1,1),(3,5),(4,6),(7,8),(8,7),(16,6)])
@pytest.mark.parametrize('batch_size', [0,1,2,17,33,38,39,65])
def test_arbitrary_fixed_width_alignment(tp, width, batch_size):
    fb = m._batch(batch_size=batch_size, width=width, spec=m._spec(width))
    m._prepare(fb, m._model_runner(), attn_tp_size=tp)
    expected = math.ceil(batch_size * width / math.lcm(tp, width)) * math.lcm(tp, width)
    assert fb.global_num_tokens_cpu == [expected]
    assert fb.input_ids.shape[0] == expected
    assert fb.batch_size == expected // width
    assert expected % tp == 0 and expected % width == 0


@pytest.mark.parametrize('tokens', [[0, 6, 12], [30, 198, 234], [35, 77, 0]])
def test_dp_sum_all_local_domains_remain_complete(tokens):
    width, tp = 6, 8
    for rank, real_tokens in enumerate(tokens):
        assert real_tokens % width == 0 or tokens == [35, 77, 0]
        batch = real_tokens // width
        fb = m._batch(batch_size=batch, width=width, spec=m._spec(width), global_num_tokens=tokens)
        m._prepare(fb, m._model_runner(), attn_tp_size=tp, attn_dp_rank=rank, attn_dp_size=len(tokens))
        for padded in fb.global_num_tokens_cpu:
            assert padded % tp == 0 and padded % width == 0

