from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_compute_mamba_ratio_uses_separate_spec_scratch_and_stash_margin():
    skill = (ROOT / ".claude/skills/compute-mamba-ratio/SKILL.md").read_text()

    assert "r_variable  =  S · token_equiv" in skill
    assert "C_spec = min(floor(max_running_requests / attention_dp_size)" in skill
    assert "target_concurrency_per_attention_dp * S + 1" in skill
    assert "non-usable tensor padding row" in skill


def test_qwen_calculator_does_not_charge_drafts_as_allocator_slots():
    calculator = (
        ROOT / "docs/src/snippets/_qwen38_mamba_ratio_calculator.jsx"
    ).read_text()

    assert "(slots + drafts) * stateBytesPerSlot" not in calculator
    assert "(concurrency + 1) *" in calculator
    assert "Math.ceil(C) * slots + 1" in calculator


def test_kimi_calculator_sizes_plain_spec_scratch_per_attention_dp():
    calculator = (
        ROOT / "docs/src/snippets/_kimi_k3_mamba_ratio_calculator.jsx"
    ).read_text()

    assert "(slots + drafts) * stateBytesPerSlot" not in calculator
    assert "Math.ceil(target / dp)" in calculator
    assert "(concurrencyPerDp + 1) * drafts * stateBytesPerSlot" in calculator
