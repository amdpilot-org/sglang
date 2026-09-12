import json

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader


def _select(tmp_path, weight_map, draft_model_idx=1):
    files = [
        str(tmp_path / "target.safetensors"),
        str(tmp_path / "draft-requested.safetensors"),
        str(tmp_path / "draft-unknown.safetensors"),
    ]
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return files, RunaiModelStreamerLoader._select_mtp_safetensors(
        files, str(tmp_path), draft_model_idx
    )


def test_mixed_recognized_and_unknown_draft_layout_falls_back(tmp_path):
    files, selected = _select(
        tmp_path,
        {
            "model.layers.0.weight": "target.safetensors",
            "mtp.1.decoder.weight": "draft-requested.safetensors",
            "model.nextn.layers.1.shared.weight": "draft-unknown.safetensors",
        },
    )
    assert selected == files


def test_recognized_layout_still_prunes_target_only_shard(tmp_path):
    files, selected = _select(
        tmp_path,
        {
            "model.layers.0.weight": "target.safetensors",
            "mtp.1.decoder.weight": "draft-requested.safetensors",
            "mtp.shared.weight": "draft-unknown.safetensors",
        },
    )
    assert selected == files[1:]
