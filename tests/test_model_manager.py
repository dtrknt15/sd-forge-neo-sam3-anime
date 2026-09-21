"""Unit tests for sam3_anime.model_manager (no torch/sam3 required)."""

from pathlib import Path

import pytest

from sam3_anime import model_manager, vram


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Isolate global _STATE and env between tests."""
    saved = dict(model_manager._STATE)
    monkeypatch.delenv("SAM3_CHECKPOINT", raising=False)
    yield
    model_manager._STATE.clear()
    model_manager._STATE.update(saved)


def _make_models_dir(tmp_path: Path, names: list[str]) -> Path:
    folder = tmp_path / "SAM3"
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"x")
    return folder


@pytest.fixture
def patch_dir(monkeypatch):
    def _patch(folder: Path):
        monkeypatch.setattr(model_manager, "_webui_models_dir", lambda: folder)
        return folder

    return _patch


# ---------------------------------------------------------------- remap


def test_remap_strips_detector_prefix():
    raw = {"detector.model.vision.0.weight": "w", "detector.other.bias": "b"}
    out = model_manager._remap_image_ckpt(raw)
    assert out == {"model.vision.0.weight": "w", "other.bias": "b"}


def test_remap_strips_detector_model_prefix():
    raw = {"detector_model.backbone.w": 1}
    out = model_manager._remap_image_ckpt(raw)
    assert out == {"backbone.w": 1}


def test_remap_drops_tracker_keys():
    raw = {
        "detector.keep": 1,
        "tracker.drop": 2,
        "tracker_model.drop": 3,
        "tracker_neck.drop": 4,
    }
    out = model_manager._remap_image_ckpt(raw)
    assert out == {"keep": 1}


def test_remap_drops_unknown_top_level_keys():
    raw = {"detector.ok": 1, "something_else": 2, "bert.whatever": 3}
    out = model_manager._remap_image_ckpt(raw)
    assert out == {"ok": 1}


def test_remap_unwraps_model_dict():
    raw = {"model": {"detector.inner": 7}}
    out = model_manager._remap_image_ckpt(raw)
    assert out == {"inner": 7}


# ---------------------------------------------------------------- listing


def test_list_checkpoints_env_and_folder(patch_dir, tmp_path, monkeypatch):
    folder = _make_models_dir(tmp_path, ["b.pth", "a.safetensors", "notes.txt"])
    env_file = tmp_path / "elsewhere.pt"
    env_file.write_bytes(b"x")
    monkeypatch.setenv("SAM3_CHECKPOINT", str(env_file))
    patch_dir(folder)

    names = model_manager.list_checkpoints()
    # official files first (in preference order), then the rest sorted
    # (folder iteration is alphabetical: a.safetensors < b.pth); non-checkpoint
    # extensions are skipped. Env files share the same official-rank logic.
    assert names == ["a.safetensors", "b.pth", "elsewhere.pt"]


def test_list_checkpoints_official_names_first(patch_dir, tmp_path):
    folder = _make_models_dir(tmp_path, ["zoo.pt", "sam3.pt", "sam3.1_multiplex.pt", "aaa.safetensors"])
    patch_dir(folder)
    names = model_manager.list_checkpoints()
    # preference order inside the official group: sam3.pt < sam3.1_multiplex.pt
    assert names[:2] == ["sam3.pt", "sam3.1_multiplex.pt"]
    assert names[2:] == ["aaa.safetensors", "zoo.pt"]


def test_list_checkpoints_missing_dir(patch_dir, tmp_path):
    patch_dir(tmp_path / "nonexistent")
    assert model_manager.list_checkpoints() == []


def test_list_checkpoints_dedupes_env_file_inside_folder(patch_dir, tmp_path, monkeypatch):
    folder = _make_models_dir(tmp_path, ["sam3.pt"])
    monkeypatch.setenv("SAM3_CHECKPOINT", str(folder / "sam3.pt"))
    patch_dir(folder)
    assert model_manager.list_checkpoints() == ["sam3.pt"]


# ---------------------------------------------------------------- preferred


def test_preferred_checkpoint_name_prefers_official():
    names = ["custom.pt", "sam3.pt"]
    assert model_manager.preferred_checkpoint_name(names) == "sam3.pt"


def test_preferred_checkpoint_name_prefers_pt_over_safetensors():
    assert model_manager.preferred_checkpoint_name(["a.safetensors", "b.pth"]) == "b.pth"


def test_preferred_checkpoint_name_falls_back_to_first():
    assert model_manager.preferred_checkpoint_name(["a.safetensors"]) == "a.safetensors"
    assert model_manager.preferred_checkpoint_name([]) is None


def test_preferred_checkpoint_name_defaults_to_listing(patch_dir, tmp_path):
    folder = _make_models_dir(tmp_path, ["sam3.pt"])
    patch_dir(folder)
    assert model_manager.preferred_checkpoint_name() == "sam3.pt"


# ---------------------------------------------------------------- warnings


def test_checkpoint_warning_official_is_empty():
    assert model_manager.checkpoint_warning("sam3.pt") == ""


def test_checkpoint_warning_safetensors_warns():
    msg = model_manager.checkpoint_warning("custom.safetensors")
    assert "WARNING" in msg
    assert "sam3.pt" in msg


def test_checkpoint_warning_unknown_pt_notes():
    assert model_manager.checkpoint_warning("custom.pt").startswith("NOTE")


def test_checkpoint_warning_empty_name():
    assert model_manager.checkpoint_warning(None) == ""
    assert model_manager.checkpoint_warning("") == ""


# ---------------------------------------------------------------- resolve


def test_resolve_ckpt_path_from_folder(patch_dir, tmp_path):
    folder = _make_models_dir(tmp_path, ["sam3.pt"])
    patch_dir(folder)
    assert model_manager.resolve_ckpt_path("sam3.pt") == folder / "sam3.pt"


def test_resolve_ckpt_path_via_env_name(patch_dir, tmp_path, monkeypatch):
    folder = _make_models_dir(tmp_path, [])
    env_file = tmp_path / "other" / "sam3.pt"
    env_file.parent.mkdir(parents=True)
    env_file.write_bytes(b"x")
    monkeypatch.setenv("SAM3_CHECKPOINT", str(env_file))
    patch_dir(folder)
    assert model_manager.resolve_ckpt_path("sam3.pt") == env_file


def test_resolve_ckpt_path_absolute_name(patch_dir, tmp_path):
    folder = _make_models_dir(tmp_path, [])
    patch_dir(folder)
    direct = tmp_path / "anywhere.pt"
    direct.write_bytes(b"x")
    assert model_manager.resolve_ckpt_path(str(direct)) == direct


def test_resolve_ckpt_path_no_name_uses_env_then_preferred(patch_dir, tmp_path, monkeypatch):
    folder = _make_models_dir(tmp_path, ["sam3.pt"])
    patch_dir(folder)

    # no env: preferred checkpoint from the folder
    assert model_manager.resolve_ckpt_path(None) == folder / "sam3.pt"

    env_file = tmp_path / "env.pt"
    env_file.write_bytes(b"x")
    monkeypatch.setenv("SAM3_CHECKPOINT", str(env_file))
    assert model_manager.resolve_ckpt_path(None) == env_file


def test_resolve_ckpt_path_unknown_name_returns_none(patch_dir, tmp_path):
    folder = _make_models_dir(tmp_path, ["sam3.pt"])
    patch_dir(folder)
    assert model_manager.resolve_ckpt_path("nope.pt") is None


# ---------------------------------------------------------------- state/busy


def test_is_loaded_requires_model_and_processor():
    model_manager._STATE["model"] = object()
    model_manager._STATE["processor"] = None
    assert model_manager.is_loaded() is False
    model_manager._STATE["processor"] = object()
    assert model_manager.is_loaded() is True


def test_unload_clears_state_and_frees_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(vram, "free_cache", lambda: calls.append(1))
    model_manager._STATE["model"] = object()
    model_manager._STATE["processor"] = object()
    model_manager._STATE["ckpt_path"] = "/tmp/x.pt"

    ok, msg = model_manager.unload()

    assert ok and "unloaded" in msg
    assert model_manager.is_loaded() is False
    assert model_manager.current_ckpt() is None
    assert calls == [1]


def test_unload_refused_while_inference_running(monkeypatch):
    calls = []
    monkeypatch.setattr(vram, "free_cache", lambda: calls.append(1))
    model_manager._STATE["model"] = object()
    model_manager._STATE["processor"] = object()

    with model_manager.inference_running():
        ok, msg = model_manager.unload()
        assert not ok
        assert "busy" in msg.lower()
        # model is untouched
        assert model_manager.is_loaded() is True

    assert calls == []


def test_load_refused_while_inference_running(monkeypatch):
    monkeypatch.setattr(vram, "cuda_available", lambda: True)
    with model_manager.inference_running():
        ok, msg = model_manager.load("sam3.pt")
        assert not ok
        assert "busy" in msg.lower()


def test_inference_running_rejects_concurrent_generate():
    with model_manager.inference_running():
        with pytest.raises(RuntimeError, match="already running"):
            with model_manager.inference_running():
                pass


def test_inference_running_clears_busy_after_exception():
    with pytest.raises(ValueError):
        with model_manager.inference_running():
            raise ValueError("boom")
    assert model_manager._STATE["busy"] is False


def test_load_skips_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(vram, "cuda_available", lambda: False)
    ok, msg = model_manager.load("sam3.pt")
    assert not ok
    assert "CUDA" in msg


def test_load_no_checkpoint_message(patch_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(vram, "cuda_available", lambda: True)
    patch_dir(tmp_path / "empty")
    ok, msg = model_manager.load("missing.pt")
    assert not ok
    assert "No checkpoint" in msg


def test_env_sam3_checkpoint_path_resolution(tmp_path, monkeypatch):
    env_file = tmp_path / "custom" / "my_sam3.pt"
    env_file.parent.mkdir(parents=True)
    env_file.write_bytes(b"x")
    monkeypatch.setenv("SAM3_CHECKPOINT", str(env_file))
    names = model_manager.list_checkpoints()
    assert names == ["my_sam3.pt"]
    assert model_manager.resolve_ckpt_path(None) == env_file


def test_missing_checkpoint_hint_mentions_env_and_hf():
    hint = model_manager.missing_checkpoint_hint()
    assert "SAM3_CHECKPOINT" in hint
    assert "huggingface" in hint.lower()
