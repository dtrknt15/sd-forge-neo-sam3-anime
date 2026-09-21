import pytest

from sam3_anime import ui_state


class TestResolveGenerationInputs:
    def test_returns_prompts_and_labels(self):
        img, prompts, labels = ui_state.resolve_generation_inputs(object(), None, ["face"], "cat ears")
        assert img is not None
        assert prompts == [("face", "anime face"), ("extra", "cat ears")]
        assert labels == {"face": "顔", "extra": "cat ears"}

    def test_canvas_sentinel_when_image_missing(self):
        img, _prompts, _labels = ui_state.resolve_generation_inputs(None, "data:image/png;base64,AAAA", ["face"], "")
        assert img == "CANVAS"

    def test_no_image_raises(self):
        with pytest.raises(ValueError, match="No image"):
            ui_state.resolve_generation_inputs(None, None, ["face"], "")

    def test_empty_canvas_string_is_ignored(self):
        with pytest.raises(ValueError, match="No image"):
            ui_state.resolve_generation_inputs(None, "", ["face"], "")

    def test_no_prompts_raises(self):
        with pytest.raises(ValueError, match="プリセット"):
            ui_state.resolve_generation_inputs(object(), None, [], "")

    def test_selected_extra_without_free_text_is_skipped(self):
        # extra alone with no free text → no prompts at all → ValueError
        with pytest.raises(ValueError, match="プリセット"):
            ui_state.resolve_generation_inputs(object(), None, ["extra"], "")


class TestBuildActiveChoices:
    def test_choices_use_labels_and_ids(self):
        choices = ui_state.build_active_choices({"face": "m", "extra": "m"}, {"face": "顔", "extra": "cat ears"})
        assert choices == [("顔 (face)", "face"), ("cat ears (extra)", "extra")]

    def test_choices_fall_back_to_constants_label(self):
        choices = ui_state.build_active_choices({"hair": "m"})
        assert choices == [("髪 (hair)", "hair")]

    def test_unknown_id_falls_back_to_id(self):
        choices = ui_state.build_active_choices({"weird": "m"})
        assert choices == [("weird (weird)", "weird")]


class TestBuildStatusMessage:
    def test_basic(self):
        assert ui_state.build_status_message(2) == "Generated 2 mask(s)."

    def test_with_notes_and_unload(self):
        msg = ui_state.build_status_message(1, notes=["img2img canvas を使用"], auto_unload="SAM3 unloaded (settings)")
        assert msg == ("Generated 1 mask(s). img2img canvas を使用 | SAM3 unloaded (settings)")


class TestBuildExportNote:
    def test_success(self):
        note = ui_state.build_export_note("tmp/mask.png")
        assert note == "Exported to **img2img → Inpaint upload**. マスク PNG: `tmp/mask.png`"

    def test_fallback_when_components_missing(self):
        note = ui_state.build_export_note("p.png", image_set=False)
        assert "手動" in note and "`p.png`" in note

    def test_set_error(self):
        note = ui_state.build_export_note("p.png", set_error="boom")
        assert "画像セットに失敗" in note and "boom" in note

    def test_missing_png_path(self):
        note = ui_state.build_export_note(None)
        assert note.endswith("マスク PNG: -")


class TestButtonInteractivity:
    def test_running_disables_all_work_buttons(self):
        st = ui_state.button_interactivity(running=True)
        assert st == {"generate": False, "load": False, "unload": False, "free": False}

    def test_idle_enables_all(self):
        st = ui_state.button_interactivity(running=False)
        assert st == {"generate": True, "load": True, "unload": True, "free": True}

    def test_cancel_is_never_managed(self):
        # Cancel must stay clickable during a run: the script never disables it.
        assert "cancel" not in ui_state.button_interactivity(running=True)
