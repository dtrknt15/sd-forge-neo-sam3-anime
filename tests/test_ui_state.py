import pytest

from sam3_anime import ui_state


class TestThirdPartyAndInitSource:
    def test_controlnet_ids_are_third_party(self):
        assert ui_state.is_third_party_image_source("controlnet_img2img_0")
        assert ui_state.is_third_party_image_source("foo", "ControlNet Unit")
        assert ui_state.is_third_party_image_source("ip_adapter_block")

    def test_plain_img2img_is_not_third_party(self):
        assert not ui_state.is_third_party_image_source("img2img_image", "tab_img2img")

    def test_accepts_img2img_canvas(self):
        assert ui_state.accepts_as_img2img_init(
            "logical_image_background",
            ["tab_img2img", "mode_img2img"],
            class_blob="logical_image_background",
        )

    def test_accepts_img_inpaint_base_even_without_parent_id(self):
        assert ui_state.accepts_as_img2img_init("img_inpaint_base", [])

    def test_rejects_controlnet_under_img2img(self):
        assert not ui_state.accepts_as_img2img_init(
            "cn_unit_0_image",
            ["tab_img2img", "controlnet"],
            class_blob="logical_image_background",
        )

    def test_rejects_txt2img_canvas(self):
        assert not ui_state.accepts_as_img2img_init(
            "txt2img_image",
            ["tab_txt2img"],
        )

    def test_rejects_unknown_tree(self):
        # Unknown third-party / orphan canvases must not be auto-used.
        assert not ui_state.accepts_as_img2img_init(
            "mystery_canvas",
            ["some_extension_panel"],
        )

    def test_rejects_controlnet_label_without_elem_id(self):
        assert not ui_state.accepts_as_img2img_init(
            "img2img_sketch",
            ["tab_img2img", "ControlNet"],
            class_blob="logical_image_background",
        )


class TestQuickSelectActive:
    AVAILABLE = ["face", "hair", "extra", "eyes", "background"]

    def test_all(self):
        assert ui_state.quick_select_active("all", self.AVAILABLE) == self.AVAILABLE

    def test_none(self):
        assert ui_state.quick_select_active("none", self.AVAILABLE, ["face"]) == []

    def test_face_focus_keeps_available_order(self):
        assert ui_state.quick_select_active("face", self.AVAILABLE) == ["face", "hair", "eyes"]

    def test_person_excludes_sky_background_extra(self):
        assert ui_state.quick_select_active("person", self.AVAILABLE) == ["face", "hair", "eyes"]

    def test_invert(self):
        assert ui_state.quick_select_active("invert", self.AVAILABLE, ["face", "eyes"]) == [
            "hair",
            "extra",
            "background",
        ]

    def test_empty_available(self):
        assert ui_state.quick_select_active("all", []) == []

    def test_unknown_mode_returns_all(self):
        assert ui_state.quick_select_active("nope", self.AVAILABLE) == self.AVAILABLE


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

    def test_no_image_message_mentions_controlnet_excluded(self):
        with pytest.raises(ValueError, match="ControlNet"):
            ui_state.resolve_generation_inputs(None, None, ["face"], "")

    def test_no_prompts_raises(self):
        with pytest.raises(ValueError, match="プリセット"):
            ui_state.resolve_generation_inputs(object(), None, [], "")

    def test_selected_extra_without_free_text_is_skipped(self):
        # extra alone with no free text → no prompts at all → ValueError
        with pytest.raises(ValueError, match="プリセット"):
            ui_state.resolve_generation_inputs(object(), None, ["extra"], "")


class TestGalleryOverlayHelpers:
    STORE = {"face": "m", "hair": "m", "eyes": "m", "extra": "m"}

    def test_gallery_visible_ids_follows_active_order(self):
        assert ui_state.gallery_visible_ids(self.STORE, ["hair", "face"]) == ["hair", "face"]

    def test_gallery_visible_ids_skips_missing(self):
        assert ui_state.gallery_visible_ids(self.STORE, ["face", "nope"]) == ["face"]

    def test_gallery_visible_ids_empty_store(self):
        assert ui_state.gallery_visible_ids({}, ["face"]) == []
        assert ui_state.gallery_visible_ids(None, ["face"]) == []

    def test_gallery_flat_index_int(self):
        assert ui_state.gallery_flat_index(2) == 2

    def test_gallery_flat_index_row_col(self):
        assert ui_state.gallery_flat_index((1, 2), columns=5) == 7

    def test_gallery_index_at_maps_active_order(self):
        active = ["hair", "face", "extra"]
        assert ui_state.gallery_index_at(0, self.STORE, active) == "hair"
        assert ui_state.gallery_index_at(1, self.STORE, active) == "face"
        assert ui_state.gallery_index_at((0, 2), self.STORE, active) == "extra"

    def test_gallery_index_at_out_of_range(self):
        assert ui_state.gallery_index_at(9, self.STORE, ["face"]) is None
        assert ui_state.gallery_index_at(-1, self.STORE, ["face"]) is None
        assert ui_state.gallery_index_at("x", self.STORE, ["face"]) is None

    def test_overlay_label_modes(self):
        assert ui_state.overlay_label("multi") == "オーバーレイ（色分け）"
        assert ui_state.overlay_label("single", "face") == "オーバーレイ — 顔 (face)"
        assert ui_state.overlay_label("single", "extra", {"extra": "cat ears"}) == "オーバーレイ — cat ears (extra)"

    def test_build_color_legend_html(self):
        html = ui_state.build_color_legend_html(["face", "hair"])
        assert "#ff5252" in html  # face red
        assert "#ffc107" in html  # hair amber
        assert "顔" in html and "髪" in html

    def test_build_color_legend_html_empty(self):
        assert ui_state.build_color_legend_html([]) == ""


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
