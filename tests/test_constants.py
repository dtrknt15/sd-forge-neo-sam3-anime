from sam3_anime import constants


def test_preset_ids_unique_and_complete():
    ids = [p[0] for p in constants.PRESETS]
    assert len(ids) == len(set(ids))
    assert "face" in ids
    assert "extra" in ids


def test_prompt_for_id():
    assert constants.prompt_for_id("face") == "anime face"
    assert constants.prompt_for_id("extra") == ""
    assert constants.prompt_for_id("does_not_exist") == ""


def test_label_for_id_fallback():
    assert constants.label_for_id("face") == "顔"
    # unknown id with no labels dict falls back to the id itself
    assert constants.label_for_id("unknown_id") == "unknown_id"
    # labels dict takes precedence
    assert constants.label_for_id("face", {"face": "カスタム"}) == "カスタム"


def test_resolve_prompts_presets_only():
    rows = constants.resolve_prompts(["face", "hair"], "")
    assert [(cid, prompt) for cid, prompt, _label in rows] == [
        ("face", "anime face"),
        ("hair", "anime hair"),
    ]


def test_resolve_prompts_skips_extra_without_free_text():
    rows = constants.resolve_prompts(["face", "extra"], "")
    assert [cid for cid, _p, _l in rows] == ["face"]


def test_resolve_prompts_free_text_single():
    rows = constants.resolve_prompts([], "cat ears")
    assert [(cid, prompt) for cid, prompt, _l in rows] == [("extra", "cat ears")]


def test_resolve_prompts_free_text_comma_split():
    rows = constants.resolve_prompts([], "cat ears, ribbon")
    assert [(cid, prompt) for cid, prompt, _l in rows] == [
        ("extra", "cat ears"),
        ("extra_1", "ribbon"),
    ]


def test_resolve_prompts_japanese_comma_and_blanks():
    rows = constants.resolve_prompts([], "cat ears、 ribbon , ,sword")
    assert [(cid, prompt) for cid, prompt, _l in rows] == [
        ("extra", "cat ears"),
        ("extra_1", "ribbon"),
        ("extra_2", "sword"),
    ]


def test_resolve_prompts_free_text_overrides_extra_checkbox():
    # free text is always applied, extra checkbox is optional
    rows = constants.resolve_prompts(["face"], "sword")
    assert [(cid, prompt) for cid, prompt, _l in rows] == [
        ("face", "anime face"),
        ("extra", "sword"),
    ]
