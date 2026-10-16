import numpy as np
import pytest
from PIL import Image

from sam3_anime import infer


def quadrant_masks():
    """Four 32x32 masks, each filling one quadrant."""
    masks = []
    for i in range(4):
        m = np.zeros((32, 32), dtype=np.uint8)
        r, c = divmod(i, 2)
        m[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] = 255
        masks.append(m)
    return masks


def test_state_to_mask_no_scores_notes_and_uses_all():
    arr = np.stack(quadrant_masks()).astype(np.float32)
    mask, note = infer._state_to_mask({"masks": arr}, 0.5)
    assert mask is not None
    assert note is not None and "スコアがありません" in note
    expected = np.bitwise_or.reduce(quadrant_masks())
    assert np.array_equal(np.array(mask), expected)


def test_state_to_mask_count_mismatch_notes():
    arr = np.stack(quadrant_masks()).astype(np.float32)
    mask, note = infer._state_to_mask({"masks": arr, "scores": np.array([0.1, 0.2])}, 0.5)
    assert mask is not None
    assert note is not None and "不一致" in note


def test_state_to_mask_threshold_excluding_all_falls_back_with_note():
    arr = np.stack(quadrant_masks()).astype(np.float32)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    mask, note = infer._state_to_mask({"masks": arr, "scores": scores}, 0.5)
    assert mask is not None
    assert note is not None and "全マスクを使用" in note
    expected = np.bitwise_or.reduce(quadrant_masks())
    assert np.array_equal(np.array(mask), expected)


def test_state_to_mask_filter_applies_without_note():
    arr = np.stack(quadrant_masks()).astype(np.float32)
    scores = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
    mask, note = infer._state_to_mask({"masks": arr, "scores": scores}, 0.5)
    assert mask is not None
    assert note is None  # filter worked; nothing to warn about
    masks = quadrant_masks()
    expected = np.bitwise_or(np.array(masks[0]), np.array(masks[2]))
    assert np.array_equal(np.array(mask), expected)


def test_state_to_mask_boundary_score_is_kept():
    arr = np.stack(quadrant_masks()).astype(np.float32)
    scores = np.array([0.5, 0.4999, 0.4999, 0.4999], dtype=np.float32)
    mask, note = infer._state_to_mask({"masks": arr, "scores": scores}, 0.5)
    assert note is None
    assert np.array_equal(np.array(mask), np.array(quadrant_masks()[0]))


def test_state_to_mask_empty_state():
    mask, note = infer._state_to_mask(None, 0.5)
    assert mask is None and note is None


def test_generate_masks_requires_loaded_model():
    with pytest.raises(RuntimeError, match="SAM3 not loaded"):
        infer.generate_masks(Image.new("RGB", (16, 16)), [("a", "x")])


def test_fit_image_downscales_only_large():
    img = Image.new("RGB", (2048, 1024))
    fitted = infer._fit_image(img, 1024)
    assert fitted.size == (1024, 512)
    same = infer._fit_image(Image.new("RGB", (512, 256)), 1024)
    assert same.size == (512, 256)


def test_filter_candidates_live_threshold():
    arr = np.stack(quadrant_masks()).astype(np.float32)
    scores = np.array([0.9, 0.2, 0.7, 0.1], dtype=np.float32)
    cand = infer._extract_candidates({"masks": arr, "scores": scores})
    assert cand is not None
    high_masks, notes = infer.filter_candidates({"face": cand}, threshold=0.8, orig_size=(64, 64))
    assert not notes
    assert high_masks["face"].size == (64, 64)
    low_masks, _ = infer.filter_candidates({"face": cand}, threshold=0.6, orig_size=(64, 64))
    assert np.array(low_masks["face"]).sum() > np.array(high_masks["face"]).sum()
