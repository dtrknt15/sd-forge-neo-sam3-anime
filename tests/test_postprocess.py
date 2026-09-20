import numpy as np
import pytest
from PIL import Image

from sam3_anime import postprocess


def make_mask(size, on_pixels):
    m = Image.new("L", size, 0)
    for xy in on_pixels:
        m.putpixel(xy, 255)
    return m


def test_combine_masks_is_or():
    a = make_mask((8, 8), [(0, 0)])
    b = make_mask((8, 8), [(7, 7)])
    c = np.array(postprocess.combine_masks([a, b]))
    assert c[0, 0] == 255 and c[7, 7] == 255
    assert c.sum() == 2 * 255


def test_combine_masks_empty_raises():
    with pytest.raises(ValueError):
        postprocess.combine_masks([])


def test_dilate_expands_erode_shrinks():
    m = make_mask((16, 16), [(8, 8), (8, 9)])
    dil = np.array(postprocess.apply_dilate_erode(m, 2))
    ero = np.array(postprocess.apply_dilate_erode(m, -2))
    assert dil.sum() > np.array(m).sum()
    assert 0 <= ero.sum() < np.array(m).sum()
    # dilate then erode on a solid blob is roughly identity
    blob = np.zeros((32, 32), dtype=np.uint8)
    blob[12:20, 12:20] = 255
    blob_img = Image.fromarray(blob, mode="L")
    round_trip = np.array(postprocess.apply_dilate_erode(postprocess.apply_dilate_erode(blob_img, 2), -2))
    assert (round_trip == blob).mean() > 0.9


def test_zero_amount_is_identity():
    m = make_mask((8, 8), [(3, 3)])
    assert np.array_equal(np.array(postprocess.apply_dilate_erode(m, 0)), np.array(m))


def test_invert_mask():
    m = make_mask((4, 4), [(0, 0)])
    inv = np.array(postprocess.invert_mask(m))
    assert inv[0, 0] == 0
    assert inv[3, 3] == 255
    assert inv.sum() == 15 * 255


def test_to_inpaint_mask_returns_l():
    rgb = Image.new("RGB", (4, 4), (255, 255, 255))
    out = postprocess.to_inpaint_mask(rgb)
    assert out.mode == "L"


def test_make_overlay_size_and_alpha():
    base = Image.new("RGB", (16, 16), (10, 20, 30))
    mask = make_mask((16, 16), [(0, 0)])
    ov = postprocess.make_overlay(base, mask, "face", alpha=0.45)
    assert ov.size == base.size
    # masked pixel is tinted toward the face color (255, 82, 82)
    r, g, b = ov.getpixel((0, 0))
    assert r > 10 and b > 30


def test_make_overlay_resizes_mask_to_base():
    base = Image.new("RGB", (16, 16), (0, 0, 0))
    mask = make_mask((8, 8), [(4, 4)])
    ov = postprocess.make_overlay(base, mask, "hair")
    assert ov.size == base.size


def test_process_pipeline_morph_then_invert_and_combined():
    a = make_mask((8, 8), [(0, 0)])
    b = make_mask((8, 8), [(7, 7)])
    processed, combined = postprocess.process_pipeline({"a": a, "b": b}, dilate=0, invert=True)
    assert set(processed) == {"a", "b"}
    assert np.array(processed["a"])[0, 0] == 0  # inverted
    assert np.array(processed["a"])[7, 7] == 255
    comb = np.array(combined)
    # inverted masks are all-white except one pixel each; OR covers every pixel
    assert comb.sum() == 64 * 255
    assert comb[0, 0] == 255 and comb[7, 7] == 255  # each covered by the other mask


def test_process_pipeline_empty():
    processed, combined = postprocess.process_pipeline({}, dilate=0, invert=False)
    assert processed == {} and combined is None
