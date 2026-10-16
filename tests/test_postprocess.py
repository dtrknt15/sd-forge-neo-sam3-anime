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


def test_feather_zero_is_identity():
    m = make_mask((16, 16), [(4, 4), (4, 5), (5, 4), (5, 5)])
    out = np.array(postprocess.feather_mask(m, 0))
    assert np.array_equal(out, np.array(m))


def test_feather_softens_edge_and_keeps_interior():
    # solid blob: interior must stay 255, edge must gain intermediate values
    blob = np.zeros((32, 32), dtype=np.uint8)
    blob[8:24, 8:24] = 255
    m = Image.fromarray(blob, mode="L")
    out = np.array(postprocess.feather_mask(m, 3))
    assert out[15, 15] == 255  # interior saturated
    edge_band = out[7, 8:24]  # just outside the blob
    assert (edge_band > 0).any() and (edge_band < 255).any()  # graded ramp
    assert out.max() == 255 and out.min() == 0


def test_feather_preserves_size_and_mode():
    m = make_mask((10, 14), [(5, 7)])
    out = postprocess.feather_mask(m, 2)
    assert out.size == m.size and out.mode == "L"


def test_inverted_feathered_mask_is_soft():
    """invert after feather must stay graded (no re-thresholding)."""
    blob = np.zeros((32, 32), dtype=np.uint8)
    blob[8:24, 8:24] = 255
    m = Image.fromarray(blob, mode="L")
    inv = np.array(postprocess.invert_mask(postprocess.feather_mask(m, 3)))
    band = inv[7, 8:24]
    assert (band > 0).any() and (band < 255).any()


def test_combine_masks_blends_soft_masks():
    soft_a = Image.fromarray(np.zeros((4, 4), dtype=np.uint8) + 100, mode="L")
    soft_b = Image.fromarray(np.zeros((4, 4), dtype=np.uint8) + 200, mode="L")
    comb = np.array(postprocess.combine_masks([soft_a, soft_b]))
    assert (comb == 200).all()  # maximum, not OR (which would give 236)


def test_process_pipeline_feather_order_morph_then_feather():
    """Feather applies after morphology: dilated edge is the one that's soft."""
    m = make_mask((32, 32), [(16, 16)])
    processed, _ = postprocess.process_pipeline({"a": m}, dilate=4, feather=3, invert=False)
    out = np.array(processed["a"])
    assert out[16, 16] == 255
    # dilated ring just outside the eroded single pixel must exist but be soft
    ring = out[12:20, 12]
    assert (ring > 0).any() and (ring < 255).any()


def test_make_multi_overlay_uses_concept_colors():
    base = Image.new("RGB", (16, 16), (0, 0, 0))
    face = make_mask((16, 16), [(0, 0)])
    hair = make_mask((16, 16), [(8, 8)])
    ov = postprocess.make_multi_overlay(base, {"face": face, "hair": hair}, alpha=1.0)
    assert ov.size == base.size
    # face palette is (255, 82, 82), hair is (255, 193, 7)
    fr, fg, fb = ov.getpixel((0, 0))
    hr, hg, hb = ov.getpixel((8, 8))
    assert fr > 200 and abs(fg - 82) < 30 and abs(fb - 82) < 30
    assert hr > 200 and abs(hg - 193) < 30 and abs(hb - 7) < 30


def test_make_multi_overlay_empty_returns_base():
    base = Image.new("RGB", (8, 8), (10, 20, 30))
    ov = postprocess.make_multi_overlay(base, {}, alpha=0.45)
    assert ov.getpixel((0, 0)) == (10, 20, 30)


def test_make_multi_overlay_later_concept_wins_overlap():
    base = Image.new("RGB", (4, 4), (0, 0, 0))
    both = make_mask((4, 4), [(1, 1)])
    ov = postprocess.make_multi_overlay(base, {"face": both, "hair": both}, alpha=1.0)
    r, g, b = ov.getpixel((1, 1))
    # hair paints after face, so hair color (255, 193, 7) wins the overlap
    assert r > 200 and abs(g - 193) < 30 and abs(b - 7) < 30


def test_make_multi_overlay_resizes_mask():
    base = Image.new("RGB", (16, 16), (0, 0, 0))
    small = make_mask((8, 8), [(0, 0)])
    ov = postprocess.make_multi_overlay(base, {"face": small}, alpha=1.0)
    assert ov.size == (16, 16)
    assert ov.getpixel((0, 0))[0] > 100


def test_make_cutout_crops_and_sets_alpha():
    base = Image.new("RGB", (16, 16), (10, 20, 30))
    # Draw a 4x6 region at x=3..6, y=5..10
    mask_arr = np.zeros((16, 16), dtype=np.uint8)
    mask_arr[5:11, 3:7] = 255
    mask = Image.fromarray(mask_arr, mode="L")
    # Put a transparent hole inside the bounding box at (x=4, y=6)
    mask.putpixel((4, 6), 0)

    cut = postprocess.make_cutout(base, mask, crop=True)
    assert cut.mode == "RGBA"
    assert cut.size == (4, 6)
    # Top-left of cropped box corresponds to (3, 5) -> opaque (10, 20, 30, 255)
    assert cut.getpixel((0, 0)) == (10, 20, 30, 255)
    # Hole at (4, 6) -> local (1, 1) -> transparent (0, 0, 0, 0)
    assert cut.getpixel((1, 1)) == (0, 0, 0, 0)


def test_make_cutout_preserves_feathered_alpha_and_no_crop():
    base = Image.new("RGB", (8, 8), (100, 150, 200))
    mask_arr = np.zeros((8, 8), dtype=np.uint8)
    mask_arr[2, 2] = 128
    mask = Image.fromarray(mask_arr, mode="L")

    cut = postprocess.make_cutout(base, mask, crop=False)
    assert cut.size == (8, 8)
    assert cut.getpixel((2, 2)) == (100, 150, 200, 128)
    assert cut.getpixel((0, 0)) == (0, 0, 0, 0)


def test_make_cutout_empty_mask_does_not_crash():
    base = Image.new("RGB", (8, 8), (50, 60, 70))
    empty = Image.new("L", (8, 8), 0)
    cut = postprocess.make_cutout(base, empty, crop=True)
    assert cut.mode == "RGBA"
    assert cut.size == (8, 8)
    assert (np.array(cut)[..., 3] == 0).all()
