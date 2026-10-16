import base64
import io
import json
from itertools import count

import numpy as np
from PIL import Image

from sam3_anime import forge_export


def test_build_inpaint_upload_payload_is_png_b64():
    img = Image.new("RGB", (8, 8), (255, 0, 0))
    mask = Image.new("L", (8, 8), 255)
    img_b64, mask_b64 = forge_export.build_inpaint_upload_payload(img, mask)
    for b64 in (img_b64, mask_b64):
        raw = base64.b64decode(b64)
        im = Image.open(io.BytesIO(raw))
        assert im.format == "PNG"
    assert Image.open(io.BytesIO(base64.b64decode(img_b64))).mode == "RGB"


def test_build_inpaint_upload_payload_resizes_mask():
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    mask = Image.new("L", (8, 8), 255)
    _img_b64, mask_b64 = forge_export.build_inpaint_upload_payload(img, mask)
    m = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
    assert m.size == (16, 16)


def test_build_inpaint_pair_rgba_mask():
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    mask = Image.new("L", (8, 8), 0)
    mask.putpixel((2, 2), 255)
    init, mask_rgba = forge_export.build_inpaint_pair(img, mask)
    assert init.mode == "RGB"
    assert mask_rgba.mode == "RGBA"
    arr = np.array(mask_rgba)
    # opaque RGBA: RGB carries white=edit, alpha is 255 everywhere so the
    # Inpaint upload preview shows true black/white (no white background blend)
    assert arr[2, 2, 0] == 255 and arr[2, 2, 3] == 255
    assert arr[0, 0, 0] == 0 and arr[0, 0, 3] == 255
    assert (arr[..., 3] == 255).all()


def test_mask_rgba_matches_luminance_reading():
    """Forge's luminance path must read the mask as white=edit region."""
    m = Image.new("L", (4, 4), 0)
    m.putpixel((1, 1), 255)
    rgba = forge_export.build_mask_rgba(m)

    def create_binary_mask(image, round=True):
        # same logic as Forge Neo modules/processing.py
        if image.mode == "RGBA" and image.getextrema()[-1] != (255, 255):
            if round:
                image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
            else:
                image = image.split()[-1].convert("L")
        else:
            image = image.convert("L")
        return image

    luminance = np.array(rgba.convert("L"))
    processed = np.array(create_binary_mask(rgba))
    assert np.array_equal(luminance, processed)
    assert luminance[1, 1] == 255 and luminance[0, 0] == 0


def test_export_to_inpaint_round_trip(tmp_path):
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    mask = Image.new("L", (8, 8), 0)
    mask.putpixel((4, 4), 255)
    ok, msg, payload, png_path = forge_export.export_to_inpaint(img, mask, save_dir=tmp_path)
    assert ok
    assert png_path is not None and png_path.is_file()
    d = json.loads(payload)
    assert d["image"] and d["mask"]
    # payload mask is opaque RGBA (white=edit) — preview must not look all-white
    payload_mask = Image.open(io.BytesIO(base64.b64decode(d["mask"])))
    assert payload_mask.mode == "RGBA"
    arr = np.array(payload_mask)
    assert arr[4, 4, 3] == 255 and arr[4, 4, 0] == 255
    assert arr[0, 0, 0] == 0 and arr[0, 0, 3] == 255


def test_save_combined_mask_timestamped_filename(tmp_path):
    mask = Image.new("L", (4, 4), 255)
    path = forge_export.save_combined_mask(mask, tmp_path)
    # timestamped name prevents overwrite races between concurrent sessions
    assert path.name.startswith("sam3_anime_combined_mask_")
    assert path.name.endswith(".png")


def test_save_combined_mask_timestamped_name_is_unique(tmp_path):
    import time

    from sam3_anime import forge_export as fe

    mask = Image.new("L", (4, 4), 255)
    fe._NAME_COUNTER = count(10_000)  # reset/force distinct sequence numbers
    try:
        paths = {forge_export.save_combined_mask(mask, tmp_path) for _ in range(4)}
    finally:
        fe._NAME_COUNTER = count()
    time.sleep(0.002)
    forge_export.save_combined_mask(mask, tmp_path)
    assert len(paths) == 4


def test_save_and_export_cutout(tmp_path):
    img = Image.new("RGB", (10, 10), (200, 100, 50))
    mask = Image.new("L", (10, 10), 0)
    mask.putpixel((3, 4), 255)
    mask.putpixel((4, 5), 255)

    saved = forge_export.save_cutout(img, mask, tmp_path, crop=True)
    assert saved.is_file()
    assert saved.name.startswith("sam3_anime_cutout_")
    with Image.open(saved) as im:
        assert im.mode == "RGBA"
        assert im.size == (2, 2)
        assert im.getpixel((0, 0)) == (200, 100, 50, 255)
        assert im.getpixel((1, 0)) == (0, 0, 0, 0)

    ok, msg, payload, png_path = forge_export.export_cutout(img, mask, save_dir=tmp_path, crop=True)
    assert ok
    assert png_path is not None and png_path.is_file()
    assert "2×2" in msg
    data = json.loads(payload)
    assert data["filename"] == png_path.name
    decoded = Image.open(io.BytesIO(base64.b64decode(data["image"])))
    assert decoded.mode == "RGBA"
    assert decoded.size == (2, 2)
