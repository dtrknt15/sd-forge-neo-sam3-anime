"""Export combined mask + init image into Forge Neo img2img inpaint.

Strategy (spec §7):
1. Prefer known component IDs: img_inpaint_base / img_inpaint_mask (Inpaint upload)
2. Fall back to JS injection into those components
3. Always save PNG and report manual guidance on failure

Do not invent private APIs. Heavy try/except by design.
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

from PIL import Image

from .postprocess import to_inpaint_mask


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_combined_mask(mask: Image.Image, directory: str | Path | None = None) -> Path:
    directory = Path(directory) if directory else Path(tempfile.gettempdir()) / "sam3_anime_masks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sam3_anime_combined_mask.png"
    to_inpaint_mask(mask).save(path, format="PNG")
    return path


def build_inpaint_upload_payload(image: Image.Image, mask: Image.Image) -> tuple[str, str]:
    """Return (image_b64, mask_b64). Mask is grayscale white=fill."""
    img = image.convert("RGB")
    m = to_inpaint_mask(mask)
    if m.size != img.size:
        m = m.resize(img.size, Image.NEAREST)
    return _pil_to_b64_png(img), _pil_to_b64_png(m)


def build_inpaint_pair(
    image: Image.Image, mask: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """Return (init RGB, mask L) ready for img_inpaint_base / img_inpaint_mask.

    Inpaint upload mask component is RGBA; white = inpaint region.
    """
    img = image.convert("RGB")
    m = to_inpaint_mask(mask)
    if m.size != img.size:
        m = m.resize(img.size, Image.NEAREST)
    # WebUI inpaint upload expects RGBA mask; keep white on transparent black
    mask_rgba = Image.merge("RGBA", (m, m, m, Image.new("L", m.size, 255)))
    return img, mask_rgba


def export_to_inpaint(
    image: Image.Image,
    mask: Image.Image,
    *,
    save_dir: str | Path | None = None,
) -> tuple[bool, str, str, Path | None]:
    """Save PNG + build JS payload.

    Returns (prepared_ok, status_message, payload_json, png_path).
    """
    try:
        png_path = save_combined_mask(mask, save_dir)
    except Exception as e:
        return False, f"Export failed: could not save mask PNG ({e})", "", None

    try:
        img_b64, mask_b64 = build_inpaint_upload_payload(image, mask)
    except Exception as e:
        return False, f"Export failed: {e}. マスク PNG は保存済み: {png_path}", "", png_path

    import json

    payload = json.dumps({"image": img_b64, "mask": mask_b64})
    msg = f"Export prepared. PNG: {png_path}"
    return True, msg, payload, png_path
