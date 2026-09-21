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
import time
from itertools import count
from pathlib import Path

from PIL import Image

from .postprocess import to_inpaint_mask


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_NAME_COUNTER = count()


def _timestamped_mask_name() -> str:
    """Timestamped + per-process sequence so concurrent WebUI sessions
    (e.g. --share / multi-tab, same process) never overwrite each other's
    exported mask — even when called within the same clock tick."""
    now = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    ms = int(now * 1000) % 1000
    seq = next(_NAME_COUNTER) % 1000
    return f"sam3_anime_combined_mask_{stamp}_{ms:03d}_{seq:03d}.png"


def save_combined_mask(mask: Image.Image, directory: str | Path | None = None) -> Path:
    directory = Path(directory) if directory else Path(tempfile.gettempdir()) / "sam3_anime_masks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _timestamped_mask_name()
    to_inpaint_mask(mask).save(path, format="PNG")
    return path


def build_mask_rgba(mask: Image.Image) -> Image.Image:
    """Return the mask as a fully OPAQUE RGBA image (white = edit region).

    Opaque is intentional:
    - Forge Neo create_binary_mask reads the LUMINANCE when the alpha channel
      has no transparency, so RGB carries the mask reliably.
    - Transparent pixels render as the UI background (white) in the Inpaint
      upload Mask preview, which made the whole mask look all-white.
    """
    m = to_inpaint_mask(mask)
    return Image.merge("RGBA", (m, m, m, Image.new("L", m.size, 255)))


def build_inpaint_upload_payload(image: Image.Image, mask: Image.Image) -> tuple[str, str]:
    """Return (image_b64, mask_b64). Mask is opaque RGBA (white = edit)."""
    img = image.convert("RGB")
    m = to_inpaint_mask(mask)
    if m.size != img.size:
        m = m.resize(img.size, Image.NEAREST)
    return _pil_to_b64_png(img), _pil_to_b64_png(build_mask_rgba(m))


def build_inpaint_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Return (init RGB, mask RGBA) ready for img_inpaint_base / img_inpaint_mask.

    The mask is fully opaque so the Inpaint upload preview shows proper
    black/white and Forge reads it via luminance (white = edit region).
    """
    img = image.convert("RGB")
    m = to_inpaint_mask(mask)
    if m.size != img.size:
        m = m.resize(img.size, Image.NEAREST)
    return img, build_mask_rgba(m)


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
