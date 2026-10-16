"""Postprocess masks: dilate/erode, feather, invert, overlay."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image

from .constants import OVERLAY_PALETTE


def _to_gray(mask: Image.Image | np.ndarray) -> np.ndarray:
    """Raw grayscale uint8 — keeps feather gradients intact."""
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L"))
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
    return arr.astype(np.uint8)


def _to_np(mask: Image.Image | np.ndarray) -> np.ndarray:
    return (_to_gray(mask) > 127).astype(np.uint8) * 255


def _morph_cv2(mask_u8: np.ndarray, amount: int) -> np.ndarray | None:
    """OpenCV morph (fast). amount > 0 dilate, amount < 0 erode."""
    k = abs(int(amount))
    if k == 0:
        return mask_u8
    try:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        op = cv2.dilate if amount > 0 else cv2.erode
        return op(mask_u8, kernel, iterations=1)
    except Exception:
        return None


def _morph_pil(mask_u8: np.ndarray, amount: int) -> np.ndarray:
    """amount > 0 dilate, amount < 0 erode. Elliptical kernel, iterations=|amount|."""
    if amount == 0:
        return mask_u8
    fast = _morph_cv2(mask_u8, amount)
    if fast is not None:
        return fast
    img = Image.fromarray(mask_u8, mode="L")
    iterations = abs(int(amount))
    from PIL import ImageFilter

    op = ImageFilter.MaxFilter if amount > 0 else ImageFilter.MinFilter
    size = 3
    out = img
    for _ in range(iterations):
        out = out.filter(op(size))
    return np.array(out, dtype=np.uint8)


def apply_dilate_erode(mask: Image.Image | np.ndarray, amount: int) -> Image.Image:
    arr = _to_gray(mask)  # grayscale morphology: identical for binary masks
    arr = _morph_pil(arr, int(amount))
    return Image.fromarray(arr, mode="L")


def feather_mask(mask: Image.Image | np.ndarray, radius: int) -> Image.Image:
    """Blur the mask edge so inpaint boundaries blend smoothly (no hard step).

    WebUI's inpaint samples mask values, so a soft 0..255 ramp at the edge
    avoids visible paint seams that a binary NEAREST-resized mask causes.
    radius <= 0 returns the mask unchanged. Interior and far background stay
    saturated at 0/255; only the transition band is graded.
    """
    r = int(radius)
    if r <= 0:
        return Image.fromarray(_to_gray(mask), mode="L")
    arr = _to_gray(mask)
    img = Image.fromarray(arr, mode="L")
    try:
        import cv2

        k = 2 * r + 1
        blurred = cv2.GaussianBlur(img, (k, k), sigmaX=0)
    except Exception:
        # PIL fallback: match cv2's sigma-from-ksize formula so both paths
        # blur identically (PIL's radius parameter IS the sigma).
        from PIL import ImageFilter

        sigma = 0.3 * ((2 * r + 1) - 1) * 0.5 - 0.3 + 0.8  # cv2: 0.3*((ksize-1)*0.5 - 1) + 0.8
        blurred = img.filter(ImageFilter.GaussianBlur(radius=max(sigma, 0.3)))
    return Image.fromarray(np.array(blurred, dtype=np.uint8), mode="L")


def _to_np_mask_image(mask: Image.Image | np.ndarray) -> Image.Image:
    """Normalize to an L-mode PIL image without thresholding."""
    return Image.fromarray(_to_gray(mask), mode="L")


def invert_mask(mask: Image.Image) -> Image.Image:
    # 255 - gray, not thresholded, so a feathered mask inverts to a soft mask
    arr = _to_gray(mask)
    return Image.fromarray(255 - arr, mode="L")


def combine_masks(masks: Iterable[Image.Image]) -> Image.Image:
    """Union of masks. np.maximum (not bitwise OR) so feathered (soft) masks
    blend correctly; identical to OR for binary 0/255 masks."""
    masks = list(masks)
    if not masks:
        raise ValueError("no masks to combine")
    acc = _to_gray(masks[0])
    for m in masks[1:]:
        acc = np.maximum(acc, _to_gray(m))
    return Image.fromarray(acc, mode="L")


def to_inpaint_mask(mask: Image.Image) -> Image.Image:
    """White = inpaint region (WebUI convention). Return L or RGB."""
    return mask.convert("L")


def make_overlay(image: Image.Image, mask: Image.Image, concept_id: str = "extra", alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    color = OVERLAY_PALETTE.get(concept_id, OVERLAY_PALETTE["extra"])
    arr = _to_np(mask)
    h, w = arr.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = (arr > 127).astype(np.uint8) * int(alpha * 255)
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    if overlay_img.size != base.size:
        overlay_img = overlay_img.resize(base.size, Image.NEAREST)
    return Image.alpha_composite(base, overlay_img).convert("RGB")


def make_multi_overlay(
    image: Image.Image,
    masks: dict[str, Image.Image],
    alpha: float = 0.45,
) -> Image.Image:
    """Composite each concept mask in its own palette color.

    Later concepts paint over earlier ones in overlap so every region stays
    identifiable. Soft (feathered) masks are used as continuous alpha so
    edges still blend. Empty `masks` returns the base RGB image.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    out = base.copy()
    for cid, mask in masks.items():
        color = np.array(OVERLAY_PALETTE.get(cid, OVERLAY_PALETTE["extra"]), dtype=np.float32)
        m = _to_gray(mask)
        if m.shape != out.shape[:2]:
            m = np.array(
                Image.fromarray(m, mode="L").resize((out.shape[1], out.shape[0]), Image.NEAREST),
                dtype=np.uint8,
            )
        a = (m.astype(np.float32) / 255.0 * float(alpha))[..., None]
        out = out * (1.0 - a) + color * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def process_pipeline(
    per_concept: dict[str, Image.Image],
    *,
    dilate: int = 0,
    feather: int = 0,
    invert: bool = False,
) -> tuple[dict[str, Image.Image], Image.Image | None]:
    """Apply morphology → feather → invert. Fixed order.

    Feathering runs before invert so an inverted (background) mask keeps the
    same soft edge as the foreground one — both are graded around the same
    boundary. Returns (processed_per_concept, combined_or_None).
    """
    out: dict[str, Image.Image] = {}
    for cid, mask in per_concept.items():
        m = apply_dilate_erode(mask, dilate)
        m = feather_mask(m, feather)
        if invert:
            m = invert_mask(m)
        out[cid] = m
    combined = combine_masks(list(out.values())) if out else None
    return out, combined


def make_cutout(
    image: Image.Image,
    mask: Image.Image | np.ndarray,
    *,
    crop: bool = True,
) -> Image.Image:
    """Cut out the masked region as an RGBA image with transparent background.

    Preserves feathered (soft) mask edges in the alpha channel. When `crop=True`,
    trims transparent margins tightly to the bounding box of non-zero alpha pixels.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    alpha = _to_gray(mask)
    if alpha.shape != rgb.shape[:2]:
        alpha = np.array(
            Image.fromarray(alpha, mode="L").resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST),
            dtype=np.uint8,
        )
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    non_zero = alpha > 0
    rgba[non_zero, :3] = rgb[non_zero]
    rgba[..., 3] = alpha
    out = Image.fromarray(rgba, mode="RGBA")
    if crop:
        bbox = Image.fromarray(alpha, mode="L").getbbox()
        if bbox is not None:
            out = out.crop(bbox)
    return out
