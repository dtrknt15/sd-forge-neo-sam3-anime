"""Postprocess masks: dilate/erode, invert, overlay."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image

from .constants import OVERLAY_PALETTE


def _to_np(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L"))
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
    return (arr > 127).astype(np.uint8) * 255


def _morph_pil(mask_u8: np.ndarray, amount: int) -> np.ndarray:
    """amount > 0 dilate, amount < 0 erode. Elliptical kernel, iterations=|amount|."""
    if amount == 0:
        return mask_u8
    img = Image.fromarray(mask_u8, mode="L")
    iterations = abs(int(amount))
    # size 3 elliptical-ish via Max/Min filter
    from PIL import ImageFilter

    op = ImageFilter.MaxFilter if amount > 0 else ImageFilter.MinFilter
    # MaxFilter/MinFilter only support odd sizes >= 3
    size = 3
    out = img
    for _ in range(iterations):
        out = out.filter(op(size))
    return np.array(out, dtype=np.uint8)


def apply_dilate_erode(mask: Image.Image | np.ndarray, amount: int) -> Image.Image:
    arr = _to_np(mask)
    arr = _morph_pil(arr, int(amount))
    return Image.fromarray(arr, mode="L")


def invert_mask(mask: Image.Image) -> Image.Image:
    arr = _to_np(mask)
    return Image.fromarray(255 - arr, mode="L")


def combine_masks(masks: Iterable[Image.Image]) -> Image.Image:
    masks = list(masks)
    if not masks:
        raise ValueError("no masks to combine")
    acc = _to_np(masks[0])
    for m in masks[1:]:
        acc = np.bitwise_or(acc, _to_np(m))
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


def dummy_mask(size: tuple[int, int], preset_index: int = 0) -> Image.Image:
    """White rectangle dummy for UI testing without SAM3."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(mask)
    # place rectangles in a grid-ish pattern so multiple presets differ
    cols = 4
    idx = max(0, preset_index)
    cx = (idx % cols) * (w // cols) + w // 16
    cy = (idx // cols) * (h // 4) + h // 8
    rw = max(8, w // 6)
    rh = max(8, h // 6)
    x2 = min(w - 1, cx + rw)
    y2 = min(h - 1, cy + rh)
    draw.rectangle([cx, cy, x2, y2], fill=255)
    return mask


def process_pipeline(
    per_concept: dict[str, Image.Image],
    *,
    dilate: int = 0,
    invert: bool = False,
) -> tuple[dict[str, Image.Image], Image.Image | None]:
    """Apply morphology then invert. Fixed order: morphology → invert.

    Returns (processed_per_concept, combined_or_None).
    """
    out: dict[str, Image.Image] = {}
    for cid, mask in per_concept.items():
        m = apply_dilate_erode(mask, dilate)
        if invert:
            m = invert_mask(m)
        out[cid] = m
    combined = combine_masks(list(out.values())) if out else None
    return out, combined
