"""SAM3 inference per concept. Sequential prompts to limit peak VRAM."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
from PIL import Image

from .constants import LOG_PREFIX
from . import model_manager, postprocess


def _fit_image(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    side = max(w, h)
    if side <= max_side:
        return image
    scale = max_side / float(side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return image.resize((nw, nh), Image.LANCZOS)


def _autocast_ctx():
    """Official video path uses bf16 autocast; image processor does not.

    Flash Attention on Ampere produces bf16 activations that then hit
    float32 Linear weights. Wrapping matches sam3_base_predictor / video code.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    except Exception:
        pass
    return nullcontext()


def _masks_from_output(output: Any) -> list[np.ndarray]:
    """Extract list of HxW uint8 masks (0/255) from SAM3 processor state."""
    if output is None:
        return []
    masks = None
    if isinstance(output, dict):
        masks = output.get("masks")
    else:
        masks = getattr(output, "masks", None)
    if masks is None:
        return []
    try:
        import torch

        if isinstance(masks, torch.Tensor):
            t = masks.detach().float().cpu()
            if t.ndim == 4:
                t = t.squeeze(1) if t.shape[1] == 1 else t.reshape(-1, t.shape[-2], t.shape[-1])
            elif t.ndim == 2:
                t = t.unsqueeze(0)
            return [((t[i].numpy() > 0.5).astype(np.uint8)) * 255 for i in range(t.shape[0])]
    except Exception:
        pass
    try:
        arr = np.asarray(masks)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim == 4:
            arr = arr[:, 0] if arr.shape[1] == 1 else arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        out = []
        for i in range(arr.shape[0]):
            a = arr[i]
            out.append(((a > 0.5) if a.dtype != np.uint8 else (a > 127)).astype(np.uint8) * 255)
        return out
    except Exception:
        pass
    if isinstance(masks, (list, tuple)):
        out = []
        for m in masks:
            a = np.asarray(m)
            if a.ndim > 2:
                a = a.reshape(a.shape[-2], a.shape[-1])
            out.append(((a > 0.5) if a.dtype != np.uint8 else (a > 127)).astype(np.uint8) * 255)
        return out
    return []


def _scores_from_output(output: Any) -> np.ndarray | None:
    if not isinstance(output, dict) or "scores" not in output:
        return None
    scores = output["scores"]
    try:
        import torch

        if isinstance(scores, torch.Tensor):
            scores = scores.detach().float().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(scores, dtype=np.float32).reshape(-1)
    except Exception:
        return None


def _or_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    if not masks:
        return None
    acc = np.zeros(shape, dtype=np.uint8)
    for m in masks:
        if m.shape != shape:
            m = np.array(Image.fromarray(m, mode="L").resize((shape[1], shape[0]), Image.NEAREST), dtype=np.uint8)
        acc = np.bitwise_or(acc, m)
    return acc if acc.any() else None


def _state_to_mask(state: Any, threshold: float) -> Image.Image | None:
    masks = _masks_from_output(state)
    if not masks:
        return None
    scores = _scores_from_output(state)
    if scores is not None and len(scores) == len(masks):
        keep = [masks[i] for i, s in enumerate(scores) if float(s) >= float(threshold)]
        if keep:
            masks = keep
    h, w = masks[0].shape[:2]
    acc = _or_masks(masks, (h, w))
    if acc is None:
        return None
    return Image.fromarray(acc, mode="L")


def _run_concepts(
    processor: Any,
    image: Image.Image,
    prompts: list[tuple[str, str]],
    threshold: float,
) -> tuple[dict[str, Image.Image], list[str]]:
    """set_image once, then sequential set_text_prompt with kept state."""
    result: dict[str, Image.Image] = {}
    skipped: list[str] = []

    with _autocast_ctx():
        state = processor.set_image(image)

    for cid, prompt in prompts:
        try:
            with _autocast_ctx():
                # Official: set_text_prompt(prompt=..., state=...)
                state = processor.set_text_prompt(prompt=prompt, state=state)
            mask = _state_to_mask(state, threshold)
        except Exception as e:
            # Fallback: full re-run (image + text) if state reuse fails
            try:
                with _autocast_ctx():
                    state2 = processor.set_image(image)
                    state2 = processor.set_text_prompt(prompt=prompt, state=state2)
                mask = _state_to_mask(state2, threshold)
            except Exception as e2:
                last = str(e2).strip().splitlines()[-1] if str(e2).strip() else repr(e2)
                print(f"{LOG_PREFIX} prompt failed: {prompt} — {last}", flush=True)
                skipped.append(f"Prompt failed: {prompt}")
                continue

        if mask is None:
            print(f"{LOG_PREFIX} Prompt produced no mask: {prompt}", flush=True)
            skipped.append(f"Prompt produced no mask: {prompt}")
            continue
        result[cid] = mask

    return result, skipped


def generate_masks(
    image: Image.Image,
    prompts: list[tuple[str, str]],
    *,
    max_side: int = 1024,
    threshold: float = 0.5,
    use_dummy: bool = False,
) -> tuple[dict[str, Image.Image], list[str]]:
    """Return ({id: mask_L at original size}, skipped_prompt_notes)."""
    image = image.convert("RGB")
    orig_size = image.size
    skipped: list[str] = []
    result: dict[str, Image.Image] = {}

    if use_dummy:
        for i, (cid, _prompt) in enumerate(prompts):
            result[cid] = postprocess.dummy_mask(orig_size, i)
        return result, skipped

    if not model_manager.is_loaded():
        raise RuntimeError("SAM3 not loaded")

    processor = model_manager.get_processor()
    if processor is None:
        raise RuntimeError("SAM3 not loaded")

    fitted = _fit_image(image, int(max_side))
    raw, skipped = _run_concepts(processor, fitted, prompts, float(threshold))

    for cid, mask in raw.items():
        if mask.size != orig_size:
            mask = mask.resize(orig_size, Image.NEAREST)
        result[cid] = mask

    return result, skipped
