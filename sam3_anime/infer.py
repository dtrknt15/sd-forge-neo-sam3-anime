"""SAM3 inference per concept. Sequential prompts to limit peak VRAM."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
from PIL import Image

from . import model_manager
from .constants import LOG_PREFIX


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
    masks = output.get("masks") if isinstance(output, dict) else getattr(output, "masks", None)
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


def _state_to_mask(state: Any, threshold: float) -> tuple[Image.Image | None, str | None]:
    """Extract combined mask + threshold-filter note from SAM3 processor state.

    Returns (mask_L_or_None, note_or_None). The note explains when the score
    threshold filter had NO effect so the user can notice it in the status:
    - output carries no scores (filter cannot run)
    - score count != mask count (filter cannot align)
    - threshold excluded every candidate (fallback: all masks are used)
    """
    masks = _masks_from_output(state)
    if not masks:
        return None, None
    scores = _scores_from_output(state)
    note: str | None = None
    if scores is None:
        note = "score threshold 無効: 出力にスコアがありません"
    elif len(scores) != len(masks):
        note = f"score threshold 無効: スコア数({len(scores)})とマスク数({len(masks)})が不一致"
    else:
        keep = [masks[i] for i, s in enumerate(scores) if float(s) >= float(threshold)]
        if keep:
            masks = keep
        else:
            note = (
                f"score threshold={threshold:g} が全候補({len(masks)})を除外したため"
                "全マスクを使用しました（フィルタ無効）"
            )
    h, w = masks[0].shape[:2]
    acc = _or_masks(masks, (h, w))
    if acc is None:
        return None, note
    return Image.fromarray(acc, mode="L"), note


def _extract_candidates(state: Any) -> dict[str, Any] | None:
    """Extract CPU-only candidate masks + optional scores for live threshold re-filtering."""
    masks = _masks_from_output(state)
    if not masks:
        return None
    cand: dict[str, Any] = {"masks": masks}
    scores = _scores_from_output(state)
    if scores is not None:
        cand["scores"] = scores
    return cand


def filter_candidates(
    candidates: dict[str, dict[str, Any]],
    threshold: float,
    orig_size: tuple[int, int] | None = None,
) -> tuple[dict[str, Image.Image], list[str]]:
    """Re-filter cached candidate masks at `threshold` without re-running SAM3."""
    result: dict[str, Image.Image] = {}
    notes: list[str] = []
    for cid, cand in (candidates or {}).items():
        mask, note = _state_to_mask(cand, threshold)
        if note and note not in notes:
            notes.append(note)
        if mask is not None:
            if orig_size is not None and mask.size != orig_size:
                mask = mask.resize(orig_size, Image.NEAREST)
            result[cid] = mask
    return result, notes


def _run_concepts(
    processor: Any,
    image: Image.Image,
    prompts: list[tuple[str, str]],
    threshold: float,
) -> tuple[dict[str, Image.Image], dict[str, dict[str, Any]], list[str], list[str], bool]:
    """set_image once, then sequential set_text_prompt with kept state.

    Returns (masks, candidates, skipped_prompts, threshold_notes, cancelled).
    threshold_notes are deduplicated messages telling the user when the
    score threshold filter had no effect for a concept.
    """
    result: dict[str, Image.Image] = {}
    candidates: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    threshold_notes: list[str] = []
    cancelled = False

    with _autocast_ctx():
        state = processor.set_image(image)

    for cid, prompt in prompts:
        if model_manager.is_cancel_requested():
            # No GPU work here — just mark every remaining prompt as skipped
            # so the status line shows exactly what was not run.
            print(f"{LOG_PREFIX} cancelled before prompt: {prompt}", flush=True)
            skipped.append(f"Cancelled: {prompt}")
            cancelled = True
            continue
        note: str | None = None
        cand: dict[str, Any] | None = None
        try:
            with _autocast_ctx():
                # Official: set_text_prompt(prompt=..., state=...)
                state = processor.set_text_prompt(prompt=prompt, state=state)
            cand = _extract_candidates(state)
            mask, note = _state_to_mask(state, threshold)
        except Exception:
            # Fallback: full re-run (image + text) if state reuse fails
            try:
                with _autocast_ctx():
                    state2 = processor.set_image(image)
                    state2 = processor.set_text_prompt(prompt=prompt, state=state2)
                cand = _extract_candidates(state2)
                mask, note = _state_to_mask(state2, threshold)
            except Exception as e2:
                last = str(e2).strip().splitlines()[-1] if str(e2).strip() else repr(e2)
                print(f"{LOG_PREFIX} prompt failed: {prompt} — {last}", flush=True)
                skipped.append(f"Prompt failed: {prompt}")
                continue

        if note:
            if note not in threshold_notes:
                threshold_notes.append(note)
            print(f"{LOG_PREFIX} {prompt}: {note}", flush=True)

        if mask is None:
            print(f"{LOG_PREFIX} Prompt produced no mask: {prompt}", flush=True)
            skipped.append(f"Prompt produced no mask: {prompt}")
            continue
        result[cid] = mask
        if cand is not None:
            candidates[cid] = cand

    return result, candidates, skipped, threshold_notes, cancelled


def generate_masks(
    image: Image.Image,
    prompts: list[tuple[str, str]],
    *,
    max_side: int = 1024,
    threshold: float = 0.5,
) -> tuple[dict[str, Image.Image], list[str], bool]:
    """Return ({id: mask_L at original size}, notes, cancelled).

    notes combines skipped prompts and score-threshold notices.
    When cancelled is True the result contains only the concepts completed
    before the cancel request; notes includes one Cancelled: entry per
    remaining prompt.
    """
    result, _candidates, notes, cancelled = generate_masks_with_candidates(
        image,
        prompts,
        max_side=max_side,
        threshold=threshold,
    )
    return result, notes, cancelled


def generate_masks_with_candidates(
    image: Image.Image,
    prompts: list[tuple[str, str]],
    *,
    max_side: int = 1024,
    threshold: float = 0.5,
) -> tuple[dict[str, Image.Image], dict[str, dict[str, Any]], list[str], bool]:
    """Return ({id: mask_L at original size}, candidates, notes, cancelled)."""
    image = image.convert("RGB")

    # Block Load/Unload/Free VRAM (and a concurrent second generate) while
    # inference is running; released in a finally block on any error path.
    with model_manager.inference_running():
        return _generate_locked(image, prompts, max_side=max_side, threshold=threshold)


def _generate_locked(
    image: Image.Image,
    prompts: list[tuple[str, str]],
    *,
    max_side: int,
    threshold: float,
) -> tuple[dict[str, Image.Image], dict[str, dict[str, Any]], list[str], bool]:
    result: dict[str, Image.Image] = {}

    if not model_manager.is_loaded():
        raise RuntimeError("SAM3 not loaded")

    processor = model_manager.get_processor()
    if processor is None:
        raise RuntimeError("SAM3 not loaded")

    orig_size = image.size
    fitted = _fit_image(image, int(max_side))
    raw, candidates, skipped, threshold_notes, cancelled = _run_concepts(processor, fitted, prompts, threshold)

    for cid, mask in raw.items():
        if mask.size != orig_size:
            mask = mask.resize(orig_size, Image.NEAREST)
        result[cid] = mask

    return result, candidates, skipped + threshold_notes, cancelled
