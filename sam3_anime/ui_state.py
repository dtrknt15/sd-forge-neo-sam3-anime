"""UI-facing state logic extracted from scripts/sam3_anime_mask.py.

These helpers are pure (no Gradio, no torch) so they can be unit-tested.
The script layer stays responsible for wiring them into Gradio events.
"""

from __future__ import annotations

from sam3_anime import constants


def resolve_generation_inputs(
    image: object | None,
    canvas_data: str | None,
    selected: list[str] | None,
    free: str | None,
) -> tuple[object | None, list[tuple[str, str]], dict[str, str]]:
    """Resolve the image and prompt rows for one generate run.

    Order: extension Image → Forge Canvas data URL → None (caller fetches
    img2img init). Returns (image_or_None, [(concept_id, prompt), ...],
    {concept_id: display_label}). Raises ValueError with a UI-safe message
    when there is no image or no prompts at all.
    """
    img = image
    if img is None and canvas_data:
        img = "CANVAS"  # sentinel: caller decodes the data URL
    if img is None:
        raise ValueError("No image. 拡張内の Image か img2img の init image を使ってください。")

    rows = constants.resolve_prompts(selected or [], free or "")
    prompts = [(cid, prompt) for cid, prompt, _label in rows]
    if not prompts:
        raise ValueError("プリセットか自由入力を1つ以上指定してください。 自由入力のみなら extra チェックは不要です。")
    labels = {cid: label for cid, _prompt, label in rows}
    return img, prompts, labels


def button_interactivity(running: bool) -> dict[str, bool]:
    """Interactivity of generate/load/unload/free while SAM3 is busy.

    Cancel is intentionally excluded — it must stay clickable during a run
    so a long generation can always be stopped.
    """
    return {
        "generate": not running,
        "load": not running,
        "unload": not running,
        "free": not running,
    }


def build_active_choices(
    processed: dict[str, object],
    labels: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Return (label, concept_id) choices for the active-masks checkbox group."""
    return [(f"{constants.label_for_id(cid, labels)} ({cid})", cid) for cid in processed]


def build_status_message(
    generated_count: int,
    notes: list[str] | None = None,
    auto_unload: str | None = None,
) -> str:
    """Compose the status line shown after Generate."""
    msg = f"Generated {generated_count} mask(s)."
    if notes:
        msg += " " + " ".join(notes)
    if auto_unload:
        msg += " | " + auto_unload
    return msg


def build_export_note(
    png_path: str | None,
    *,
    image_set: bool = True,
    set_error: str | None = None,
) -> str:
    """Compose the export status markdown.

    image_set=False → components were missing, PNG download fallback.
    set_error → image/mask could not be written into the components.
    """
    png = f"`{png_path}`" if png_path else "-"
    if set_error:
        return f"Export prepared（画像セットに失敗: {set_error}）. PNG: {png}"
    if not image_set:
        return (
            f"Export prepared（送信先: **img2img → Inpaint upload**）. "
            f"マスク PNG: {png}。"
            "自動送信に失敗した場合は PNG を手動で Inpaint upload へどうぞ。"
        )
    return f"Exported to **img2img → Inpaint upload**. マスク PNG: {png}"
