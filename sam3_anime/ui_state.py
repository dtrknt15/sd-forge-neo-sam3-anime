"""UI-facing state logic extracted from scripts/sam3_anime_mask.py.

These helpers are pure (no Gradio, no torch) so they can be unit-tested.
The script layer stays responsible for wiring them into Gradio events.
"""

from __future__ import annotations

from sam3_anime import constants

# Unit-based / third-party panels whose images must never be auto-used as
# the SAM3 reference (ControlNet units are the common false positive).
THIRD_PARTY_IMAGE_MARKERS: tuple[str, ...] = (
    "controlnet",
    "control_net",
    "control-net",
    "alwayson",
    "t2i_adapter",
    "ip_adapter",
    "ipadapter",
    "reference_only",
    "animatediff",
    "reactor",
    "adetailer",
)


def _blob(*parts: str | None) -> str:
    return " ".join(p for p in parts if p).lower()


def is_third_party_image_source(*parts: str | None) -> bool:
    """True when any id/class/label blob looks like a third-party image unit."""
    joined = _blob(*parts)
    return any(marker in joined for marker in THIRD_PARTY_IMAGE_MARKERS)


def accepts_as_img2img_init(
    elem_id: str | None,
    parent_ids: list[str | None] | None = None,
    *,
    class_blob: str = "",
    label_blob: str = "",
) -> bool:
    """True only for legitimate img2img init sources.

    Rejects ControlNet / third-party units, txt2img canvases, and anything
    that cannot be positively identified as img2img (no more capture-all).
    """
    chain = [elem_id or "", class_blob, label_blob, *(parent_ids or [])]
    if is_third_party_image_source(*chain):
        return False
    joined = _blob(*chain)
    if "txt2img" in joined:
        return False
    eid = (elem_id or "").lower()
    if eid.startswith("img_inpaint"):
        return True
    return "img2img" in joined


def resolve_generation_inputs(
    image: object | None,
    canvas_data: str | None,
    selected: list[str] | None,
    free: str | None,
) -> tuple[object | None, list[tuple[str, str]], dict[str, str]]:
    """Resolve the image and prompt rows for one generate run.

    Order: extension Image → Forge Canvas data URL → None (caller fetches
    img2img init / Inpaint upload). Returns (image_or_None,
    [(concept_id, prompt), ...], {concept_id: display_label}). Raises
    ValueError with a UI-safe message when there is no image or no prompts.
    """
    img = image
    if img is None and canvas_data:
        img = "CANVAS"  # sentinel: caller decodes the data URL
    if img is None:
        raise ValueError(
            "No image. 拡張内の「参照画像」か img2img の init image を使ってください。"
            "（ControlNet など他拡張の画像は自動参照しません）"
        )

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


def gallery_visible_ids(
    store_masks: dict | None,
    active: list[str] | None,
) -> list[str]:
    """Concept ids currently shown in the gallery (same order as gallery items)."""
    if not store_masks:
        return []
    return [cid for cid in (active or []) if cid in store_masks]


def gallery_flat_index(index: object, columns: int = 5) -> int:
    """Normalize Gradio SelectData.index (int or (row, col)) to a flat index."""
    if isinstance(index, (tuple, list)):
        if not index:
            return -1
        if len(index) == 1:
            return int(index[0])
        row, col = int(index[0]), int(index[1])
        return row * max(int(columns), 1) + col
    try:
        return int(index)
    except (TypeError, ValueError):
        return -1


def gallery_index_at(
    index: object,
    store_masks: dict | None,
    active: list[str] | None,
    columns: int = 5,
) -> str | None:
    """Map a gallery SelectData.index to a concept id (or None if out of range)."""
    ids = gallery_visible_ids(store_masks, active)
    if not ids:
        return None
    flat = gallery_flat_index(index, columns=columns)
    if flat < 0 or flat >= len(ids):
        return None
    return ids[flat]


def overlay_label(mode: str = "multi", concept_id: str | None = None, labels: dict[str, str] | None = None) -> str:
    if mode == "single" and concept_id:
        return f"オーバーレイ — {constants.label_for_id(concept_id, labels)} ({concept_id})"
    return "オーバーレイ（色分け）"


def build_color_legend_html(
    concept_ids: list[str],
    labels: dict[str, str] | None = None,
) -> str:
    """Compact color legend for multi-overlay (HTML chips for Gradio Markdown)."""
    if not concept_ids:
        return ""
    chips: list[str] = []
    for cid in concept_ids:
        r, g, b = constants.OVERLAY_PALETTE.get(cid, constants.OVERLAY_PALETTE["extra"])
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        label = constants.label_for_id(cid, labels)
        chips.append(
            f'<span style="color:{hex_color};font-weight:600">■</span>'
            f'<span style="margin-right:0.75em"> {label}</span>'
        )
    return f'<div style="font-size:12px;line-height:1.8">{"".join(chips)}</div>'


FACE_FOCUS_IDS: tuple[str, ...] = ("face", "hair", "eyes", "mouth")
PERSON_FOCUS_IDS: tuple[str, ...] = (
    "face",
    "hair",
    "eyes",
    "mouth",
    "body",
    "clothes",
    "hands",
    "fingers",
    "legs",
    "feet",
)

QUICK_SELECT_MODES: tuple[str, ...] = ("all", "none", "face", "person", "invert")


def quick_select_active(mode: str, available: list[str], current: list[str] | None = None) -> list[str]:
    """Return concept ids to activate for a quick-select mode.

    Modes: all / none / face (顔まわり) / person (人物) / invert.
    Result keeps the order of `available` and only uses ids that exist there.
    """
    avail = list(available or [])
    cur = set(current or [])
    if mode == "all":
        return avail
    if mode == "none":
        return []
    if mode == "invert":
        return [cid for cid in avail if cid not in cur]
    if mode == "face":
        want = set(FACE_FOCUS_IDS)
    elif mode == "person":
        want = set(PERSON_FOCUS_IDS)
    else:
        return avail
    return [cid for cid in avail if cid in want]


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
