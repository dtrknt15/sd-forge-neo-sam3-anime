"""Anime concept presets for SAM3 text prompts.

Each entry: (id, Japanese UI label, English short noun phrase for SAM3).
Only English short noun phrases are sent to SAM3.
"""

from __future__ import annotations

PRESETS: list[tuple[str, str, str]] = [
    ("face", "顔", "anime face"),
    ("hair", "髪", "anime hair"),
    ("eyes", "目", "anime eyes"),
    ("mouth", "口", "anime mouth"),
    ("body", "体", "anime body"),
    ("clothes", "服", "anime clothes"),
    ("hands", "手", "hands"),
    ("fingers", "指", "fingers"),
    ("legs", "脚", "legs"),
    ("feet", "足", "feet"),
    ("sky", "空", "sky"),
    ("background", "背景", "background"),
    ("extra", "自由入力", ""),
]

PRESET_IDS: list[str] = [p[0] for p in PRESETS]
PRESET_LABELS: list[str] = [p[1] for p in PRESETS]
DEFAULT_SELECTED: list[str] = ["face"]

# Fixed overlay colors (RGB) per concept, used only for preview overlay.
OVERLAY_PALETTE: dict[str, tuple[int, int, int]] = {
    "face": (255, 82, 82),
    "hair": (255, 193, 7),
    "eyes": (33, 150, 243),
    "mouth": (233, 30, 99),
    "body": (156, 39, 176),
    "clothes": (0, 188, 212),
    "hands": (76, 175, 80),
    "fingers": (139, 195, 74),
    "legs": (255, 152, 0),
    "feet": (121, 85, 72),
    "sky": (3, 169, 244),
    "background": (158, 158, 158),
    "extra": (255, 235, 59),
}

LOG_PREFIX = "[SAM3 Anime]"

ACCORDION_LABEL = "SAM3 Anime Mask"


def prompt_for_id(preset_id: str) -> str:
    for pid, _label, prompt in PRESETS:
        if pid == preset_id:
            return prompt
    return ""


def label_for_id(preset_id: str, labels: dict[str, str] | None = None) -> str:
    if labels and preset_id in labels:
        return labels[preset_id]
    for pid, label, _prompt in PRESETS:
        if pid == preset_id:
            return label
    return preset_id


def resolve_prompts(
    selected: list[str],
    free_text: str,
) -> list[tuple[str, str, str]]:
    """Return [(id, english_prompt, display_label), ...] for presets + free text.

    Free text is always used when non-empty (extra checkbox optional).
    Comma-separated items become separate concepts (extra, extra_1, ...).
    """
    result: list[tuple[str, str, str]] = []
    for pid in selected or []:
        if pid == "extra":
            continue
        prompt = prompt_for_id(pid)
        if prompt:
            result.append((pid, prompt, label_for_id(pid)))
    free = (free_text or "").strip()
    if free:
        parts = [p.strip() for p in free.replace("、", ",").split(",") if p.strip()]
        if len(parts) == 1:
            result.append(("extra", parts[0], parts[0]))
        else:
            for i, part in enumerate(parts):
                cid = "extra" if i == 0 else f"extra_{i}"
                result.append((cid, part, part))
    return result
