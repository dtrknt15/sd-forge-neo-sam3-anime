"""Global singleton SAM3 model/processor management.

Official builder (facebookresearch/sam3):
  build_sam3_image_model(checkpoint_path=..., device="cuda", load_from_HF=False)
  Sam3Processor(model)

Official _load_checkpoint uses torch.load and keys containing "detector.".
safetensors needs a separate load path.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from .constants import LOG_PREFIX
from . import vram

_STATE: dict[str, Any] = {
    "model": None,
    "processor": None,
    "ckpt_path": None,
    "device": "cuda",
    "dtype": None,
}

_CKPT_EXTS = {".pt", ".pth", ".safetensors"}
_OFFICIAL_CKPT_NAMES = ("sam3.pt", "sam3.1_multiplex.pt", "sam3.1.pt")


def _webui_models_dir() -> Path:
    try:
        from modules import paths_internal

        base = Path(paths_internal.models_path)
    except Exception:
        try:
            from modules import shared

            base = Path(shared.cmd_opts.ckpt_dir) if getattr(shared.cmd_opts, "ckpt_dir", None) else Path("models")
        except Exception:
            base = Path("models")
    return base / "SAM3"


def models_dir() -> Path:
    return _webui_models_dir()


def missing_checkpoint_hint() -> str:
    folder = _webui_models_dir()
    return (
        f"チェックポイント未検出です。公式の `sam3.pt` を `{folder}` に置いて "
        "**Refresh checkpoints** を押してください。"
        " Hugging Face の [facebook/sam3](https://huggingface.co/facebook/sam3) は gated なので事前申請が必要です。"
        " または環境変数 `SAM3_CHECKPOINT` にファイルパスを指定できます。"
    )


def list_checkpoints() -> list[str]:
    names: list[str] = []
    env = os.environ.get("SAM3_CHECKPOINT", "").strip()
    if env and Path(env).is_file():
        names.append(Path(env).name)
    folder = _webui_models_dir()
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in _CKPT_EXTS and p.name not in names:
                names.append(p.name)
    # Official facebookresearch checkpoints first (sam3.pt is the safe default)
    names.sort(key=lambda n: (0 if n in _OFFICIAL_CKPT_NAMES else 1, n.lower()))
    return names


def preferred_checkpoint_name(names: list[str] | None = None) -> str | None:
    names = names if names is not None else list_checkpoints()
    for preferred in _OFFICIAL_CKPT_NAMES:
        if preferred in names:
            return preferred
    for n in names:
        if n.lower().endswith((".pt", ".pth")):
            return n
    return names[0] if names else None


def is_official_checkpoint_name(name: str | None) -> bool:
    if not name:
        return False
    return name in _OFFICIAL_CKPT_NAMES


def checkpoint_warning(name: str | None) -> str:
    """UI hint when the selected file is not the official SAM3 image checkpoint."""
    if not name:
        return ""
    if is_official_checkpoint_name(name):
        return ""
    if name.lower().endswith(".safetensors"):
        return (
            "WARNING: 非公式 safetensors の可能性があります。"
            "精度が出ない場合は models/SAM3 に公式の sam3.pt を置いて選択してください。"
        )
    return "NOTE: 公式チェックポイント名（sam3.pt）ではありません。"


def resolve_ckpt_path(name: str | None) -> Path | None:
    env = os.environ.get("SAM3_CHECKPOINT", "").strip()
    folder = _webui_models_dir()

    if name:
        direct = folder / name
        if direct.is_file():
            return direct
        if env and Path(env).name == name and Path(env).is_file():
            return Path(env)
        if Path(name).is_file():
            return Path(name)
        return None

    if env and Path(env).is_file():
        return Path(env)
    preferred = preferred_checkpoint_name()
    if preferred:
        return folder / preferred
    return None


def is_loaded() -> bool:
    return _STATE["model"] is not None and _STATE["processor"] is not None


def current_ckpt() -> str | None:
    return _STATE["ckpt_path"]


def unload() -> None:
    _STATE["model"] = None
    _STATE["processor"] = None
    _STATE["ckpt_path"] = None
    _STATE["dtype"] = None
    gc.collect()
    vram.free_cache()
    print(f"{LOG_PREFIX} unloaded SAM3", flush=True)


def free_vram() -> None:
    unload()
    vram.free_cache()
    print(f"{LOG_PREFIX} free VRAM done", flush=True)


def _remap_image_ckpt(raw: dict) -> dict:
    """Map checkpoint keys onto Sam3Image state_dict.

    Official .pt: keys under ``detector.*`` / ``tracker.*``
    Some .safetensors: ``detector_model.*`` / ``tracker_model.*`` / ``tracker_neck.*``
    Official loader strips ``detector.`` and maps ``tracker.`` only when
    inst_interactive_predictor exists (we disable it for image text prompts).
    """
    if "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    out: dict = {}
    for k, v in raw.items():
        nk = k
        if nk.startswith("detector_model."):
            nk = nk[len("detector_model.") :]
        elif nk.startswith("detector."):
            nk = nk[len("detector.") :]
        elif nk.startswith("tracker_model.") or nk.startswith("tracker_neck.") or nk.startswith("tracker."):
            # image-only build has no tracker weights to load
            continue
        else:
            continue
        out[nk] = v
    return out


def _load_safetensors_into(model: Any, path: Path) -> None:
    from safetensors.torch import load_file

    raw = load_file(str(path))
    ckpt = _remap_image_ckpt(raw)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"{LOG_PREFIX} safetensors missing keys (first 8): {missing[:8]}", flush=True)
    if unexpected:
        print(f"{LOG_PREFIX} safetensors unexpected keys (first 8): {unexpected[:8]}", flush=True)
    if not ckpt:
        raise RuntimeError("safetensors remap produced no detector weights")


def load(ckpt_name: str | None = None) -> tuple[bool, str]:
    """Load SAM3. Returns (ok, message). Message is UI-safe."""
    if not vram.cuda_available():
        return False, "CUDA not available"

    path = resolve_ckpt_path(ckpt_name)
    if path is None:
        models_dir = _webui_models_dir()
        return False, f"No checkpoint in {models_dir}"

    path_str = str(path)
    if is_loaded() and _STATE["ckpt_path"] == path_str:
        return True, f"already loaded: {path.name}"

    if is_loaded():
        unload()

    try:
        import torch
    except Exception as e:
        return False, f"torch not available: {e}"

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as e:
        return False, f"sam3 package not installed: {e}"

    try:
        print(f"{LOG_PREFIX} loading {path.name} ...", flush=True)
        is_st = path.suffix.lower() == ".safetensors"

        # Official builder: device must be exactly "cuda" for .cuda() branch.
        # load_from_HF=False avoids network when we already have a local ckpt.
        model = build_sam3_image_model(
            checkpoint_path=None if is_st else path_str,
            load_from_HF=False,
            device="cuda",
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )

        if is_st:
            _load_safetensors_into(model, path)
            model = model.cuda()
            model.eval()

        # Keep official float32 weights: Sam3Processor feeds float32 activations.
        # Forcing .half() causes "Input type FloatTensor / weight HalfTensor" errors.

        processor = Sam3Processor(model)
        _STATE["model"] = model
        _STATE["processor"] = processor
        _STATE["ckpt_path"] = path_str
        _STATE["device"] = "cuda"
        _STATE["dtype"] = torch.float32
        print(f"{LOG_PREFIX} loaded {path.name}", flush=True)
        return True, f"loaded: {path.name}"
    except Exception as e:
        _STATE["model"] = None
        _STATE["processor"] = None
        _STATE["ckpt_path"] = None
        gc.collect()
        vram.free_cache()
        text = str(e).strip() or repr(e)
        last = text.splitlines()[-1] if text else "unknown error"
        print(f"{LOG_PREFIX} load failed: {text}", flush=True)
        return False, f"load failed: {last}"


def get_processor() -> Any:
    return _STATE["processor"]


def get_state() -> dict[str, Any]:
    return _STATE
