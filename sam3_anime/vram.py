"""VRAM status and cache helpers (RTX 3090 / float16 target)."""

from __future__ import annotations

from .constants import LOG_PREFIX


def allocated_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_allocated() / (1024**3))
    except Exception:
        return None


def reserved_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_reserved() / (1024**3))
    except Exception:
        return None


def status_text(loaded: bool, ckpt_name: str | None = None) -> str:
    state = "loaded" if loaded else "not loaded"
    parts = [f"SAM3: {state}"]
    if loaded and ckpt_name:
        parts.append(f"ckpt={ckpt_name}")
    alloc = allocated_gb()
    reserved = reserved_gb()
    if alloc is not None:
        parts.append(f"allocated {alloc:.1f} GiB")
    if reserved is not None:
        parts.append(f"reserved {reserved:.1f} GiB")
    if alloc is None and not loaded:
        parts.append("CUDA unknown / not ready")
    return " | ".join(parts)


def free_cache() -> None:
    import gc

    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"{LOG_PREFIX} free_cache failed: {e}", flush=True)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_string() -> str:
    return "cuda:0" if cuda_available() else "cpu"
