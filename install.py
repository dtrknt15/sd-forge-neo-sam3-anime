"""Forge Neo extension installer for SAM3 Anime Mask.

Tries to install the official `sam3` package. Never crashes WebUI on failure.
Does not install `segment-anything` (name collision).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys


def _log(msg: str) -> None:
    print(f"[SAM3 Anime] {msg}", flush=True)


def main() -> None:
    if os.environ.get("SAM3_ANIME_SKIP_INSTALL") or "--skip-install" in sys.argv:
        _log("install skipped")
        return

    if importlib.util.find_spec("sam3") is not None:
        _log("sam3 package already available")
        return

    _log("sam3 not found — attempting install (official package)")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "git+https://github.com/facebookresearch/sam3.git",
    ]
    try:
        subprocess.check_call(cmd)
        _log("sam3 install finished")
    except Exception as e:
        _log(f"sam3 install failed (UI still loads): {e}")
        _log("manual: pip install git+https://github.com/facebookresearch/sam3.git")
        _log("see README for checkpoint and Python/CUDA requirements")


main()
