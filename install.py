"""Forge Neo extension installer for SAM3 Anime Mask.

Tries to install the official `sam3` package from GitHub.
Safety policy:
- Never crashes WebUI on failure.
- Never upgrades/changes existing packages in the WebUI environment
  (installs with --no-deps so torch/numpy/etc. are left untouched).
- Preflight checks: Python and torch versions must satisfy the official
  sam3 requirements (Python 3.12+, torch 2.7+). Otherwise skipped.
- Post-install verification: confirms the package imports with the
  existing environment (missing deps => message, no side effects).
Does not install `segment-anything` (name collision).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
import sys

# Official requirements (upstream facebookresearch/sam3).
MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 7)

SAM3_PIP_SPEC = "git+https://github.com/facebookresearch/sam3.git"


def _log(msg: str) -> None:
    print(f"[SAM3 Anime] {msg}", flush=True)


def _parse_version(v: str) -> tuple[int, ...]:
    """Best-effort numeric version tuple: '2.7.1+cu126' -> (2, 7, 1)."""
    parts: list[int] = []
    for piece in v.split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def _at_least(version: str, minimum: tuple[int, ...]) -> bool:
    got = _parse_version(version)
    need = minimum + (0,) * (len(got) - len(minimum))
    return got >= need


def _check_python() -> str | None:
    if sys.version_info >= MIN_PYTHON:
        return None
    return (
        f"Python {sys.version_info.major}.{sys.version_info.minor} is too old for sam3 "
        f"(requires {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+). Skipping install."
    )


def _check_torch() -> str | None:
    try:
        import torch  # noqa: PLC0415 (installer runs before UI; import is safe here)
    except Exception:
        # torch missing entirely: pip would pull it in and we forbid
        # dependency changes -> skip with guidance.
        return (
            "torch not found in this environment. sam3 requires torch 2.7+ but "
            "installing it here could change the WebUI environment. Skipping install. "
            "Install torch manually first if needed."
        )
    version = getattr(torch, "__version__", "") or ""
    if not _at_least(version, MIN_TORCH):
        return (
            f"torch {version or 'unknown'} is too old for sam3 (requires {MIN_TORCH[0]}.{MIN_TORCH[1]}+). "
            "Skipping install to avoid upgrading torch in the WebUI environment."
        )
    return None


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _skip(reason: str) -> None:
    _log(f"sam3 install skipped: {reason}")
    _log(f"manual: pip install {SAM3_PIP_SPEC}")


def _already_installed_with_metadata() -> bool:
    """True if sam3 is importable AND recorded by pip metadata (fully installed)."""
    if not _find_spec("sam3"):
        return False
    try:
        importlib.metadata.version("sam3")
        return True
    except Exception:
        return False


def _verify_post_install() -> bool:
    """Confirm sam3 imports against the current environment.

    With --no-deps, a missing runtime dependency only shows up here.
    No auto-fix: touching the env is exactly what we avoid.
    """
    try:
        subprocess.check_call(
            [sys.executable, "-c", "import sam3"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def main() -> None:
    if os.environ.get("SAM3_ANIME_SKIP_INSTALL") or "--skip-install" in sys.argv:
        _log("install skipped")
        return

    if _already_installed_with_metadata():
        _log("sam3 package already available")
        return

    if _find_spec("sam3"):
        # Importable but no pip metadata: leave as-is rather than reinstall
        # (e.g. vendored or previously installed in a way we cannot see).
        _log("sam3 module found without pip metadata; leaving it untouched")
        return

    # --- preflight: refuse rather than mutate the WebUI environment ---
    for reason in (_check_python(), _check_torch()):
        if reason:
            _skip(reason)
            return

    _log("sam3 not found — attempting install with --no-deps (existing torch/numpy/etc. will NOT be changed)")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        SAM3_PIP_SPEC,
    ]
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        _log(f"sam3 install failed (UI still loads): {e}")
        _log(f"manual: pip install {SAM3_PIP_SPEC}")
        _log("see README for checkpoint and Python/CUDA requirements")
        return

    if _verify_post_install():
        _log("sam3 install finished and imports cleanly")
    else:
        _log(
            "sam3 installed but 'import sam3' failed — a runtime dependency may be "
            "missing (installed with --no-deps). Check the error in a console with: "
            f'{sys.executable} -c "import sam3"'
        )
        _log(f"fix manually in the WebUI venv if needed, e.g.: pip install {SAM3_PIP_SPEC}")


if __name__ == "__main__":
    main()
