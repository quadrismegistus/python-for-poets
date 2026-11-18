"""
Give matplotlib (which draws our plots) a folder it can write to.

Matplotlib keeps a font cache in your home folder (~/.matplotlib on a Mac).
AI assistants' sandboxes often can't write there, and then matplotlib warns,
and rebuilds its font cache in a temporary folder every time. If its usual
folder isn't writable, we point it at p4p_data/matplotlib in the current
folder instead. This has to happen before matplotlib is first imported.
"""

import os
import sys
from pathlib import Path


def _usual_folders():
    # Where matplotlib would look, following matplotlib._get_config_or_cache_dir.
    if sys.platform.startswith(("linux", "freebsd")):
        home = Path.home()
        return [
            Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / "matplotlib",
            Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache") / "matplotlib",
        ]
    if sys.platform == "win32":
        old = Path.home() / ".matplotlib"
        if old.is_dir() or not os.environ.get("LOCALAPPDATA"):
            return [old]
        return [Path(os.environ["LOCALAPPDATA"]) / "matplotlib"]
    return [Path.home() / ".matplotlib"]


def _writable(folder):
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(folder, os.W_OK)


def use_writable_folder():
    if os.environ.get("MPLCONFIGDIR") or "matplotlib" in sys.modules:
        return  # already chosen, or too late to change
    try:
        if all(_writable(folder) for folder in _usual_folders()):
            return
    except RuntimeError:  # no home folder at all
        pass
    os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join("p4p_data", "matplotlib"))


use_writable_folder()
