#!/usr/bin/env python3
"""Prove that nothing was written outside the project folder.

    python scripts/leakcheck.py --mark      # BEFORE installing: snapshot global cache locations
    python scripts/leakcheck.py --report    # AFTER installing/running: compare against the snapshot

It checks (1) well-known global cache folders (pip, uv, Hugging Face, PyTorch, CUDA, Triton...)
for any new or changed file, and (2) new top-level entries in the user's profile folders and
leftover temp files that look like they come from our tools. Exit code 1 means a possible leak.
Run it with the system Python, NOT from an activated project environment.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from _common import CACHE, dir_stats, human

MARK_FILE = CACHE / "leakcheck-mark.json"
SUSPECT_TEMP_NAME = re.compile(r"(torch|pip-|uv|huggingface|hf_|tsfm|triton|inductor|cuda|numba|^\.tmp)", re.IGNORECASE)


def _home() -> Path:
    return Path.home()


def watched_dirs() -> dict[str, Path]:
    home = _home()
    watch: dict[str, Path] = {}
    system = platform.system()
    if system == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        roaming = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        watch.update({
            "pip cache (LOCALAPPDATA\\pip)": local / "pip",
            "uv (LOCALAPPDATA\\uv)": local / "uv",
            "uv (APPDATA\\uv)": roaming / "uv",
            "torch (LOCALAPPDATA\\torch)": local / "torch",
            "huggingface (LOCALAPPDATA\\huggingface)": local / "huggingface",
            "CUDA JIT cache (APPDATA\\NVIDIA\\ComputeCache)": roaming / "NVIDIA" / "ComputeCache",
        })
    elif system == "Darwin":
        caches = home / "Library" / "Caches"
        watch.update({
            "pip cache": caches / "pip",
            "uv cache": caches / "uv",
            "torch cache": caches / "torch",
            "huggingface cache": caches / "huggingface",
        })
    else:
        watch.update({
            "pip cache (~/.cache/pip)": home / ".cache" / "pip",
            "uv cache (~/.cache/uv)": home / ".cache" / "uv",
            "huggingface (~/.cache/huggingface)": home / ".cache" / "huggingface",
            "torch (~/.cache/torch)": home / ".cache" / "torch",
            "uv data (~/.local/share/uv)": home / ".local" / "share" / "uv",
        })
    for name in (".cache", ".triton", ".nv", ".huggingface", ".torch", ".nv/ComputeCache"):
        watch[f"~/{name}"] = home / name
    return watch


def toplevel_roots() -> dict[str, Path]:
    home = _home()
    roots = {"~ (profile root)": home}
    if platform.system() == "Windows":
        roots["LOCALAPPDATA"] = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        roots["APPDATA"] = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    elif platform.system() == "Darwin":
        roots["~/Library/Caches"] = home / "Library" / "Caches"
        roots["~/Library/Application Support"] = home / "Library" / "Application Support"
    else:
        roots["~/.cache"] = home / ".cache"
        roots["~/.local/share"] = home / ".local" / "share"
    return roots


def system_temp() -> Path:
    if platform.system() == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", _home() / "AppData" / "Local")) / "Temp"
    return Path("/tmp")


def list_names(path: Path) -> list[str]:
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def snapshot() -> dict:
    return {
        "time": time.time(),
        "iso": datetime.now().astimezone().isoformat(timespec="seconds"),
        "watch": {
            name: dict(zip(("files", "bytes", "newest"), dir_stats(path)), path=str(path), exists=path.exists())
            for name, path in watched_dirs().items()
        },
        "toplevel": {name: list_names(path) for name, path in toplevel_roots().items()},
    }


def do_mark() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    snap = snapshot()
    MARK_FILE.write_text(json.dumps(snap, indent=1), encoding="utf-8")
    print(f"Baseline saved at {snap['iso']}  ({MARK_FILE.relative_to(MARK_FILE.parents[1])})")
    for name, info in snap["watch"].items():
        state = f"{info['files']} files, {human(info['bytes'])}" if info["exists"] else "absent"
        print(f"  {name:55s} {state}")


def do_report() -> int:
    if not MARK_FILE.exists():
        sys.exit("No baseline found: run with --mark BEFORE installing.")
    before = json.loads(MARK_FILE.read_text(encoding="utf-8"))
    now = snapshot()
    print(f"Baseline: {before['iso']}\nNow     : {now['iso']}\n")
    problems = 0

    print("Global cache folders (must be unchanged):")
    for name, old in before["watch"].items():
        new = now["watch"][name]
        changed = (old["files"], old["bytes"], old["exists"]) != (new["files"], new["bytes"], new["exists"]) \
            or new["newest"] > before["time"]
        if changed:
            problems += 1
            print(f"  CHANGED   {name}: {old['files']} -> {new['files']} files, "
                  f"{human(old['bytes'])} -> {human(new['bytes'])}")
        else:
            state = f"{new['files']} files, {human(new['bytes'])}" if new["exists"] else "absent"
            print(f"  unchanged {name}: {state}")

    print("\nNew top-level entries in profile/app-data folders (informational, other apps may write here too):")
    any_new = False
    for name, old_names in before["toplevel"].items():
        added = sorted(set(now["toplevel"][name]) - set(old_names))
        if added:
            any_new = True
            print(f"  {name}: {', '.join(added)}")
    if not any_new:
        print("  none")

    print("\nTemp files that look like they come from our tools:")
    tmp = system_temp()
    suspects = []
    for entry in list_names(tmp):
        if "claude" in entry.lower() or not SUSPECT_TEMP_NAME.search(entry):
            continue
        try:
            if (tmp / entry).stat().st_mtime > before["time"]:
                suspects.append(entry)
        except OSError:
            continue
    if suspects:
        problems += 1
        print("  " + ", ".join(suspects[:20]))
    else:
        print("  none")

    print("\nRESULT: " + ("possible leak, review the lines above" if problems else
                          "OK, nothing outside the project folder was touched"))
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mark", action="store_true")
    group.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.mark:
        do_mark()
    else:
        sys.exit(do_report())


if __name__ == "__main__":
    main()
