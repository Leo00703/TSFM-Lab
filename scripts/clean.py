#!/usr/bin/env python3
"""Remove what this project installed. Deleting the whole folder removes everything anyway.

    python scripts/clean.py --cache          # downloaded weights/datasets, pip/uv/torch caches, tools
    python scripts/clean.py --env            # the .venv
    python scripts/clean.py --all --yes      # both, without asking
Your own data (data/user), run results (runs/) and the brand assets are never touched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat

from _common import CACHE, ROOT, dir_stats, human


def _force(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove(path, label: str, assume_yes: bool) -> None:
    files, size, _ = dir_stats(path)
    if not path.exists():
        print(f"{label}: nothing to remove")
        return
    if not assume_yes and input(f"Delete {label} ({human(size)}, {files} files)? [y/N] ").strip().lower() != "y":
        print("  skipped")
        return
    shutil.rmtree(path, onerror=_force)
    print(f"{label}: removed ({human(size)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", action="store_true", help="remove .venv")
    parser.add_argument("--cache", action="store_true", help="remove .cache")
    parser.add_argument("--all", action="store_true", help="remove .venv and .cache")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()
    if not (args.env or args.cache or args.all):
        parser.error("choose --env, --cache or --all")
    if args.env or args.all:
        remove(ROOT / ".venv", ".venv", args.yes)
    if args.cache or args.all:
        remove(CACHE, ".cache", args.yes)


if __name__ == "__main__":
    main()
