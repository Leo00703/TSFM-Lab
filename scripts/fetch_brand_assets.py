#!/usr/bin/env python3
"""Fetch the Mistral/Timestral brand assets used by the UI, for LOCAL PRIVATE USE ONLY.

The ALTMistral font and the Timestral logos belong to Mistral AI. They are downloaded
from the same public URLs the official sites serve to browsers, stored in the git-ignored
folder ``web/assets/mistral/``, and must never be committed, published or redistributed.
Without them the UI falls back to Inter and a neutral icon.

    python scripts/fetch_brand_assets.py [--force]
"""

from __future__ import annotations

import argparse
import re
import urllib.request

from _common import ROOT, human

DEST = ROOT / "web" / "assets" / "mistral"
FONT_WEIGHTS = ["Regular", "Italic", "Medium", "MediumItalic", "Semibold", "SemiboldItalic"]

# (url, relative destination, expected leading bytes)
ASSETS = [
    *[
        (f"https://mistral.ai/fonts/alt-mistral/ALTMistral-{w}.woff2", f"fonts/ALTMistral-{w}.woff2", b"wOF2")
        for w in FONT_WEIGHTS
    ],
    ("https://mistral.ai/cms-media/api/media/file/timestral-icon-model.svg",
     "logos/timestral-icon-model.svg", b"<svg"),
    ("https://huggingface.co/mistralai/Timestral-209M/resolve/main/timestral-hourglass.png",
     "logos/timestral-hourglass.png", b"\x89PNG\r\n\x1a\n"),
    ("https://raw.githubusercontent.com/mistralai/timestral/main/assets/timestral-banner.jpg",
     "logos/timestral-banner.jpg", b"\xff\xd8\xff"),
]
UNSAFE_SVG = re.compile(rb"<script|\bon\w+\s*=|javascript:|<foreignObject|href\s*=\s*[\"']https?:", re.IGNORECASE)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tsfm-lab-fetch-brand-assets"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    failures = 0
    for url, rel, magic in ASSETS:
        target = DEST / rel
        if target.exists() and not args.force:
            print(f"skip   {rel} (exists)")
            continue
        try:
            data = fetch(url)
            if not data.lstrip().startswith(magic):
                raise ValueError("unexpected file signature")
            if rel.endswith(".svg") and UNSAFE_SVG.search(data):
                raise ValueError("SVG contains scripts or external references")
        except Exception as error:  # noqa: BLE001 - report and continue; assets are optional
            failures += 1
            print(f"FAILED {rel}: {error}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"ok     {rel} ({human(len(data))})")
    if failures:
        print(f"{failures} asset(s) missing: the UI will use its fallbacks.")


if __name__ == "__main__":
    main()
