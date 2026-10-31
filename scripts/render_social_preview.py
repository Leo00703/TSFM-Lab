"""Render the repository's social preview image from its HTML source (1280x640).

GitHub shows this size on the repository's link previews. Upload the PNG by hand under
Settings > General > Social preview: GitHub has no API for it.

The light and the dark version share one design; with no arguments the light one is rendered.

    python scripts/render_social_preview.py [SOURCE.html [TARGET.png]]
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main(source: Path, target: Path) -> None:
    with sync_playwright() as p:
        for channel in ("chrome", "msedge", None):  # an installed browser first, like tests/test_ui_smoke.py
            try:
                browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                break
            except PlaywrightError as error:
                print(f"{channel or 'playwright chromium'}: {str(error).splitlines()[0]}", file=sys.stderr)
        else:
            raise SystemExit("no Chrome, Edge or Playwright Chromium on this machine")
        page = browser.new_page(viewport={"width": 1280, "height": 640}, device_scale_factor=1)
        page.goto(source.as_uri())
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(300)
        page.screenshot(path=str(target), clip={"x": 0, "y": 0, "width": 1280, "height": 640})
        browser.close()
    print(f"{target.relative_to(ROOT)}: {target.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    src = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "docs" / "images" / "social-preview-light.html"
    dst = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src.with_suffix(".png")
    main(src, dst)
