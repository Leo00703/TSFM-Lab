"""Take the screenshots the README shows, in the light and the dark theme, from a running TSFM-Lab.

    python scripts/readme_screenshots.py [--base http://127.0.0.1:8765] [--suite ID] [--run ID] [--out docs/images]

The server must be running and have a finished Leaderboard run and a finished run (it only reads them: the Playground
forecast and the Showcase save nothing). The brand assets under web/assets/mistral are blocked in the browser, so the
pictures show what a fresh clone shows: the built-in pixel mark, not the Timestral icon. Uses the Chrome or Edge that is
installed, like tests/test_ui_smoke.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 1280
SCALE = 1.5  # 1280 css pixels -> 1920 px wide files, as sharp as a retina screen needs at the README's 900 px
PAD = 28  # blank space kept under the last thing a picture shows
COMPARED = ("Chronos-2 Small", "TimesFM 2.5", "TiRex-2")


def api(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return json.loads(response.read())


def pick_suite(base: str) -> str:
    """The finished Leaderboard run that ranks every model on the most datasets."""
    suites = [s for s in api(base, "/api/suites") if s["status"] == "done" and len(s["models"]) >= 10]
    if not suites:
        raise SystemExit("no finished Leaderboard run with ten models: pass --suite")
    return max(suites, key=lambda s: s["datasets"])["id"]


def pick_run(base: str) -> str:
    runs = api(base, "/api/runs")
    if not runs:
        raise SystemExit("no run to show: pass --run")
    return runs[0]["id"]


def bottom(page: Page, text: str) -> float:
    box = page.get_by_text(text).first.bounding_box()
    if box is None:
        raise SystemExit(f"'{text}' is not on the page")
    return box["y"] + box["height"]


def shot(page: Page, path: Path, y1: float, x0: float = 0, x1: float = WIDTH, y0: float = 0) -> None:
    page.screenshot(path=str(path), clip={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
    print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB")


def showcase(page: Page, base: str, out: Path, theme: str) -> None:
    page.goto(f"{base}/#/showcase")
    page.wait_for_selector("text=Likely range")
    page.evaluate("document.querySelector('main section').style.display = 'none'")  # the introduction above the examples
    page.get_by_text("Forecasting…", exact=True).wait_for(state="detached", timeout=120000)  # the spinner of the chart, not the page title
    page.wait_for_timeout(1500)
    shot(page, out / f"showcase-{theme}.png", bottom(page, "Likely range (50% and 80%)") + 14)  # ends before the card's next divider


def playground(page: Page, base: str, out: Path, theme: str) -> None:
    page.goto(f"{base}/#/playground")
    page.wait_for_selector("text=Run forecast")
    page.get_by_label("Horizon").fill("96")
    page.get_by_label("Context (steps)").fill("240")
    for name in COMPARED:
        page.get_by_label(name, exact=True).check()
    page.get_by_role("button", name="Run forecast").click()
    page.wait_for_selector("text=Results for this window", timeout=240000)
    page.wait_for_timeout(1500)
    page.evaluate("window.scrollTo(0, 0)")  # the page scrolls to the results; the clip below is measured from the top
    box = page.locator("main figure").first.bounding_box()  # the chart card; the table under it is wider than its card, so it stays out
    if box is None:
        raise SystemExit("the Playground chart did not render")
    shot(page, out / f"playground-{theme}.png", box["y"] + box["height"] + 8, box["x"] - 8, box["x"] + box["width"] + 8, box["y"] - 8)


def leaderboard(page: Page, base: str, out: Path, theme: str, suite: str) -> None:
    page.goto(f"{base}/#/leaderboard/{suite}")
    page.wait_for_selector("text=Skill is measured against")
    page.wait_for_timeout(1000)
    shot(page, out / f"leaderboard-{theme}.png", bottom(page, "Skill is measured against") + PAD)


def calibration(page: Page, base: str, out: Path, theme: str, run: str) -> None:
    """The calibration chart of a run page: small enough to stay readable next to the Playground chart in the README."""
    page.goto(f"{base}/#/runs/{run}")
    page.wait_for_selector("text=Share of actuals below each predicted quantile")
    page.wait_for_timeout(1500)
    box = page.get_by_text("Share of actuals below each predicted quantile").first.locator("xpath=ancestor::figure[1]").bounding_box()
    if box is None:
        raise SystemExit("the calibration chart did not render")
    top = box["y"] + page.evaluate("window.scrollY")
    path = out / f"calibration-{theme}.png"
    page.screenshot(path=str(path), full_page=True, clip={"x": box["x"] - 8, "y": top - 8, "width": box["width"] + 16, "height": box["height"] + 16})
    print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--suite")
    parser.add_argument("--run")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    args = parser.parse_args()
    suite, run = args.suite or pick_suite(args.base), args.run or pick_run(args.base)
    with sync_playwright() as p:
        for channel in ("chrome", "msedge", None):
            try:
                browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                break
            except PlaywrightError as error:
                print(f"{channel or 'playwright chromium'}: {str(error).splitlines()[0]}", file=sys.stderr)
        else:
            raise SystemExit("no Chrome, Edge or Playwright Chromium on this machine")
        for theme in ("light", "dark"):
            context = browser.new_context(viewport={"width": WIDTH, "height": 2200}, device_scale_factor=SCALE, color_scheme=theme)
            context.add_init_script(f"localStorage.setItem('tsfm-theme', '{theme}')")
            context.route("**/assets/mistral/**", lambda route: route.abort())
            page = context.new_page()
            showcase(page, args.base, args.out, theme)
            playground(page, args.base, args.out, theme)
            leaderboard(page, args.base, args.out, theme, suite)
            calibration(page, args.base, args.out, theme, run)
            context.close()
        browser.close()


if __name__ == "__main__":
    main()
