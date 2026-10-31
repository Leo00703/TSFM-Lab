"""Interface smoke test: a real browser opens every page of the app and fails on the first script error.

The interface is plain JavaScript modules with no build step and no unit tests of its own, so a typo in a page (a variable that
a comment swallowed, a class that does not exist) only shows when the page is opened. This opens them all, with real runs to
look at. It drives an installed Chrome or Edge through Playwright and is skipped where there is none.
"""

from __future__ import annotations

import contextlib
import shutil
import socket
import threading
import time
import urllib.request

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

from tsfm_lab.api.app import create_app

pytestmark = pytest.mark.ui
SPEC = {"source": "synthetic", "id": "multi_seasonality", "params": {"n_series": 3, "length": 300}}
BENIGN = ("/assets/mistral/", "/favicon")  # optional brand assets that are git-ignored and absent on a fresh clone


def _post(base: str, path: str, body: dict) -> dict:
    import json

    request = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(request, timeout=30).read())


def _get(base: str, path: str):
    import json

    return json.loads(urllib.request.urlopen(base + path, timeout=30).read())


def _finished(base: str, job_id: str, timeout: float = 90.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = _get(base, f"/api/jobs/{job_id}")
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError("the job did not finish")


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """The app on a free local port, with two finished runs (so the run and comparison pages have something to show)."""
    import uvicorn

    from tsfm_lab.api.build_css import ensure_css

    with contextlib.suppress(Exception):  # the stylesheet is built when the server starts; without it every page is unstyled but still runs
        ensure_css()
    root = tmp_path_factory.mktemp("ui")
    # the app serves the interface from <root>/web: a copy of the project's own, without the git-ignored brand assets, so the
    # test writes its runs to a temporary folder and never to the project's
    from tsfm_lab.isolation import find_root

    shutil.copytree(find_root() / "web", root / "web", ignore=shutil.ignore_patterns("assets"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(root=root)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    end = time.time() + 20
    while not server.started and time.time() < end:
        time.sleep(0.05)
    assert server.started, "the test server did not start"
    try:
        runs = []
        for label in ("first", "second"):
            job = _post(base, "/api/jobs/experiment", {"model": "naive", "dataset": SPEC, "horizon": 24, "windows": 3, "label": f"smoke {label}"})
            done = _finished(base, job["id"])
            assert done["status"] == "done", done
            runs.append(done["result"]["run_id"])
        yield {"base": base, "runs": runs}
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        app.state.worker.stop()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        launched = None
        for channel in ("chrome", "msedge", None):  # an installed browser first; Playwright's own Chromium if someone installed it
            try:
                launched = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                break
            except PlaywrightError:
                continue
        if launched is None:
            pytest.skip("no Chrome, Edge or Playwright Chromium on this machine")
        yield launched
        launched.close()


def _problems(page):
    problems: list[str] = []
    page.on("pageerror", lambda e: problems.append(f"script error: {e}"))
    page.on("console", lambda m: problems.append(f"console error: {m.text}") if m.type == "error" and not any(b in (m.location.get("url") or "") for b in BENIGN) else None)
    return problems


def _open(page, base: str, route: str, ready: str, problems: list[str]) -> None:
    page.goto(f"{base}/#/{route}")
    try:
        page.wait_for_selector(ready, timeout=15000)
    except PlaywrightError as error:  # say what the page complained about instead of only that it stayed empty
        raise AssertionError(f"{route} did not render: {problems[:4]} | {page.inner_text('body')[:200]!r}") from error
    page.wait_for_timeout(700)  # what the page does after its first paint


PAGES = [  # (route, text that shows the page really rendered)
    ("showcase", "See what the model can do"), ("models", "Model library"), ("datasets", "Online benchmarks"), ("playground", "Single forecast"),
    ("experiments", "Compare with"), ("leaderboard", "Start leaderboard run"), ("runs", "Runs"), ("system", "Hardware and software"), ("about", "Licences and definitions"),
    ("runs/{run}", "Summary"), ("compare?runs={a},{b}", "Forecast by forecast"),
    ("runs/nope", "This run could not be opened"), ("leaderboard/nope", "This run could not be opened"), ("compare?runs=nope,nada", "These runs could not be compared"),
]


@pytest.mark.parametrize(("route", "text"), PAGES, ids=[p[0] for p in PAGES])
def test_every_page_opens_without_a_script_error(live, browser, route, text) -> None:
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    try:
        problems = _problems(page)
        a, b = live["runs"]
        _open(page, live["base"], route.format(run=a, a=a, b=b), "main h1, main h2", problems)
        assert text.lower() in page.inner_text("main").lower() or text.lower() in page.inner_text("body").lower(), f"{route}: '{text}' is not on the page"
        if "nope" in route:  # a page for something that does not exist: the 404 of the API is the point, the script must still be clean
            problems = [p for p in problems if "404" not in p]
        assert problems == [], f"{route}: {problems[:3]}"
    finally:
        page.close()


def test_a_missing_brand_icon_leaves_one_mark_after_navigating(live, browser) -> None:
    """The test server has no web/assets (git-ignored), so the Timestral icon 404s, as on a fresh clone: the fallback mark must stay the only one."""
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    try:
        problems = _problems(page)
        _open(page, live["base"], "runs", "main h1, main h2", problems)  # not the Showcase: its forecast would keep the worker busy for the tests after this one
        page.get_by_role("link", name="Models").first.click()
        page.wait_for_selector("text=Model library")
        page.wait_for_timeout(700)  # the icon fails to load a moment after the page draws it
        assert page.locator("img[src*='/assets/mistral/']").count() == 0, "a broken icon is drawn next to the fallback mark"
        assert page.locator("header span.bg-surface-brand, nav span.bg-surface-brand").count() >= 1
        assert problems == [], problems[:3]
    finally:
        page.close()


def test_curves_over_quantile_levels_are_drawn_under_their_own_levels(live, browser) -> None:
    """TimesFM answers 0.1 to 0.9 and Timestral 21 levels: a calibration curve must not be drawn under another model's labels."""
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    try:
        problems = _problems(page)
        _open(page, live["base"], "runs", "main h1, main h2", problems)
        aligned = page.evaluate(
            """async () => { const { alignByLevel } = await import('/js/charts.js');
              return alignByLevel([{ levels: [0.01, 0.1, 0.5, 0.9], values: [1, 2, 3, 4] }, { levels: [0.1, 0.30000000000000004, 0.9], values: [20, 30, 40] }]); }"""
        )
        assert aligned["x"] == [0.01, 0.1, 0.3, 0.5, 0.9]
        assert aligned["data"] == [[1, 2, None, 3, 4], [None, 20, 30, None, 40]]
        assert problems == [], problems[:3]
    finally:
        page.close()


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_theme_follows_the_system_until_one_is_chosen(live, browser, scheme) -> None:
    context = browser.new_context(color_scheme=scheme, viewport={"width": 1280, "height": 800})
    page = context.new_page()
    try:
        problems = _problems(page)
        is_dark = "document.documentElement.classList.contains('dark')"
        _open(page, live["base"], "models", "main h1, main h2", problems)
        assert page.evaluate(is_dark) is (scheme == "dark"), "a first visit must show the theme of the operating system"
        other = "Light theme" if scheme == "dark" else "Dark theme"
        page.get_by_role("button", name=other).click()
        page.wait_for_function(f"{is_dark} === {str(scheme != 'dark').lower()}")
        page.reload()
        page.wait_for_selector("main h1, main h2")
        assert page.evaluate(is_dark) is (scheme != "dark"), "a theme chosen by hand must beat the system one after a reload"
        assert problems == [], problems[:3]
    finally:
        context.close()


def test_the_menus_open_and_close_and_the_phone_menu_lists_every_page(live, browser) -> None:
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    try:
        problems = _problems(page)
        _open(page, live["base"], "showcase", "main h1", problems)
        page.get_by_role("button", name="Evaluate").click()
        page.get_by_role("link", name="Leaderboard").first.wait_for(state="visible")  # the page redraws after the click: wait for it, do not read it at once
        blurb = page.get_by_text("Rank the models on a benchmark or your data")
        for _ in range(20):  # the page listens for Escape from an effect that runs a moment after the menu appears: press again until it has
            page.keyboard.press("Escape")
            try:
                blurb.wait_for(state="hidden", timeout=500)
                break
            except PlaywrightError:
                continue
        else:
            raise AssertionError("Escape did not close the Evaluate menu")
        page.get_by_role("link", name="Models").first.click()
        page.wait_for_selector("text=Model library")
        page.get_by_role("button", name="Dark theme").click()
        page.wait_for_function("document.documentElement.className.includes('dark')")  # the theme class is set by the page a moment after the click
        page.get_by_role("button", name="Light theme").click()
        page.wait_for_function("!document.documentElement.className.includes('dark')")
        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_role("button", name="Open menu").click()
        page.wait_for_timeout(350)  # the menu grows open over 200 ms
        names = {page.get_by_role("link", name=n).first.inner_text().strip() for n in ("Showcase", "Models", "Datasets", "Playground", "Experiments", "Leaderboard", "Runs", "System", "About")}
        assert len(names) == 9
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "the page scrolls sideways on a phone"
        assert problems == [], problems[:3]
    finally:
        page.close()


def test_the_benchmark_form_does_not_offer_a_device_the_model_cannot_run_on(live, browser) -> None:
    """TimesFM 2.5 cannot use the Apple GPU: the form shows it switched off for that model (the machine here has none, so one is simulated)."""
    import re

    page = browser.new_page(viewport={"width": 1280, "height": 900})
    try:
        problems = _problems(page)

        def with_apple_gpu(route):
            data = route.fetch().json()
            data["devices"] = [d for d in data["devices"] if d["kind"] != "mps"]  # a Mac has a real one: the test brings its own
            data["devices"].append({"id": "mps", "kind": "mps", "name": "Apple M5 (simulated)", "memory_gb": None, "supported": True, "reason": None})
            route.fulfill(json=data)

        def all_downloaded(route):
            data = route.fetch().json()
            for model in data:
                if "cache" in model:  # the baselines have no weights
                    model["cache"]["downloaded"] = True
            route.fulfill(json=data)

        page.route("**/api/system", with_apple_gpu)
        page.route("**/api/models", all_downloaded)
        _open(page, live["base"], "system", "main h1", problems)
        apple = page.get_by_role("checkbox", name=re.compile("MPS"))
        refusal = page.get_by_text("not supported by this model")
        expect(apple).to_be_enabled()
        page.get_by_role("combobox").filter(has_text="Timestral-5M").click()
        page.get_by_role("option", name=re.compile("TimesFM 2.5")).click()
        expect(apple).to_be_disabled()
        expect(refusal).to_be_visible()
        page.get_by_role("combobox").filter(has_text="TimesFM 2.5").click()
        page.get_by_role("option", name=re.compile("TimesFM 3.0")).click()
        expect(apple).to_be_enabled()
        expect(refusal).to_have_count(0)
        assert problems == [], problems[:3]
    finally:
        page.close()
