"""Rules about the interface source that a browser-less test run can still enforce."""

import re
from pathlib import Path

WEB_JS = Path(__file__).resolve().parents[1] / "web" / "js"


def test_no_native_browser_dialogs_are_used() -> None:
    """Confirmations go through dialog.js (the app's own style); the browser's confirm/alert/prompt boxes are not used."""
    offenders = []
    for path in sorted(WEB_JS.rglob("*.js")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = re.sub(r"//.*$", "", line)  # comments may mention the old calls
            if re.search(r"(?<![\w.])(confirm|alert|prompt)\(", code) and "dialog.js" not in path.name:
                offenders.append(f"{path.relative_to(WEB_JS)}:{number}")
    assert offenders == [], f"use confirmDialog() instead of the native dialogs: {offenders}"


def test_the_dialog_module_is_mounted_and_used_by_every_destructive_action() -> None:
    main = (WEB_JS / "main.js").read_text(encoding="utf-8")
    assert "DialogHost" in main
    users = [p.name for p in (WEB_JS / "pages").glob("*.js") if "confirmDialog(" in p.read_text(encoding="utf-8")]
    assert sorted(users) == ["catalog.js", "datasets.js", "leaderboard.js", "models.js", "runs.js"]
