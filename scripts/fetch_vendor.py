#!/usr/bin/env python3
"""Vendor the two browser libraries the UI needs, straight from the npm registry, verified.

Pinned versions and the registry's own sha512 integrity values; the tarball is rejected if it does not
match. Files land in web/vendor/ (committed, so the app works offline and without Node or a CDN).

    python scripts/fetch_vendor.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import sys
import tarfile
import urllib.request

from _common import ROOT

VENDOR = ROOT / "web" / "vendor"
PACKAGES = [
    {
        "name": "htm", "version": "3.1.1", "license": "Apache-2.0 (bundles Preact 10, MIT)",
        "url": "https://registry.npmjs.org/htm/-/htm-3.1.1.tgz",
        "integrity": "sha512-983Vyg8NwUE7JkZ6NmOqpCZ+sh1bKv2iYTlUkzlWmA5JD2acKoxd4KVxbMmxX/85mtfdnDmTFoNKcg5DGAvxNQ==",
        "files": {"package/preact/standalone.module.js": "htm-preact.js", "package/LICENSE": "LICENSE-htm.txt"},
    },
    {
        "name": "echarts", "version": "6.1.0", "license": "Apache-2.0",
        "url": "https://registry.npmjs.org/echarts/-/echarts-6.1.0.tgz",
        "integrity": "sha512-q0yaFPggC9FUdsWH4blavRWFmxdrIodbkoKNAjJudAI6CA9gNPxHtV2RcZNEepZVlk4yvBYkOkbk6HIVpIyHZA==",
        "files": {"package/dist/echarts.min.js": "echarts.min.js", "package/LICENSE": "LICENSE-echarts.txt",
                  "package/NOTICE": "NOTICE-echarts.txt"},
    },
]
PREACT_LICENSE = "https://raw.githubusercontent.com/preactjs/preact/main/LICENSE"


def main() -> int:
    VENDOR.mkdir(parents=True, exist_ok=True)
    for pkg in PACKAGES:
        if all((VENDOR / out).exists() for out in pkg["files"].values()):
            print(f"skip   {pkg['name']} {pkg['version']} (present)")
            continue
        raw = urllib.request.urlopen(urllib.request.Request(pkg["url"], headers={"User-Agent": "tsfm-lab"})).read()
        algo, expected = pkg["integrity"].split("-", 1)
        actual = base64.b64encode(hashlib.new(algo, raw).digest()).decode()
        if actual != expected:
            print(f"FAILED {pkg['name']}: integrity mismatch, nothing written")
            return 1
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            for member, out in pkg["files"].items():
                (VENDOR / out).write_bytes(tar.extractfile(member).read())
        print(f"ok     {pkg['name']} {pkg['version']} ({pkg['license']}) integrity verified")
    if not (VENDOR / "LICENSE-preact.txt").exists():
        (VENDOR / "LICENSE-preact.txt").write_bytes(urllib.request.urlopen(PREACT_LICENSE).read())
    (VENDOR / "VERSIONS.txt").write_text(
        "".join(f"{p['name']} {p['version']}  {p['license']}  {p['integrity'][:30]}...\n" for p in PACKAGES),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
