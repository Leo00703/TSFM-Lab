import tomllib
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "tsfm_lab"


def test_every_data_file_in_the_package_is_shipped_in_the_wheel() -> None:
    """A file read at run time (the benchmark catalogues, the model manifests) that package-data forgets is silently missing from a wheel."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["tsfm_lab"]
    data_files = sorted(p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file() and p.suffix not in {".py", ".pyc"} and "__pycache__" not in p.parts)
    assert data_files, "the package has data files (catalogues, manifests)"
    missing = [f for f in data_files if not any(fnmatch(f, pattern) for pattern in patterns)]
    assert not missing, f"not listed in [tool.setuptools.package-data]: {missing}"
