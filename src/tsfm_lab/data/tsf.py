"""Reader for Monash's ``.tsf`` time-series files (the format of the Monash Time Series Forecasting Archive).

A file is a text header (``@attribute``, ``@frequency``, ``@horizon``, ``@missing``, ``@equallength``, then ``@data``) followed
by one series per line: the values of the declared attributes separated by colons, then the comma-separated observations
(``?`` = missing). Only numbers and short strings are parsed; nothing in a file is ever executed.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Frequency words of the archive -> the calendar step between two observations.
STEPS: dict[str, dict[str, int]] = {
    "yearly": {"years": 1}, "quarterly": {"months": 3}, "monthly": {"months": 1}, "weekly": {"days": 7}, "daily": {"days": 1},
    "hourly": {"hours": 1}, "half_hourly": {"minutes": 30}, "minutely": {"minutes": 1}, "10_minutes": {"minutes": 10},
    "4_seconds": {"seconds": 4},
}
# ... and a pandas alias that describes it (for display, seasonal period and axis labels).
ALIASES = {"yearly": "YS", "quarterly": "QS", "monthly": "MS", "weekly": "7D", "daily": "D", "hourly": "h",
           "half_hourly": "30min", "minutely": "min", "10_minutes": "10min", "4_seconds": "4s"}
DEFAULT_START = "1900-01-01 00-00-00"  # what the archive's own loader assumes when a file has no start timestamp


@dataclass
class TsfHeader:
    relation: str = ""
    attributes: list[tuple[str, str]] = field(default_factory=list)  # (name, type) in column order
    frequency: str | None = None
    horizon: int | None = None
    missing: bool = False
    equal_length: bool = False
    description: str = ""


def _open_text(source: Path | str | io.IOBase) -> io.TextIOBase:
    if isinstance(source, io.IOBase):
        return io.TextIOWrapper(source, encoding="utf-8", errors="replace", newline="")
    path = Path(source)
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        name = next(n for n in archive.namelist() if n.endswith(".tsf"))
        return io.TextIOWrapper(archive.open(name), encoding="utf-8", errors="replace", newline="")
    return open(path, encoding="utf-8", errors="replace", newline="")


def parse_start(text: str) -> np.datetime64:
    """'1979-01-01 00-00-00' -> datetime64 (the archive writes the time with dashes)."""
    day, _, clock = text.strip().partition(" ")
    return np.datetime64(f"{day}T{clock.replace('-', ':')}" if clock else day, "ms")


def _numbers(text: str) -> np.ndarray:
    return np.array(text.replace("?", "nan").split(","), dtype=np.float64)


def read(source: Path | str | io.IOBase, wanted: Callable[[int], bool] | None = None
         ) -> tuple[TsfHeader, Iterator[tuple[dict[str, Any], np.ndarray | None]]]:
    """Parse the header now and the series lazily: returns ``(header, iterator of (attributes, values))``.

    ``wanted(index)`` decides which lines have their numbers parsed; the others come back as ``(attributes, None)``, which keeps
    sampling a file of 145,000 series cheap."""
    handle = _open_text(source)
    header = TsfHeader()
    for raw in handle:
        line = raw.strip()
        if line.startswith("#"):
            header.description += line.lstrip("# ") + " "
        elif line.lower().startswith("@relation"):
            header.relation = line.split(None, 1)[1] if " " in line else ""
        elif line.lower().startswith("@attribute"):
            parts = line.split()
            header.attributes.append((parts[1], parts[2].lower()))
        elif line.lower().startswith("@frequency"):
            header.frequency = line.split(None, 1)[1].strip().lower()
        elif line.lower().startswith("@horizon"):
            header.horizon = int(line.split()[1])
        elif line.lower().startswith("@missing"):
            header.missing = line.split()[1].lower() == "true"
        elif line.lower().startswith("@equallength"):
            header.equal_length = line.split()[1].lower() == "true"
        elif line.lower().startswith("@data"):
            break
    names = [n for n, _ in header.attributes]

    def series() -> Iterator[tuple[dict[str, Any], np.ndarray | None]]:
        try:
            index = -1
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split(":", len(names))
                if len(parts) != len(names) + 1:
                    raise ValueError(f"a series line has {len(parts) - 1} attributes, the header declares {len(names)}")
                index += 1
                yield dict(zip(names, parts[:-1], strict=True)), (_numbers(parts[-1]) if wanted is None or wanted(index) else None)
        finally:
            handle.close()

    return header, series()


def timestamps(start: np.datetime64, frequency: str, length: int) -> np.ndarray:
    """Regular timestamps (datetime64[ms]) for ``length`` observations starting at ``start``."""
    step = STEPS.get(frequency or "")
    if step is None:
        raise ValueError(f"unknown frequency '{frequency}'")
    start = np.datetime64(start, "ms")
    months = step.get("years", 0) * 12 + step.get("months", 0)
    if not months:  # a fixed step: plain arithmetic, which stays fast for series of millions of observations
        fixed = sum({"weeks": 604_800_000, "days": 86_400_000, "hours": 3_600_000, "minutes": 60_000, "seconds": 1000}[k] * v
                    for k, v in step.items())
        return start + np.arange(length, dtype=np.int64) * np.timedelta64(fixed, "ms")
    first = start.astype("datetime64[M]")  # calendar steps keep the day of the month (the archive's starts are on the 1st)
    return (first + np.arange(length, dtype=np.int64) * months).astype("datetime64[ms]") + (start - first.astype("datetime64[ms]"))
