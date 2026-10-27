"""tsfm-lab: local sandbox to run, compare and inspect time-series foundation models.

Unofficial; not affiliated with Mistral AI. Importing this package keeps every cache
inside the project folder (see :mod:`tsfm_lab.isolation`).
"""

from tsfm_lab import isolation as _isolation

_isolation.apply()

__version__ = "0.1.0"
