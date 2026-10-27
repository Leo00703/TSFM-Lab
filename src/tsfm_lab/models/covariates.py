"""Pre-flight checks for covariates, derived from reproducible upstream problems.

Upstream issue #7 (mistralai/timestral): with known-future covariates such as prices, the 0.99
quantile can explode when (1) the covariate is constant over the context but changes in the
horizon (per-row normalisation divides by ~0), or (2) ``symmetric_averaging`` negates a
sign-fixed covariate, producing out-of-distribution inputs.
"""

from __future__ import annotations

import numpy as np


def _is_binary(row: np.ndarray) -> bool:
    values = np.unique(row[np.isfinite(row)])
    return values.size <= 2 and set(values.tolist()) <= {0.0, 1.0}


def flat_context_rows(known: np.ndarray, context_len: int, max_context: int | None) -> np.ndarray:
    """Boolean (B, Ck): row is constant over the (truncated) context window yet not over the horizon."""
    start = max(0, context_len - max_context) if max_context else 0
    ctx = known[:, :, start:context_len]
    horizon = known[:, :, context_len:]
    with np.errstate(all="ignore"):
        flat = np.nanmax(ctx, axis=-1) - np.nanmin(ctx, axis=-1) == 0
        changes = np.nanmax(np.abs(horizon - ctx[:, :, -1:]), axis=-1) > 0
    return flat & changes


def sign_fixed_rows(known: np.ndarray) -> np.ndarray:
    """Boolean (Ck,): non-binary covariate whose values never change sign (prices, quantities)."""
    out = np.zeros(known.shape[1], dtype=bool)
    for c in range(known.shape[1]):
        row = known[:, c, :].ravel()
        row = row[np.isfinite(row)]
        if row.size and not _is_binary(row) and (row.min() >= 0 or row.max() <= 0):
            out[c] = True
    return out


def inspect_known_covariates(known: np.ndarray, context_len: int, max_context: int | None,
                             symmetric_averaging: bool) -> tuple[list[str], np.ndarray]:
    """Return (warnings, flat_mask). ``flat_mask`` (B, Ck) marks rows that trigger upstream issue #7."""
    warnings: list[str] = []
    flat = flat_context_rows(known, context_len, max_context)
    if flat.any():
        n, total = int(flat.sum()), flat.size
        warnings.append(
            f"{n}/{total} known-covariate rows are constant over the context but change in the horizon: "
            "upstream issue #7 can make the upper quantiles explode. Option 'mask_flat_covariates' "
            "sets such rows to missing.")
    if symmetric_averaging:
        signs = sign_fixed_rows(known)
        if signs.any():
            idx = ", ".join(str(i) for i in np.flatnonzero(signs))
            warnings.append(
                f"Known covariate(s) [{idx}] never change sign and symmetric averaging is on: the mirrored "
                "pass negates them (out-of-distribution). If the upper quantiles look wrong, try "
                "symmetric_averaging=false.")
    return warnings, flat


def mask_flat_rows(known: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Copy of ``known`` with flagged rows set to NaN over the whole window (upstream-suggested workaround)."""
    out = known.copy()
    b_idx, c_idx = np.nonzero(flat)
    out[b_idx, c_idx, :] = np.nan
    return out
