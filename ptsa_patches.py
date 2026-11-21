"""ptsa_patches.py — monkey-patches for PTSA defects, applied on import.

Imported at the top of helper.py, so every entry point gets them. Idempotent.
Patches the numpy scalar aliases (`np.float` etc.) PTSA still references.
"""
from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _patch_numpy_removed_aliases()
    _APPLIED = True


def _patch_numpy_removed_aliases() -> None:
    """numpy 1.24 removed `np.float` / `np.int` / ... ; PTSA's older pyFR read
    paths still use them and fail with AttributeError. That dropped a
    site-correlated subset of sessions (TJ_pyFR; 69 errors in one roi_power run),
    which would bias group comparisons. Each alias is restored to the builtin it
    aliased (`np.float` WAS `float`), only where missing.
    """
    import numpy as np

    for alias, builtin_type in (("float", float), ("int", int), ("bool", bool),
                                ("object", object), ("str", str),
                                ("complex", complex)):
        if not hasattr(np, alias):
            setattr(np, alias, builtin_type)


apply()
