"""exclusion_log.py — event/session exclusion-count logging for the FC pipeline.

Each (session, contrast) accumulates input / per-stage-excluded / final event
COUNTS as a long-format table and writes ONE per-session CSV under
``EXCLUSION_COUNTS_DIR/per_session/`` — race-free under the parallel (per-session)
pipeline.

Long schema (one row per count):
    config, subject, experiment, session, contrast, tag, value
``tag`` is ``input_count``, ``<stage>_excluded`` (one per ordered exclusion
stage), or ``final_count``; ``value`` is an event count. Paths + the ``config``
stamp come from config.yaml via project_paths.
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportUnknownLambdaType=warning, reportAttributeAccessIssue=warning
from __future__ import annotations

import os
from os.path import join
from typing import Any, Sized, cast

import numpy as np
import pandas as pd

from project_paths import CONFIG_NAME, EXCLUSION_COUNTS_DIR

ID_COLS = ["config", "subject", "experiment", "session", "contrast"]
LONG_COLS = ID_COLS + ["tag", "value"]


def _count(x: "int | Sized") -> int:
    """Accept either an int count or any sized object (DataFrame/array/list)."""
    return int(x) if isinstance(x, (int, np.integer)) else int(len(x))


def _session_keys(dfrow: pd.Series) -> dict[str, Any]:
    row = cast("dict[str, Any]", dict(dfrow))
    return {
        "subject": str(row["sub"]),
        "experiment": str(row["exp"]),
        "session": int(row["sess"]),
    }


class ExclusionLog:
    """Accumulate event-exclusion counts for one (session, contrast).

    Usage (each call is one line at the call site):
        log = ExclusionLog(dfrow, "word_on")
        log.input(all_words)                                  # input_count
        log.excluded("word_on_serialpos1", before, after)     # word_on_serialpos1_excluded
        log.final(kept)                               # final_count
        log.write()
    ``input``/``final``/``excluded`` accept either counts or sized objects.
    """

    def __init__(self, dfrow: pd.Series, contrast: str,
                 config: str = CONFIG_NAME,
                 out_dir: "str | os.PathLike[str]" = EXCLUSION_COUNTS_DIR) -> None:
        self.keys = _session_keys(dfrow)
        self.contrast = str(contrast)
        self.config = str(config)
        self.out_dir = str(out_dir)
        self.records: list[tuple[str, int]] = []

    def count(self, tag: str, value: "int | Sized") -> "ExclusionLog":
        self.records.append((str(tag), _count(value)))
        return self

    def input(self, events: "int | Sized", tag: str = "input_count") -> "ExclusionLog":
        return self.count(tag, events)

    def final(self, events: "int | Sized", tag: str = "final_count") -> "ExclusionLog":
        return self.count(tag, events)

    def excluded(self, name: str, before: "int | Sized", after: "int | Sized") -> "ExclusionLog":
        return self.count(f"{name}_excluded", _count(before) - _count(after))

    def to_frame(self) -> pd.DataFrame:
        ids = {"config": self.config, **self.keys, "contrast": self.contrast}
        return pd.DataFrame(
            [{**ids, "tag": tag, "value": value} for tag, value in self.records],
            columns=LONG_COLS)

    def _filename(self) -> str:
        keys = self.keys
        return f"{self.config}__{keys['experiment']}_{keys['subject']}_{keys['session']}__{self.contrast}.csv"

    def write(self) -> str:
        per_session_dir = join(self.out_dir, "per_session")
        os.makedirs(per_session_dir, exist_ok=True)
        path = join(per_session_dir, self._filename())
        self.to_frame().to_csv(path, index=False)
        return path
