"""load_events.py — per-session word_on event extraction.

For each session row, build the word_on event table (WORD + PRE_WORD copies)
under <root_dir>/word_on/events/. Called from prepare_sessions.run_get_events_worker
per session, then aggregated by prepare_sessions.stage_build_final_df.

The pyright rules below are narrowed to warnings for the pandas chains and
untyped match_events.MatchedEvents attributes.
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportAttributeAccessIssue=warning
from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
from os.path import join, exists as ex

import pandas as pd

from misc import get_dfrow, ftag
from project_paths import SCRATCH_DIR as _SCRATCH_DIR, LONGETAL
import helper
from helper import get_sr
from exclusion_log import ExclusionLog

root_dir: str = str(_SCRATCH_DIR)

NDArrayAny = npt.NDArray[Any]


def fix_event_cols(events: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant cols + coerce a few to int, so the events-JSON schema is
    stable across data sources."""
    for col in ['answer', 'exp_version', 'intrusion', 'is_stim', 'iscorrect', 'item_num', 'msoffset', 'recalled', 'stim_list', 'stim_params', 'test']:
        if col in events.columns:
            events = events.drop(col, axis=1)
    for col in ['session', 'list', 'trial', 'eegoffset', 'mstime']:
        if col in events.columns:
            events[col] = events[col].astype(int)
    return events


def add_pre_events(post_evs: pd.DataFrame, pre_type: str) -> tuple[pd.DataFrame, NDArrayAny]:
    """Append PRE-window copies of the post events for a pre/post physiological
    contrast. The copies share the post events' onsets but are relabeled
    (type=pre_type, e.g. PRE_WORD); the separate-load contrast pathway epochs
    them at the pre window. Returns (combined, mask) with mask True for the POST
    (original) events and False for the PRE copies.
    """
    pre_evs = post_evs.copy()
    pre_evs['type'] = pre_type
    combined = pd.concat([post_evs, pre_evs], ignore_index=True)
    mask = np.concatenate([np.ones(len(post_evs), dtype=bool),
                           np.zeros(len(pre_evs), dtype=bool)])
    return combined, mask





def get_events(dfrow: pd.Series | list[Any] | NDArrayAny) -> None:
    """Write `<root_dir>/word_on/events/<ftag>_events.json` (+ metadata) for one
    session: WORD events and their PRE_WORD copies (mask True = WORD)."""
    helper.root_dir = root_dir
    np.random.seed(202406)
    dfrow = get_dfrow(cast("list[Any]", list(dfrow)) if isinstance(dfrow, pd.Series) else dfrow)
    sr = get_sr(dfrow)

    import match_events
    matcher = match_events.MatchedEvents(dfrow, sr)
    beh = 'word_on'
    elog = ExclusionLog(dfrow, beh)
    all_words = matcher.events.query('type == "WORD"')
    elog.input(all_words)
    # first word of each list follows the countdown, not a blank screen
    word_evs = all_words.copy() if (LONGETAL or {}).get('include_first_word') else all_words.query('serialpos > 1').copy()
    elog.excluded('word_on_serialpos1', all_words, word_evs); elog.final(word_evs); elog.write()
    # later recalled; NaN if the session has no recalls
    recalled_by_mstime = matcher.study_events.set_index('mstime')['correct_recall'] if matcher.study_events is not None else {}
    word_evs['correct_recall'] = word_evs['mstime'].map(recalled_by_mstime).astype(float)
    word_evs, mask = add_pre_events(fix_event_cols(word_evs), 'PRE_WORD')
    metadata = pd.Series({'beh': beh, 'sr': sr, 'mask': mask, 'matching_successful': True})
    metadata.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events_metadata.json'))
    word_evs.to_json(join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json'))


def check_events(dfrow: pd.Series, beh: str) -> bool:
    """True if the events JSON for this session/behavior exists on disk."""
    name_tuple = cast("tuple[Any, ...]", dfrow.name)
    dfrow = get_dfrow(list(name_tuple))
    fname = join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json')
    return ex(fname)