'''match_events.py — successful-vs-matched-unsuccessful event selection.

Builds, per session, the study/recall event tables and the time-matched event
sets behind the en / rm / ri contrasts. Events come from the session's BIDS
events.tsv, adapted in `MatchedEvents.load_events` to the CML event schema
(type / mstime / eegoffset / rectime / trial) the matching code was written for.

Last modified 06/11/2024 by Aditya Rao.
Adapted from a script created by Brandon Katerman, 2/24.
The find_recalls() function is Daniel Schonhaut's (I believe).
'''
# pandas method chains dominate; narrow stub noise.
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportAttributeAccessIssue=warning, reportMissingTypeStubs=none, reportUnknownLambdaType=warning, reportUnknownParameterType=warning, reportMissingParameterType=warning, reportOptionalMemberAccess=warning, reportOptionalSubscript=warning, reportOptionalIterable=warning, reportIndexIssue=warning, reportCallIssue=warning, reportPossiblyUnboundVariable=warning, reportUnusedVariable=warning, reportArgumentType=warning, reportAssignmentType=warning, reportReturnType=warning, reportGeneralTypeIssues=warning, reportOperatorIssue=warning
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from copy import deepcopy

import helper

# Value CML events carry in `rectime` / `serialpos` when the field does not apply.
NA_INT = -999


class MatchedEvents():
    
    def __init__(self, dfrow: Any, sr: float) -> None:
        
        self.dfrow = dfrow
        self.sub, self.exp, self.sess = dfrow
        self.sr = sr
        
        self.status = {}
        self.no_recalls = 0
        self.study_events = None
        self.all_recs = None
        self.no_matchable_events = {}
        
        self.default_status = {'matching_successful': False,
                               'no_successful': np.nan,
                               'no_unsuccessful': np.nan,
                               'no_successful_admissible': np.nan,
                               'no_unsuccessful_admissible': np.nan,
                               'no_matched': np.nan}
        
        self.load_events()
        self.find_recalls()
        
        self.matched_events = {}
        self.mask = {}

    def load_events(self) -> None:
        '''The session's BIDS events in the CML schema: `type` (trial_type),
        `mstime` (onset, ms), `eegoffset` (sample), `rectime` (response_time,
        ms, NA_INT where absent), `trial` (list), plus one constant `eegfile`
        (a BIDS session is a single recording).'''
        events = helper.bids_reader(self.dfrow, eeg=False).load_events()
        events = events.rename(columns={'trial_type': 'type', 'sample': 'eegoffset', 'list': 'trial'})
        events['mstime'] = (events['onset'] * 1000).round().astype(int)
        response_s = pd.to_numeric(events['response_time'], errors='coerce')
        events['rectime'] = np.where(response_s.notna(), (response_s * 1000).round(), NA_INT).astype(int)
        events['eegfile'] = f'{self.sub}_{self.exp}_{self.sess}'
        events['subject'], events['session'], events['experiment'] = self.sub, self.sess, self.exp
        events = events.sort_values(by=['mstime', 'eegoffset'], kind='stable').reset_index(drop=True)

        dfrow_key = (self.sub, self.exp, int(self.sess))
        bad_list_query = {('R1329T', 'FR1', 0): 'trial != 4',
                  ('R1341T', 'FR1', 1): 'trial != 8',
                  ('R1374T', 'FR1', 0): 'trial != 1',
                  ('R1488T', 'catFR1', 0): 'trial != 11',
                  ('TJ040', 'pyFR', 0): 'trial != 6',
                  ('FR060', 'pyFR', 1): 'trial != 1'}
        
        if dfrow_key in bad_list_query.keys(): events = events.query(bad_list_query[dfrow_key])
        
        self.events = events
        self.n_lists = np.max(events['trial'])

    def find_recalls(self) -> None:
        
        def flag_intrusions(row,
                        word_list,
                        intrusion_type):
            """Return whether word is an intrusion (in-list or out-of-list)."""
            if intrusion_type == 'pli':
                curr_words = np.unique(word_list.loc[(word_list.index==row['trial'])].values.ravel())
                if np.isin(row[item_col_name], curr_words):
                    return 0
                else:
                    prev_words = np.unique(word_list.loc[(word_list.index<row['trial'])].values.ravel())
                    return 1 * np.isin(row[item_col_name], prev_words)
            elif intrusion_type == 'oli':
                words_so_far = np.unique(word_list.loc[(word_list.index<=row['trial'])].values.ravel())
                return 0 if np.isin(row[item_col_name], words_so_far) else 1

        def flag_repeats(words):
            """Return whether word has already been said for the current list."""
            
            words = list(words)
            return [1 if word in words[:i] else 0 for i, word in enumerate(words)]

        all_recs = []

        events = self.events
        self.no_recalls = len(events.query('type == "REC_WORD"'))
        if self.no_recalls == 0: return

        keep_cols = ['subject', 'session', 'trial', 'mstime', 'eegoffset',
                 'eegfile', 'item', 'item_name', 'serialpos', 'rectime', 'type', 'experiment']
        events = events[events.columns[np.isin(events.columns, keep_cols)]]
        col_map = {'trial': 'list'}

        item_col_name = 'item_name' if 'item_name' in events.columns else 'item'

        study_events = events.query(f'(type=="WORD") & (0<trial<={self.n_lists})').reset_index(drop=True)
        study_events.drop(columns=['rectime', 'type'], inplace=True)
        word_list = pd.pivot(study_events, index='trial', columns='serialpos', values=item_col_name)

        rec_events = events.query(f'(type==["REC_WORD", "REC_WORD_VV"]) & (0<trial<={self.n_lists})')
        assert np.all(np.asarray([type(item) == str for item in events.query('type == "REC_WORD_VV"')[item_col_name].values]))
        rec_events = rec_events[np.asarray([type(item) == str for item in rec_events[item_col_name].values])]
        rec_events.reset_index(inplace=True,drop=True)
        
        study_events['correct_recall'] = study_events.apply(lambda r: bool(np.isin(r[item_col_name], 
                                    rec_events[rec_events['trial'] == r['trial']][item_col_name])), 
                                                          axis=1) 
        study_events.rename(columns=col_map, inplace=True)
        self.study_events = study_events
        
        rec_events.drop(columns=['serialpos'], inplace=True)
        rec_events['correct'] = rec_events.apply(lambda row: 1 * np.isin(row[item_col_name], word_list.loc[row['trial']]), axis=1)
        rec_events['pli'] = rec_events.apply(lambda row: flag_intrusions(row, word_list, 'pli'), axis=1)
        rec_events['oli'] = rec_events.apply(lambda row: flag_intrusions(row, word_list, 'oli'), axis=1)
        rec_events['repeat'] = np.concatenate(rec_events.groupby('trial')[item_col_name].apply(flag_repeats).values)
        
        rec_events['study_trial'] = -1
        rec_events.loc[rec_events['oli']==0, 'study_trial'] = (rec_events.loc[rec_events['oli']==0, item_col_name].apply(lambda item: word_list.iloc[np.where(word_list==item)].index[0]))
        rec_events['study_pos'] = -1
        rec_events.loc[rec_events['oli']==0, 'study_pos'] = (rec_events.loc[rec_events['oli']==0, item_col_name].apply(lambda item: word_list.iloc[np.where(word_list==item)].columns[0]))
        
        def get_rec_pos(words): 
            return [len(words.iloc[:(i+1)].query('type == "REC_WORD"')) for i, _ in words.reset_index().iterrows()]
        rec_events['rec_pos'] = np.concatenate(rec_events.groupby('trial').apply(get_rec_pos).tolist())

        rec_events['rectime_diff'] = rec_events['rectime'].diff()
        rec_events.loc[rec_events.eval('rec_pos == 1'), 'rectime_diff'] = rec_events.loc[rec_events.eval('rec_pos == 1'), 'rectime']

        all_recs.append(pd.concat([rec_events.reset_index(drop=True),
                                   events.query(f'(type=="REC_START") & (0<trial<={self.n_lists})')],
                                   axis=0).sort_values('mstime').reset_index(drop=True))


        all_recs = pd.concat(all_recs).reset_index(drop=True)
        all_recs.rename(columns=col_map, inplace=True)
        
        self.all_recs = all_recs

    def find_silence(
        self,
        recall_events: pd.DataFrame,
        silence_len: int,
        post_rec_distance: int,
        pre_rec_distance: int,
    ) -> pd.DataFrame:
        
        # REC_START is not a recall
        recall_events = recall_events.query('type != "REC_START"') 
        item_col_name = 'item_name' if 'item_name' in recall_events.columns else 'item'

        cols = ['list', 'eegoffset', 'eegfile' ,'mstime', 'rectime', 'type', item_col_name]
        start_df = pd.DataFrame([], columns = cols)
        stop_df = pd.DataFrame([], columns = cols)
        
        def set_search_limits(evs):
    
            evs = evs.query('type == "REC_WORD"')
            if len(evs) == 0: return (np.nan, np.nan)
            min_idx, max_idx = evs['rec_pos'].idxmin(), evs['rec_pos'].idxmax()
            return (evs.loc[min_idx, 'mstime'], evs.loc[max_idx, 'mstime'])

        search_limits = {list_no:(set_search_limits(recall_events[recall_events['list'] == list_no])) for list_no in recall_events['list'].unique()}

        for i in range(recall_events.shape[0]-1):
            
            silence_list = recall_events.iloc[i].list            
            eegfile = recall_events.iloc[i].eegfile
            if eegfile != recall_events.iloc[i+1].eegfile: continue
            if silence_list != recall_events.iloc[i+1].list: continue 

            silence_rec_start = recall_events.iloc[i].rectime + post_rec_distance
            silence_ms_start = recall_events.iloc[i].mstime + (post_rec_distance) 
            silence_eegoffset = recall_events.iloc[i].eegoffset + (post_rec_distance * (self.sr / 1000.))

            silence_rec_stop = recall_events.iloc[i+1].rectime - pre_rec_distance
            silence_ms_stop = recall_events.iloc[i+1].mstime - (pre_rec_distance)
            silence_eegoffset_stop = recall_events.iloc[i+1].eegoffset - (pre_rec_distance * (self.sr / 1000.)) 
            silence_eegoffset_len = silence_eegoffset_stop - silence_eegoffset
            silence_rectime_len = silence_rec_stop - silence_rec_start
            if (silence_rec_start < silence_rec_stop) and (silence_rectime_len >= silence_len) and (silence_ms_start >= search_limits[silence_list][0]) and (silence_ms_stop <= search_limits[silence_list][1]):
                
                assert silence_eegoffset < silence_eegoffset_stop, 'silence period start eegoffset is less than stop eegoffset' 

                # split long silences into silence_len chunks
                if silence_rectime_len >= (silence_len*2):

                    # same split count on all three clocks keeps them aligned
                    rec_time_range = np.arange(silence_rec_start, silence_rec_stop+1)
                    ms_time_range = np.arange(silence_ms_start, silence_ms_stop+1)
                    eegoffset_time_range = np.arange(silence_eegoffset, silence_eegoffset_stop+1)

                    num_splits = int(silence_rectime_len/silence_len)

                    rec_splits = np.array_split(rec_time_range, num_splits)
                    ms_splits = np.array_split(ms_time_range, num_splits)
                    eeg_splits = np.array_split(eegoffset_time_range, num_splits)

                    for n in range(num_splits):

                        silence_rec_start = min(rec_splits[n])
                        silence_rec_stop = max(rec_splits[n])

                        silence_ms_start = min(ms_splits[n])
                        silence_ms_stop = max(ms_splits[n])

                        silence_eegoffset = min(eeg_splits[n])
                        silence_eegoffset_stop = max(eeg_splits[n])

                        start_event = 'SILENCE_START'
                        stop_event = 'SILENCE_STOP'

                        start_event_df = pd.DataFrame([[silence_list, int(silence_eegoffset), eegfile, int(silence_ms_start), int(silence_rec_start), start_event, 'nan']], columns = cols) 
                        start_df = pd.concat([start_df, start_event_df], axis=0)

                        stop_event_df = pd.DataFrame([[silence_list ,int(silence_eegoffset_stop), eegfile, int(silence_ms_stop), int(silence_rec_stop), stop_event, 'nan']], columns = cols) 
                        stop_df = pd.concat([stop_df, stop_event_df], axis=0)

                elif (silence_len * 2) > silence_rectime_len >= silence_len :

                    start_event = 'SILENCE_START'
                    stop_event = 'SILENCE_STOP'

                    start_event_df = pd.DataFrame([[silence_list, int(silence_eegoffset), eegfile, int(silence_ms_start), int(silence_rec_start), start_event, 'nan']], columns = cols) 
                    start_df = pd.concat([start_df, start_event_df], axis=0)

                    stop_event_df = pd.DataFrame([[silence_list, int(silence_eegoffset_stop), eegfile, int(silence_ms_stop), int(silence_rec_stop), stop_event, 'nan']], columns = cols) 
                    stop_df = pd.concat([stop_df, stop_event_df], axis=0)

        for col in ['subject', 'session', 'experiment']:
            col_values = recall_events[col].unique()
            assert len(col_values) == 1
            start_df[col] = col_values[0]
            stop_df[col] = col_values[0]
            
        rec_starts = recall_events.query("type == 'REC_START'")
        rec_starts = rec_starts[['subject', 'session', 'list', 'eegoffset', 'eegfile', 'mstime', 'rectime', 'type', item_col_name, 'experiment']]

        silence_df = pd.concat([start_df, stop_df, rec_starts]).sort_values('mstime')  
        silence_df.reset_index(inplace = True, drop = True)  
        
        self.silence_df = silence_df

        return silence_df

    def find_time_matches(
        self,
        succ_events: pd.DataFrame,
        unsucc_events: pd.DataFrame,
        beh: str,
        proximity_buffer: int,
        list_thres: int = 25,
    ) -> Any:
        
        def find_adjacent_lists(current_list):

            n_lists = self.n_lists

            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx]

            all_lists = np.arange(1, n_lists+1)
            other_lists = [n for n in all_lists if n != current_list]

            # random flip so equidistant lists are searched back or forward first equally often
            if np.random.rand() > 0.5:
                other_lists = np.flip(other_lists)

            list_search_order = []

            for _ in range(len(other_lists)):
                nearest_list = find_nearest(other_lists, current_list)
                list_search_order.append(nearest_list)
                other_lists = [n for n in other_lists if n not in list_search_order]

            list_search_order = np.asarray(list_search_order)
            total_possible_lists = all_lists.shape[0] - 1
            total_found_lists = list_search_order.shape[0]
            assert total_possible_lists == total_found_lists, 'Total number of lists does not equal number of lists found'

            return list_search_order

        matches_per_event = []
        
        for _, choice_event in succ_events.iterrows():
            
            beh_to_time_col = {'en': 'serialpos',
                   'rm': 'rectime',
                   'ri': 'rectime'}
            time_col = beh_to_time_col[beh]
            
            choice_time = choice_event[time_col]

            choice_list = choice_event['list']
            list_pick_order = find_adjacent_lists(choice_list) 
            list_pick_order = [list_no for list_no in list_pick_order if abs(list_no-choice_list) < list_thres]
            
            beh_to_exact_match_distance = {'en': 0,
                                  'rm': 500,
                                  'ri': 500}
            exact_match_distance = beh_to_exact_match_distance[beh]

            event_matches = []

            for list_no in list_pick_order:

                list_unsucc_events = unsucc_events[unsucc_events['list'] == list_no]
                if beh == 'rm':
                    list_unsucc_events = list_unsucc_events.query("type == 'SILENCE_STOP'")

                # exact matches first
                for unsucc_event_time in list_unsucc_events[time_col]:

                    if np.abs(choice_time - unsucc_event_time) <= exact_match_distance:

                        event_matches.append(int(list_unsucc_events[list_unsucc_events[time_col] == unsucc_event_time].index.values)) 

                for unsucc_event_time in list_unsucc_events[time_col]:

                    if np.abs(choice_time - unsucc_event_time) <= proximity_buffer:

                        if (idx := int(list_unsucc_events[list_unsucc_events[time_col] == unsucc_event_time].index.values)) not in event_matches:

                            event_matches.append(idx)
            matches_per_event.append(event_matches)

        succ_events['match_index'] = matches_per_event
        succ_events['num_match'] = [len(m) for m in matches_per_event]
        
        # stable sort: reproducible pair selection on num_match ties (code-issues #65)
        sorted_df = succ_events.query('num_match > 0').sort_values(
            by='num_match', ascending=True, kind='stable')
    
        self.no_matchable_events[beh] = len(sorted_df)
        if self.no_matchable_events[beh] == 0: return
        
        sorted_df['matched'] = np.zeros(sorted_df.shape[0], dtype=int)
        sorted_df.reset_index(inplace=True, drop=True)

        self.sorted_df = sorted_df

        return sorted_df

    def Create_Matched_Events(
        self, time_matches: Any, unsucc_events: pd.DataFrame, beh: str
    ) -> pd.DataFrame:

        matched_events = pd.DataFrame([], columns = list(unsucc_events.columns))

        for _ in range(time_matches.shape[0]):

            time_matches = time_matches[time_matches['num_match'] != 0]
            if len(time_matches) == 0:
                continue

            match_index = time_matches.iloc[0].match_index[0]
            current_event = time_matches.iloc[0]

            matched_count = np.zeros(time_matches.shape[0])
            matched_count[0] += 1 

            matched_events = pd.concat([matched_events, pd.DataFrame(current_event[unsucc_events.columns]).T], axis=0)
            matched_events = pd.concat([matched_events, pd.DataFrame(unsucc_events.loc[match_index]).T], axis=0) #silence stops

            # drop the used match from every remaining event's candidates
            time_matches['match_index'] = time_matches.apply(lambda r: [m for m in r['match_index'] if m != match_index], axis=1)
            time_matches['num_match'] = time_matches.apply(lambda r: len(r['match_index']), axis=1)
            time_matches['matched'] = matched_count 

            time_matches = time_matches.query('(num_match != 0) & (matched != 1)')
            time_matches = time_matches.sort_values(by='num_match', ascending=True, kind='stable')

        matched_events.reset_index(inplace = True, drop = True)

        return matched_events

    def match_events(
        self,
        beh: str,
        proximity_buffer: int,
        event_count_threshold: int,
        rec_window: list[int] | None = None,
        post_rec_distance: int | None = None,
        pre_rec_distance: int | None = None,
    ) -> pd.DataFrame | None:

        self.status[beh] = deepcopy(self.default_status)
        self.mask[beh] = None
        self.no_matchable_events[beh] = 0

        if not self.no_recalls: return

        if beh == 'en':
            
            choice_succ_events = self.study_events.query('correct_recall==1')
            choice_unsucc_events = self.study_events.query('correct_recall==0')

            for col in ['no_successful', 'no_successful_admissible']:
                self.status[beh][col] = len(choice_succ_events)
            for col in ['no_unsuccessful', 'no_unsuccessful_admissible']:
                self.status[beh][col] = len(choice_unsucc_events)
                
            for col in ['no_successful_admissible', 'no_unsuccessful_admissible']:
                if self.status[beh][col] < event_count_threshold:
                    return
                
        elif beh in ['rm', 'ri']:
            
            for arg in ['rec_window', 'post_rec_distance', 'pre_rec_distance']:
                assert locals()[arg] is not None, f'{arg} has not been specified'
                
            def flag_contamination(r, rec_window, post_rec_distance):
    
                if r['rec_pos'] == 1: return True if r['rectime'] >= 1000 else False
                else: return True if r['rectime_diff'] >= np.abs(rec_window[0]) + post_rec_distance else False
    
            self.all_recs['contamination_free'] = self.all_recs.apply(lambda r: flag_contamination(r, rec_window, post_rec_distance), axis=1)
            
            correct_recall = self.all_recs.query('correct == 1')
            self.status[beh]['no_successful'] = len(correct_recall)

            inter_rec_min = abs(rec_window[0]) + post_rec_distance

            choice_succ_events = self.all_recs.query('(contamination_free == True) & (correct==1) & (repeat==0) & (rec_pos != 1)')
            self.status[beh]['no_successful_admissible'] = len(choice_succ_events)
            
            if self.status[beh]['no_successful_admissible'] < event_count_threshold:
                return

            if beh == 'rm':
                choice_unsucc_events = self.find_silence(self.all_recs, 
                           silence_len = np.diff(rec_window)[0],
                           post_rec_distance = post_rec_distance,
                           pre_rec_distance = pre_rec_distance)

                for col in ['no_unsuccessful', 'no_unsuccessful_admissible']:
                    self.status[beh][col] = len(choice_unsucc_events)
                
                if self.status[beh]['no_unsuccessful_admissible'] < event_count_threshold:
                    return

            elif beh == 'ri':
                unsucc_events = self.all_recs.query('type == "REC_WORD" and correct==0')
                self.status[beh]['no_unsuccessful'] = len(unsucc_events)

                choice_unsucc_events = self.all_recs.query('(contamination_free == True) & (type == "REC_WORD") & ((pli == 1) or (oli == 1)) & (repeat == 0) & (rec_pos != 1)') 
                self.status[beh]['no_unsuccessful_admissible'] = len(unsucc_events)
                
                if self.status[beh]['no_unsuccessful_admissible'] < event_count_threshold:
                    return

        time_matches = self.find_time_matches(choice_succ_events, choice_unsucc_events, beh, proximity_buffer)
        if self.no_matchable_events[beh] == 0:
            return

        matched_events = self.Create_Matched_Events(time_matches, choice_unsucc_events, beh)

        self.matched_events[beh] = matched_events
        self.status[beh]['no_matched'] = int(len(matched_events)/2)
        if self.status[beh]['no_matched'] < event_count_threshold: return
        self.status[beh]['matching_successful'] = True
        
        beh_to_mask = {'en': ('correct_recall', 1),
                       'rm': ('type', 'REC_WORD'),
                       'ri': ('correct', 1)}
        col, value = beh_to_mask[beh]
        mask = np.asarray(matched_events[col] == value)
        self.mask[beh] = mask
        
        return matched_events