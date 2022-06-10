import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.myconstants import *
from src.rolerep import RoleRep
from src.soccercpd import SoccerCPD

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)


class ContextualSoccerCPD:
    def __init__(self, match, traces, apply_cpd=True, formcpd_type='gseg_avg', rolecpd_type='gseg_avg'):
        self.match = match
        self.traces = traces
        self.offense_cpd = SoccerCPD(match, apply_cpd, formcpd_type, rolecpd_type)
        self.defense_cpd = SoccerCPD(match, apply_cpd, formcpd_type, rolecpd_type)
        self.fgp_df = None

    def smooth_team_poss(self, thres_dur=3):
        self.traces[LABEL_IN_POSS] = -1
        activity_id = self.match.record[LABEL_ACTIVITY_ID]

        for session in self.traces[LABEL_SESSION].unique():
            session_poss = self.traces[self.traces[LABEL_SESSION] == session]

            offenses = session_poss[session_poss[LABEL_TEAM_POSS] == activity_id]
            offense_ids = (offenses[LABEL_TIME].diff().round(1) > 0.1).astype(int).cumsum()
            session_poss.loc[offenses.index, LABEL_ID] = offense_ids.apply(lambda x: f'O{x:03d}')

            defenses = session_poss[session_poss[LABEL_TEAM_POSS] != activity_id]
            defense_ids = (defenses[LABEL_TIME].diff().round(1) > 0.1).astype(int).cumsum()
            session_poss.loc[defenses.index, LABEL_ID] = defense_ids.apply(lambda x: f'D{x:03d}')

            grouped = session_poss.groupby(LABEL_ID)
            start_times = grouped[LABEL_TIME].first().rename(LABEL_START_TIME) - 0.1
            end_times = grouped[LABEL_TIME].last().rename(LABEL_END_TIME)
            poss_records = pd.concat([start_times, end_times], axis=1).sort_values(LABEL_END_TIME)
            poss_records[LABEL_DURATION] = poss_records[LABEL_END_TIME] - poss_records[LABEL_START_TIME]

            poss_records = poss_records[poss_records[LABEL_DURATION] >= thres_dur].reset_index()

            poss_records[LABEL_END_TIME][:-1] = poss_records[LABEL_START_TIME][1:]
            session_end_time = self.traces.loc[self.traces[LABEL_SESSION] == session, LABEL_TIME].max()
            if poss_records[LABEL_END_TIME].iloc[-1] != session_end_time:
                poss_records[LABEL_END_TIME].iloc[-1] = session_end_time

            poss_records[LABEL_DURATION] = poss_records[LABEL_END_TIME] - poss_records[LABEL_START_TIME]
            poss_records[LABEL_IN_POSS] = poss_records[LABEL_ID].apply(lambda x: x.startswith('O')).astype(int)

            for i in poss_records.index:
                record = poss_records.loc[i]
                self.traces.loc[
                    (self.traces[LABEL_SESSION] == session) &
                    (self.traces[LABEL_TIME] > record[LABEL_START_TIME]) &
                    (self.traces[LABEL_TIME] <= record[LABEL_END_TIME]),
                    LABEL_IN_POSS
                ] = record[LABEL_IN_POSS]
        
        self.match.ugp_df = pd.merge(
            self.match.ugp_df, self.traces[[LABEL_DATETIME, LABEL_IN_POSS]],
            left_index=True, right_on=LABEL_DATETIME
        ).set_index(LABEL_DATETIME)

        self.offense_cpd.ugp_df = self.match.ugp_df[self.match.ugp_df[LABEL_IN_POSS] == 1]
        self.defense_cpd.ugp_df = self.match.ugp_df[self.match.ugp_df[LABEL_IN_POSS] == 0]
            
    def run(self):
        self.smooth_team_poss()

        print()
        print('********************** SoccerCPD for Offensive Instants **********************')
        print()
        self.offense_cpd.run(use_precomputed_fgp=False)
        offense_ugp_df = self.offense_cpd.ugp_df[HEADER_UGP].reset_index()
        offense_fgp_df = self.offense_cpd.fgp_df[HEADER_FGP[:6] + HEADER_FGP[-3:]]
        offense_fgp_df = pd.merge(offense_ugp_df, offense_fgp_df, how='left')
        offense_fgp_df = offense_fgp_df.sort_values([LABEL_PLAYER_ID, LABEL_DATETIME]).fillna(method='bfill')
        offense_fgp_df[LABEL_IN_POSS] = 1

        print()
        print('********************** SoccerCPD for Defensive Instants **********************')
        print()
        self.defense_cpd.run(use_precomputed_fgp=False)
        defense_ugp_df = self.defense_cpd.ugp_df[HEADER_UGP].reset_index()
        defense_fgp_df = self.defense_cpd.fgp_df[HEADER_FGP[:6] + HEADER_FGP[-3:]]
        defense_fgp_df = pd.merge(defense_ugp_df, defense_fgp_df, how='left')
        defense_fgp_df = defense_fgp_df.sort_values([LABEL_PLAYER_ID, LABEL_DATETIME]).fillna(method='bfill')
        defense_fgp_df[LABEL_IN_POSS] = 0

        self.fgp_df = pd.concat([offense_fgp_df, defense_fgp_df])
        self.fgp_df.sort_values([LABEL_PLAYER_ID, LABEL_DATETIME], ignore_index=True, inplace=True)
