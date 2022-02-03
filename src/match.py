import numpy as np
import pandas as pd
from datetime import datetime
from .myconstants import *
pd.options.mode.chained_assignment = None


# For match data preprocessing
class Match:
    def __init__(self, activity_record, player_periods, roster, ugp_df, pitch_size=(10800, 7200), outliers=None):
        self.record = activity_record
        self.player_periods = player_periods
        self.ugp_df = ugp_df
        self.roster = self._upgrade_roster(roster, outliers)
        self.pitch_size = pitch_size

    # Synchronize the player movement data with the official roster
    def _upgrade_roster(self, roster, outliers=None):
        if outliers is not None:
            self.ugp_df = self.ugp_df[~self.ugp_df[LABEL_PLAYER_ID].isin(outliers)]
            for period in self.player_periods.index:
                player_ids = self.player_periods.at[period, LABEL_PLAYER_IDS]
                self.player_periods.at[period, LABEL_PLAYER_IDS] = list(set(player_ids) - set(outliers))

        player_ids = []
        for player_id in roster.index:
            if not self.ugp_df[self.ugp_df[LABEL_PLAYER_ID] == player_id].empty:
                player_ids.append(player_id)
            if player_id not in self.player_periods.at[0, LABEL_PLAYER_IDS]:
                # Change 'player_name' to 0 for the player not in the official roster,
                # so that it can be filtered to be manually checked
                roster.at[player_id, LABEL_PLAYER_NAME] = 0
        roster = roster.loc[player_ids]

        if len(self.player_periods) > 1:
            for period in self.player_periods.index[1:]:
                roster[period] = 0
                for player_id in roster.index:
                    if player_id in self.player_periods.at[period, LABEL_PLAYER_IDS]:
                        roster.at[player_id, period] = 1
            return roster.sort_values(by=[1, LABEL_SQUAD_NUM], ascending=[False, True]).reset_index()
        else:
            return roster.sort_values(LABEL_SQUAD_NUM).reset_index()

    # Compute relative elapsed time in a session from unixtime
    @staticmethod
    def _compute_gametime(current_ut, start_ut):
        seconds_total = current_ut - start_ut
        minutes = int(seconds_total / SCALAR_TIME)
        seconds_rest = seconds_total % SCALAR_TIME
        return '{0:02d}:{1:04.1f}'.format(minutes, seconds_rest)

    # Filter in-play data from the measured data using the start, end, and substitution records
    def construct_inplay_df(self):
        freq = f'{self.ugp_df[LABEL_DURATION].iloc[1].round(1)}S'
        ugp_df_inplay = pd.DataFrame(columns=HEADER_UGP)

        for i in self.roster.index:
            # If a player in the official roster didn't actually played,
            # then exclude his/her entire data from the analysis
            if self.roster.at[i, LABEL_PLAYER_NAME] != 0 and self.roster.iloc[i, len(HEADER_ROSTER):].sum() == 0:
                continue

            player_id = self.roster.at[i, LABEL_PLAYER_ID]
            try:
                player_ugp_df = self.ugp_df[self.ugp_df[LABEL_PLAYER_ID] == player_id]
            except IndexError:
                continue
            else:
                player_ugp_df_inplay = pd.DataFrame(columns=HEADER_UGP)
                session_start_ut = 0
                for j in self.player_periods.index[1:]:
                    player_period = self.player_periods.loc[j]
                    start_dt = player_period[LABEL_START_DT]
                    end_dt = player_period[LABEL_END_DT]
                    dt_idx = pd.DataFrame(index=pd.date_range(start_dt, end_dt, freq=freq))[1:]
                    period_ugp_df = pd.merge(player_ugp_df, dt_idx, how='right', left_index=True, right_index=True)
                    period_ugp_df[LABEL_PLAYER_PERIOD] = j
                    period_ugp_df[LABEL_SESSION] = player_period[LABEL_SESSION]

                    if player_period[LABEL_TYPE].startswith('START'):
                        session_start_ut = (start_dt - datetime(1970, 1, 1)).total_seconds()
                    period_ugp_df[LABEL_UNIXTIME] = period_ugp_df.index.view(np.int64) // SCALAR_MICRO / SCALAR_MILLI
                    period_ugp_df[LABEL_GAMETIME] = period_ugp_df[LABEL_UNIXTIME].apply(
                        lambda x: self._compute_gametime(x, session_start_ut)
                    )
                    period_ugp_df[LABEL_DURATION] = float(freq[:-1])

                    # Remove the player's period data if he/she didn't actually play in that period,
                    # except for the players not in the official roster (to manually check the validity)
                    if player_id in self.player_periods.at[0, LABEL_PLAYER_IDS] and \
                            player_id not in self.player_periods.at[j, LABEL_PLAYER_IDS]:
                        period_ugp_df[HEADER_UGP[5:]] = np.nan
                    player_ugp_df_inplay = player_ugp_df_inplay.append(period_ugp_df[HEADER_UGP])

                player_ugp_df_inplay[LABEL_PLAYER_ID] = int(player_id)
                ugp_df_inplay = ugp_df_inplay.append(player_ugp_df_inplay)

        self.ugp_df = ugp_df_inplay

    # Rotate the pitch for one of the sessions so that the team always attacks from left to right
    def rotate_pitch(self):
        xlim = self.pitch_size[0]
        ylim = self.pitch_size[1]
        rotated = 2 - self.record[LABEL_ROTATED_SESSION]
        for session in self.player_periods[LABEL_SESSION].unique()[1:]:
            # If rotated == 0, rotate the even-numbered sessions (sessions with session % 2 == 0)
            # If rotated == 1, rotate the odd-numbered sessions (sessions with session % 2 == 1)
            if session % 2 == rotated:
                session_idx = self.ugp_df[LABEL_SESSION] == session
                self.ugp_df.loc[session_idx, LABEL_X] = xlim - self.ugp_df[LABEL_X].loc[session_idx]
                self.ugp_df.loc[session_idx, LABEL_Y] = ylim - self.ugp_df[LABEL_Y].loc[session_idx]
