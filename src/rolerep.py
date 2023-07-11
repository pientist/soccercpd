import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal

from src.match import Match
from src.myconstants import *
from src.record_manager import RecordManager

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)


# Frame-by-frame role assignment proposed by Bialkowski et al. (2014)
class RoleRep:
    def __init__(self, ugp_):
        self.ugp = ugp_
        self.fgp = None
        self.role_distns = None

    @staticmethod
    def normalize_locs(moment_fgp):
        locs = moment_fgp[[LABEL_X, LABEL_Y]]
        moment_fgp[[LABEL_X_NORM, LABEL_Y_NORM]] = locs - locs.mean()
        return moment_fgp

    @staticmethod
    def generate_fgp(ugp, freq):
        ugp = ugp[ugp[LABEL_X].notna()]
        fgp = []
        role = 1

        for player_id in ugp[LABEL_PLAYER_ID].unique():
            player_ugp = ugp[ugp[LABEL_PLAYER_ID] == player_id]
            resampler = player_ugp.resample(freq, closed="right", label="right")
            player_fgp = resampler[HEADER_FGP[:4]].last()
            player_fgp[LABEL_X] = resampler[LABEL_X].mean().round()
            player_fgp[LABEL_Y] = resampler[LABEL_Y].mean().round()
            player_fgp[LABEL_X_NORM] = np.nan
            player_fgp[LABEL_Y_NORM] = np.nan
            player_fgp[LABEL_FORM_PERIOD] = resampler[LABEL_FORM_PERIOD].last()
            player_fgp[LABEL_ROLE_PERIOD] = resampler[LABEL_ROLE_PERIOD].last()
            player_fgp[LABEL_ROLE] = role
            player_fgp[LABEL_BASE_ROLE] = role
            player_fgp[LABEL_SWITCH_RATE] = 0
            fgp.append(player_fgp[HEADER_FGP])
            role += 1

        fgp = pd.concat(fgp).reset_index().rename(columns={LABEL_INDEX: LABEL_DATETIME})
        return fgp.groupby(LABEL_DATETIME, group_keys=False).apply(RoleRep.normalize_locs)

    @staticmethod
    def estimate_mvn(df, col_x=LABEL_X_NORM, col_y=LABEL_Y_NORM, filter=True):
        if filter:
            coords = df[df[LABEL_SWITCH_RATE] <= MAX_SWITCH_RATE][[col_x, col_y]]
        else:
            coords = df[[col_x, col_y]]

        if filter and len(coords) < 30:
            return np.nan
        else:
            return multivariate_normal(coords.mean(), coords.cov())

    @staticmethod
    def update_params(fgp, by_phase=False):
        cols = [LABEL_PLAYER_PERIOD, LABEL_ROLE] if by_phase else [LABEL_ROLE]
        role_distns = fgp.groupby(cols).apply(RoleRep.estimate_mvn).reset_index()
        return role_distns.dropna().rename(columns={0: LABEL_DISTN})

    @staticmethod
    def align_formations(fgp, role_distns, label_group=LABEL_SESSION):
        groups = fgp[label_group].unique()
        base_group = groups[role_distns.groupby(label_group)[LABEL_ROLE].count().argmax()]
        base_role_distns = role_distns[role_distns[label_group] == base_group]

        for group in groups:
            if group == base_group:
                continue
            group_role_distns = role_distns[role_distns[label_group] == group]
            cost_mat = distance_matrix(
                group_role_distns[LABEL_DISTN].apply(lambda x: pd.Series(x.mean)).values,
                base_role_distns[LABEL_DISTN].apply(lambda x: pd.Series(x.mean)).values,
            )
            row_idx, col_idx = linear_sum_assignment(cost_mat)
            role_dict = dict(
                zip(group_role_distns[LABEL_ROLE].iloc[row_idx], base_role_distns[LABEL_ROLE].iloc[col_idx])
            )
            role_dict[0] = 0
            role_distns.loc[role_distns[label_group] == group, LABEL_ROLE] = col_idx + 1
            fgp.loc[fgp[label_group] == group, LABEL_ROLE] = fgp.loc[fgp[label_group] == group, LABEL_ROLE].apply(
                lambda role: role_dict[role]
            )
            fgp.loc[fgp[label_group] == group, LABEL_BASE_ROLE] = fgp.loc[
                fgp[label_group] == group, LABEL_BASE_ROLE
            ].apply(lambda role: role_dict[role])

        return fgp, role_distns.sort_values(by=[label_group, LABEL_ROLE]).reset_index(drop=True)

    def hungarian(self, moment_fgp, role_distns):
        cost_mat = moment_fgp[moment_fgp.columns[(len(HEADER_FGP) + 1) :]].values
        row_idx, col_idx = linear_sum_assignment(cost_mat)
        base_roles = moment_fgp[LABEL_BASE_ROLE].iloc[row_idx].values
        temp_roles = role_distns[LABEL_ROLE].iloc[col_idx].values
        self.fgp.loc[moment_fgp.index, LABEL_ROLE] = temp_roles
        self.fgp.loc[moment_fgp.index, LABEL_SWITCH_RATE] = (base_roles != temp_roles).sum() / len(row_idx)
        return cost_mat[row_idx, col_idx].mean()

    def run(self, freq="1S", verbose=True):
        temp_fgp = self.ugp.groupby(LABEL_PLAYER_PERIOD).apply(RoleRep.generate_fgp, freq=freq)
        temp_fgp = temp_fgp.reset_index(drop=True).dropna()
        temp_role_distns = RoleRep.update_params(temp_fgp, by_phase=True)
        temp_fgp = pd.merge(temp_fgp, temp_role_distns[[LABEL_PLAYER_PERIOD, LABEL_ROLE]])
        self.fgp, _ = RoleRep.align_formations(temp_fgp, temp_role_distns, LABEL_PLAYER_PERIOD)
        self.role_distns = RoleRep.update_params(self.fgp)

        max_iter = 10
        cost_prev = float("inf")
        tol = 0.005
        self.fgp.reset_index(drop=True, inplace=True)

        for i_iter in range(max_iter):
            cost_df = pd.DataFrame(
                self.role_distns[LABEL_DISTN]
                .apply(lambda n: pd.Series(-np.log(n.pdf(self.fgp[[LABEL_X_NORM, LABEL_Y_NORM]]))))
                .transpose()
                .values,
                index=self.fgp.index,
            )
            fgp_cost_df = pd.concat([self.fgp, cost_df], axis=1)
            costs = fgp_cost_df.groupby(LABEL_DATETIME).apply(self.hungarian, self.role_distns).mean()
            cost_new = costs.mean()
            if verbose:
                print("- Cost after iteration {0}: {1:.3f}".format(i_iter + 1, cost_new))
            self.role_distns = RoleRep.update_params(self.fgp)
            if cost_new + tol > cost_prev:
                if verbose:
                    print("Iteration finished since there are no significant changes.")
                    break
            cost_prev = cost_new

        session = self.ugp[LABEL_SESSION].iloc[0]
        self.role_distns[LABEL_SESSION] = session

        return self.fgp


# if __name__ == '__main__':
#     rm = RecordManager()
#     activity_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(DIR_UGP_DATA) if f.endswith('.ugp')]
#     activity_records = rm.activity_records[(rm.activity_records[LABEL_DATA_SAVED] == 1) &
#                                            (rm.activity_records[LABEL_STATS_SAVED] == 0)]
#     print()
#     print('Activity Records:')
#     print(activity_records)
#
#     for i in activity_records.index:
#         activity_id = activity_records.at[i, LABEL_ACTIVITY_ID]
#         date = activity_records.at[i, LABEL_DATE]
#         team_name = activity_records.at[i, LABEL_TEAM_NAME]
#         print()
#         print(f'[{i}] activity_id: {activity_id}, date: {date}, team_name: {team_name}')
#
#         activity_args = rm.load_activity_data(activity_id)
#         match = Match(*activity_args)
#         if match.player_periods[LABEL_PLAYER_IDS].iloc[1:].apply(len).max() >= 10:
#             match.construct_inplay_df()
#             match.rotate_pitch()
#
#             match.player_periods[LABEL_FORM_PERIOD] = match.player_periods[LABEL_SESSION]
#             match.player_periods[LABEL_ROLE_PERIOD] = match.player_periods[LABEL_SESSION]
#             match.ugp[LABEL_FORM_PERIOD] = match.ugp[LABEL_SESSION]
#             match.ugp[LABEL_ROLE_PERIOD] = match.ugp[LABEL_SESSION]
#             match_role_distns = pd.DataFrame(columns=HEADER_ROLE_RECORDS)
#
#             match_fgp = pd.DataFrame(columns=[LABEL_DATETIME] + HEADER_FGP)
#             for j in match.ugp[LABEL_ROLE_PERIOD].unique():
#                 form_ugp = match.ugp[match.ugp[LABEL_ROLE_PERIOD] == j]
#                 rolerep = RoleRep(form_ugp)
#                 print(f'\nRunning RoleRep for session {j}...')
#                 rolerep.run(freq='1S')
#                 match_role_distns = match_role_distns.append(rolerep.role_distns, sort=True)
#                 match_fgp = match_fgp.append(rolerep.fgp)
#
#             match_fgp_path = f'{DIR_DATA}/fgp_avg/{activity_id}.csv'
#             match_fgp.to_csv(match_fgp_path, index=False, encoding='utf-8-sig')
#             print(f"'{match_fgp_path}' saving done.")
#
#         else:
#             print('Not enough players to estimate a formation.')
#             continue
#
#         match_role_distns[LABEL_ACTIVITY_ID] = activity_id
#         match_role_distns = match_role_distns[[LABEL_ACTIVITY_ID] + HEADER_ROLE_RECORDS]
#         print()
#         print(match_role_distns)
#
#         # rm.activity_records.at[i, LABEL_STATS_SAVED] = 1
#         # rm.save_records(VARNAME_ACTIVITY_RECORDS)
