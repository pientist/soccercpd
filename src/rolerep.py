import os
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from collections import Counter
from src.myconstants import *
from src.preprocessor import Preprocessor
from src.match import Match

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)

MIN_PHASE_SEC = 600
MAX_SR = 0.5


# Frame-by-frame role assignment algorithm proposed by Bialkowski et al. (2014)
class RoleRep:
    def __init__(self, ugp_df_):
        self.ugp_df = ugp_df_
        self.fgp_df = None
        self.role_records = None
        self.role_assigns = None

    @staticmethod
    def _normalize_locs(moment_fgp_df):
        locs = moment_fgp_df[[LABEL_X, LABEL_Y]]
        moment_fgp_df[[LABEL_X_NORM, LABEL_Y_NORM]] = locs - locs.mean()
        return moment_fgp_df

    @staticmethod
    def _generate_fgp(ugp_df, freq):
        ugp_df = ugp_df[ugp_df[LABEL_X].notna()]
        fgp_df = pd.DataFrame(columns=HEADER_FGP)
        role = 1

        for player_id in ugp_df[LABEL_PLAYER_ID].unique():
            player_ugp_df = ugp_df[ugp_df[LABEL_PLAYER_ID] == player_id]
            resampler = player_ugp_df.resample(freq, closed='right', label='right')
            player_fgp_df = resampler[HEADER_FGP[:4]].last()
            player_fgp_df[LABEL_X] = resampler[LABEL_X].mean().round()
            player_fgp_df[LABEL_Y] = resampler[LABEL_Y].mean().round()
            player_fgp_df[LABEL_X_NORM] = np.nan
            player_fgp_df[LABEL_Y_NORM] = np.nan
            player_fgp_df[LABEL_FORM_PERIOD] = resampler[LABEL_FORM_PERIOD].last()
            player_fgp_df[LABEL_ROLE_PERIOD] = resampler[LABEL_ROLE_PERIOD].last()
            player_fgp_df[LABEL_ROLE] = role
            player_fgp_df[LABEL_BASE_ROLE] = role
            player_fgp_df[LABEL_SWITCH_RATE] = 0
            fgp_df = fgp_df.append(player_fgp_df)
            role += 1

        fgp_df = fgp_df.reset_index().rename(columns={LABEL_INDEX: LABEL_DATETIME})
        return fgp_df.groupby(LABEL_DATETIME).apply(RoleRep._normalize_locs)

    @staticmethod
    def _estimate_mvn(fgp_df):
        valid_locs = fgp_df[fgp_df[LABEL_SWITCH_RATE] <= MAX_SR][[LABEL_X_NORM, LABEL_Y_NORM]]
        if len(valid_locs) < 30:
            return np.nan
        else:
            return multivariate_normal(valid_locs.mean(), valid_locs.cov())

    @staticmethod
    def update_params(fgp_df, by_phase=False):
        cols = [LABEL_PLAYER_PERIOD, LABEL_ROLE] if by_phase else [LABEL_ROLE]
        role_records = fgp_df.groupby(cols).apply(RoleRep._estimate_mvn).reset_index()
        return role_records.dropna().rename(columns={0: LABEL_DISTN})

    @staticmethod
    def align_formations(fgp_df, role_records, label_group=LABEL_SESSION):
        groups = fgp_df[label_group].unique()
        base_group = groups[role_records.groupby(label_group)[LABEL_ROLE].count().argmax()]
        base_role_records = role_records[role_records[label_group] == base_group]

        for group in groups:
            if group == base_group:
                continue
            group_role_records = role_records[role_records[label_group] == group]
            cost_mat = distance_matrix(
                group_role_records[LABEL_DISTN].apply(lambda x: pd.Series(x.mean)).values,
                base_role_records[LABEL_DISTN].apply(lambda x: pd.Series(x.mean)).values
            )
            row_idx, col_idx = linear_sum_assignment(cost_mat)
            role_dict = dict(zip(group_role_records[LABEL_ROLE].iloc[row_idx],
                                 base_role_records[LABEL_ROLE].iloc[col_idx]))
            role_dict[0] = 0
            role_records.loc[role_records[label_group] == group, LABEL_ROLE] = col_idx + 1
            fgp_df.loc[fgp_df[label_group] == group, LABEL_ROLE] = \
                fgp_df.loc[fgp_df[label_group] == group, LABEL_ROLE].apply(lambda role: role_dict[role])
            fgp_df.loc[fgp_df[label_group] == group, LABEL_BASE_ROLE] = \
                fgp_df.loc[fgp_df[label_group] == group, LABEL_BASE_ROLE].apply(lambda role: role_dict[role])

        return fgp_df, role_records.sort_values(by=[label_group, LABEL_ROLE]).reset_index(drop=True)

    def _hungarian(self, moment_fgp_df, role_records):
        cost_mat = moment_fgp_df[moment_fgp_df.columns[(len(HEADER_FGP) + 1):]].values
        row_idx, col_idx = linear_sum_assignment(cost_mat)
        base_roles = moment_fgp_df[LABEL_BASE_ROLE].iloc[row_idx].values
        temp_roles = role_records[LABEL_ROLE].iloc[col_idx].values
        self.fgp_df.loc[moment_fgp_df.index, LABEL_ROLE] = temp_roles
        self.fgp_df.loc[moment_fgp_df.index, LABEL_SWITCH_RATE] = (base_roles != temp_roles).sum() / len(row_idx)
        return cost_mat[row_idx, col_idx].mean()

    @staticmethod
    def _most_common(player_roles):
        try:
            counter = Counter(player_roles[player_roles.notna()])
            return counter.most_common(1)[0][0]
        except IndexError:
            return np.nan

    def run(self, freq='1S', verbose=True):
        temp_fgp_df = self.ugp_df.groupby(LABEL_PLAYER_PERIOD).apply(RoleRep._generate_fgp, freq=freq)
        temp_fgp_df = temp_fgp_df.reset_index(drop=True).dropna()
        temp_role_records = RoleRep.update_params(temp_fgp_df, by_phase=True)
        temp_fgp_df = pd.merge(temp_fgp_df, temp_role_records[[LABEL_PLAYER_PERIOD, LABEL_ROLE]])
        self.fgp_df, _ = RoleRep.align_formations(temp_fgp_df, temp_role_records, LABEL_PLAYER_PERIOD)
        self.role_records = RoleRep.update_params(self.fgp_df)

        max_iter = 10
        cost_prev = float('inf')
        tol = 0.005
        self.fgp_df.reset_index(drop=True, inplace=True)

        for i_iter in range(max_iter):
            cost_df = pd.DataFrame(self.role_records[LABEL_DISTN].apply(
                lambda n: pd.Series(-np.log(n.pdf(self.fgp_df[[LABEL_X_NORM, LABEL_Y_NORM]])))
            ).transpose().values, index=self.fgp_df.index)
            fgp_cost_df = pd.concat([self.fgp_df, cost_df], axis=1)
            costs = fgp_cost_df.groupby(LABEL_DATETIME).apply(self._hungarian, self.role_records).mean()
            cost_new = costs.mean()
            if verbose:
                print('- Cost after iteration {0}: {1:.3f}'.format(i_iter + 1, cost_new))
            self.role_records = self.update_params(self.fgp_df)
            if cost_new + tol > cost_prev:
                if verbose:
                    print('Iteration finished since there are no significant changes.')
                    break
            cost_prev = cost_new

        session = self.ugp_df[LABEL_SESSION].iloc[0]
        self.role_records[LABEL_SESSION] = session

    # def summarize(self):
        # roster_by_phase = self.fgp_df[HEADER_ROLE_ASSIGNS[:4]].drop_duplicates()
        # valid_fgp_df = self.fgp_df[self.fgp_df[LABEL_SWITCH_RATE] <= MAX_SR]
        # role_assigns = valid_fgp_df.groupby(LABEL_PLAYER_ID)[LABEL_ROLE].apply(self._most_common)
        # role_assigns = role_assigns.dropna().reset_index().rename(columns={0: LABEL_ROLE}).astype(int)
        # self.role_assigns = pd.merge(roster_by_phase, role_assigns).sort_values(LABEL_PHASE, ignore_index=True)


if __name__ == '__main__':
    pp = Preprocessor()
    activity_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(DIR_UGP_DATA) if f.endswith('.ugp')]
    activity_records = pp.activity_records[
        (pp.activity_records[LABEL_DATA_SAVED] == 1) &
        (pp.activity_records[LABEL_STATS_SAVED] == 0)
    ]
    print()
    print('Activity Records:')
    print(activity_records)

    for i in activity_records.index:
        activity_id = activity_records.at[i, LABEL_ACTIVITY_ID]
        date = activity_records.at[i, LABEL_DATE]
        team_name = activity_records.at[i, LABEL_TEAM_NAME]
        print()
        print(f'[{i}] activity_id: {activity_id}, date: {date}, team_name: {team_name}')

        activity_args = pp.load_activity_data(activity_id)
        match = Match(*activity_args)
        if match.player_periods[LABEL_PLAYER_IDS].iloc[1:].apply(len).max() >= 10:
            match.construct_inplay_df()
            match.rotate_pitch()

            match.player_periods[LABEL_FORM_PERIOD] = match.player_periods[LABEL_SESSION]
            match.player_periods[LABEL_ROLE_PERIOD] = match.player_periods[LABEL_SESSION]
            match.ugp_df[LABEL_FORM_PERIOD] = match.ugp_df[LABEL_SESSION]
            match.ugp_df[LABEL_ROLE_PERIOD] = match.ugp_df[LABEL_SESSION]
            match_role_records = pd.DataFrame(columns=HEADER_ROLE_RECORDS)
            # match_role_assigns = pd.DataFrame(columns=HEADER_ROLE_ASSIGNS)

            match_fgp_df = pd.DataFrame(columns=[LABEL_DATETIME] + HEADER_FGP)
            for j in match.ugp_df[LABEL_ROLE_PERIOD].unique():
                form_ugp_df = match.ugp_df[match.ugp_df[LABEL_ROLE_PERIOD] == j]
                rolerep = RoleRep(form_ugp_df)
                print(f'\nRunning RoleRep for session {j}...')
                rolerep.run(freq='1S')
                # match_role_records = match_role_records.append(rolerep.role_records, sort=True)
                # match_role_assigns = match_role_assigns.append(rolerep.role_assigns, sort=True)
                match_fgp_df = match_fgp_df.append(rolerep.fgp_df)

            match_fgp_path = f'{DIR_DATA}/fgp_avg/{activity_id}.csv'
            match_fgp_df.to_csv(match_fgp_path, index=False, encoding='utf-8-sig')
            print(f"'{match_fgp_path}' saving done.")

        else:
            print('Not enough players to estimate a formation.')
            continue

        # match_role_records[LABEL_ACTIVITY_ID] = activity_id
        # match_role_assigns[LABEL_ACTIVITY_ID] = activity_id
        # match_role_assigns = pd.merge(match.roster[HEADER_ROSTER], match_role_assigns)
        # match_role_assigns.sort_values([LABEL_SUBPHASE, LABEL_PHASE], inplace=True, ignore_index=True)
        # print()
        # print(match_role_records[[LABEL_ACTIVITY_ID] + HEADER_ROLE_RECORDS])
        # print(match_role_assigns[FULL_HEADER_ROLE_ASSIGNS])

        # pp.activity_records.at[i, LABEL_STATS_SAVED] = 1
        # pp.save_records(VARNAME_ACTIVITY_RECORDS)
