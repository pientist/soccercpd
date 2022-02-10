import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.spatial import Delaunay
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from collections import Counter
from pprint import pprint

import rpy2.robjects as robjects
import rpy2.rinterface_lib.embedded as rembedded
import ruptures as rpt

from src.myconstants import *
from src.rolerep import RoleRep

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)


# Formation and role change-point detection (main algorithm)
class SoccerCPD:
    def __init__(
        self, match, apply_cpd=True, formcpd_type='gseg_avg', rolecpd_type='gseg_avg',
        max_sr=MAX_SWITCH_RATE, max_pval=MAX_PVAL, min_pdur=MIN_PERIOD_DUR, min_fdist=MIN_FORM_DIST
    ):
        self.apply_cpd = apply_cpd
        self.formcpd_type = formcpd_type
        self.rolecpd_type = rolecpd_type
        # Available FormCPD types: 'gseg_avg', 'gseg_union', 'kernel_linear', 'kernel_rbf', 'kernel_cosine', 'rank'
        # Available RoleCPD types: 'gseg_avg', 'gseg_union'

        self.max_sr = max_sr
        self.max_pval = max_pval
        self.min_pdur = min_pdur
        self.min_fdist = min_fdist

        self.match = match
        self.activity_id = self.match.record[LABEL_ACTIVITY_ID]
        self.player_periods = self.match.player_periods
        self.ugp_df = self.match.ugp_df

        self.fgp_df = pd.DataFrame(columns=[LABEL_DATETIME] + HEADER_FGP)
        self.form_periods = pd.DataFrame(columns=HEADER_FORM_PERIODS)
        self.role_periods = pd.DataFrame(columns=HEADER_ROLE_PERIODS)
        self.role_records = None

        self.target_dir = f'{DIR_DATA}/{formcpd_type}' if apply_cpd else f'{DIR_DATA}/noncpd'

    # Apply Delaunay triangulation to the given player coordinates to obtain the role-adjacency matrix
    @staticmethod
    def delaunay_edge_mat(coords):
        tri_pts = Delaunay(coords).simplices
        edges = np.concatenate((tri_pts[:, :2], tri_pts[:, 1:], tri_pts[:, ::2]), axis=0)
        edge_mat = np.zeros((coords.shape[0], coords.shape[0]))
        edge_mat[edges[:, 0], edges[:, 1]] = 1
        return np.clip(edge_mat + edge_mat.T, 0, 1)

    @staticmethod
    def complete_perm(perm, role_set):
        if perm.isnull().sum():
            return perm.fillna(list(role_set - set(perm.dropna()))[0])
        else:
            return perm

    @staticmethod
    def hamming(perm1, perm2):
        return (perm1 != perm2).astype(int).sum()

    @staticmethod
    def manhattan(mat1, mat2):
        return np.abs(mat1 - mat2).sum()

    # Recursive change-point detection for the input sequence
    def detect_change_times(self, input_seq, sub_dts, mode='form'):
        # if mode == 'form' (FormCPD), the input is a sequence of role-adjacency matrices
        # if mode == 'role' (RoleCPD), the input a sequence of role permutations

        start_time = input_seq.index[0].time()
        end_time = input_seq.index[-1].time()

        if (mode == 'role') or ('gseg' in self.formcpd_type):
            metric = SoccerCPD.manhattan if mode == 'form' else SoccerCPD.hamming
            dists = pd.DataFrame(pairwise_distances(input_seq.drop_duplicates(), metric=metric))

            # Save the input sequence and the pairwise distances so that we can use them in the R script below
            if not os.path.exists(DIR_TEMP_DATA):
                os.mkdir(DIR_TEMP_DATA)
            input_seq.to_csv(f'{DIR_TEMP_DATA}/{self.activity_id}_temp_seq.csv', index=False)
            dists.to_csv(f'{DIR_TEMP_DATA}/{self.activity_id}_temp_dists.csv', index=False)

            try:
                print(f"Applying g-segmentation to the sequence between {start_time} and {end_time}...")

                if mode == 'form':
                    gseg_type = self.formcpd_type.split('_')[1][0]
                else:
                    gseg_type = self.rolecpd_type.split('_')[1][0]

                # Run the R function 'gseg1_discrete' to find a change-point
                # rpackages.importr('gSeg', lib_loc=rpackages.importr('base')._libPaths()[0])
                robjects.r(f'''
                    dir = '{DIR_TEMP_DATA}'
                    seq_path = paste(dir, '{self.activity_id}_temp_seq.csv', sep='/')
                    seq = read.csv(seq_path)
                    dists_path = paste(dir, '{self.activity_id}_temp_dists.csv', sep='/')
                    dists = read.csv(dists_path)

                    n = dim(seq)[1]
                    edge_mat = nnl(dists, 1)
                    seq_str = do.call(paste, seq)
                    ids = match(seq_str, unique(seq_str))
                    output = gseg1_discrete(n, edge_mat, ids, statistics='generalized', n0=0.1*n, n1=0.9*n)
                    
                    chg_idx = output$scanZ$generalized$tauhat_{gseg_type}
                    pval = output$pval.appr$generalized_{gseg_type}
                ''')

            except rembedded.RRuntimeError:
                return []

            # Check whether the detected change-point is significant, using the following three conditions
            # Condition (1): The p-value of the scan statistic must be less than 0.1
            if robjects.r['pval'][0] >= self.max_pval:
                print('Change-point insignificant: The p-value is not small enough.\n')
                return []
            else:
                chg_idx = robjects.r['chg_idx'][0]

        elif 'kernel' in self.formcpd_type:
            print(f"Applying kernel-based CPD to the sequence between {start_time} and {end_time}...")
            kernel_type = self.formcpd_type.split('_')[1]
            algo = rpt.Binseg(model=kernel_type).fit(input_seq.values)
            chg_idx = algo.predict(n_bkps=1)[0]
        
        elif 'rank' in self.formcpd_type:
            print(f"Applying rank-based CPD to the sequence between {start_time} and {end_time}...")
            algo = rpt.Binseg(model='rank').fit(input_seq.values)
            chg_idx = algo.predict(n_bkps=1)[0]

        else:
            raise ValueError('Invalid formcpd_type.')

        chg_dt = input_seq.index[chg_idx]

        # Fine-tune chg_dt to the closest substitution time (if exists)
        if len(sub_dts) > 0:
            tds = np.abs(sub_dts - chg_dt.to_pydatetime())
            if tds.min().total_seconds() <= 180:
                chg_dt = sub_dts[tds.argmin()]

        # Condition (2): Both of the segments must last for at least five minutes
        seq1 = input_seq[:chg_dt]
        seq2 = input_seq[chg_dt:]
        if (len(seq1) < self.min_pdur) or (len(seq2) < self.min_pdur):
            print('Change-point insignificant: One of the periods has not enough duration.\n')
            return []

        if mode == 'form':
            # Condition (3) for FormCPD: The respective mean role-adjacency matrices
            # from the segments before and after chg_dt are far enough from each other
            form1_edge_mat = seq1.mean(axis=0).values
            form2_edge_mat = seq2.mean(axis=0).values
            if self.manhattan(form1_edge_mat, form2_edge_mat) < self.min_fdist:
                print('Change-point insignificant: The formation is not changed.\n')
                return []
            else:
                # If significant, recursively detect another change-points before and after chg_dt
                print(f'A significant fine-tuned change-point at {chg_dt.time()}.\n')
                prev_chg_dts = self.detect_change_times(seq1, sub_dts)
                next_chg_dts = self.detect_change_times(seq2, sub_dts)
                return prev_chg_dts + [chg_dt] + next_chg_dts

        elif mode == 'role':
            # Condition (3) for RoleCPD: The most frequent permutations differ between before and after chg_dt
            seq1_str = seq1.apply(lambda row: np.array2string(row.values), axis=1)
            seq2_str = seq2.apply(lambda row: np.array2string(row.values), axis=1)
            counter1 = Counter(seq1_str)
            counter2 = Counter(seq2_str)
            if counter1.most_common(1)[0][0] == counter2.most_common(1)[0][0]:
                print('Change-point insignificant: The most frequent permutation is not changed.\n')
                return []
            else:
                # If significant, recursively detect another change-points before and after chg_dt
                print(f'A significant fine-tuned change-point at {chg_dt.time()}.')
                print(f'- Frequent permutations before {chg_dt.time()}:')
                pprint(counter1.most_common(5))
                print(f'- Frequent permutations after {chg_dt.time()}:')
                pprint(counter2.most_common(5))
                print()
                prev_chg_dts = self.detect_change_times(seq1, sub_dts)
                next_chg_dts = self.detect_change_times(seq2, sub_dts)
                return prev_chg_dts + [chg_dt] + next_chg_dts

        else:
            raise ValueError('Invalid mode')

    # Align corresponding roles from different formation periods
    @staticmethod
    def align_formations(fgp_df, form_period_records):
        base_form_period_record = form_period_records.iloc[0]

        for form_period in form_period_records.index[1:]:
            cur_form_period_record = form_period_records.loc[form_period]
            cost_mat = distance_matrix(
                base_form_period_record[LABEL_COORDS],
                cur_form_period_record[LABEL_COORDS]
            )
            _, perm = linear_sum_assignment(cost_mat)
            form_period_records.at[form_period, LABEL_COORDS] = cur_form_period_record[LABEL_COORDS][perm]
            form_period_records.at[form_period, LABEL_EDGE_MAT] = cur_form_period_record[LABEL_EDGE_MAT][perm][:, perm]

            inverse_perm = dict(zip(np.array(perm) + 1, np.arange(10) + 1))
            cur_fgp_df = fgp_df[fgp_df[LABEL_FORM_PERIOD] == form_period]
            for col in [LABEL_ROLE, LABEL_BASE_ROLE]:
                fgp_df.loc[cur_fgp_df.index, col] = cur_fgp_df[col].apply(lambda role: inverse_perm[role])

        return fgp_df, form_period_records

    @staticmethod
    def most_common(player_roles):
        try:
            counter = Counter(player_roles[player_roles.notna()])
            return counter.most_common(1)[0][0]
        except IndexError:
            return np.nan

    def reassign_base_role(self, fgp_row):
        base_perm = self.role_periods.at[fgp_row[LABEL_ROLE_PERIOD], LABEL_BASE_PERM]
        fgp_row[LABEL_BASE_ROLE] = base_perm[fgp_row[LABEL_BASE_ROLE]]
        return fgp_row

    # Recompute the 'switch rate' of a frame by the Hamming distance
    # between the temporary roles and the instructed roles
    @staticmethod
    def recompute_switch_rate(moment_fgp_df):
        hamming = SoccerCPD.hamming(moment_fgp_df[LABEL_ROLE], moment_fgp_df[LABEL_BASE_ROLE])
        moment_fgp_df[LABEL_SWITCH_RATE] = hamming / len(moment_fgp_df)
        return moment_fgp_df

    # Refind base roles per player period and recompute switch rate per frame for the precomputed FGP data
    def reset_precomputed_fgp(self):
        for i in self.player_periods.index[1:]:
            fgp_df = self.fgp_df[self.fgp_df[LABEL_PLAYER_PERIOD] == i]
            perms = fgp_df.pivot_table(LABEL_ROLE, LABEL_DATETIME, LABEL_PLAYER_ID, 'first')
            if perms.empty:
                continue
            # role_set = set(perms.dropna().iloc[0])
            role_set = set(np.arange(10) + 1)
            perms = perms.apply(SoccerCPD.complete_perm, axis=1, args=(role_set,)).astype(int)
            perms_str = perms.apply(lambda perm: np.array2string(perm.values), axis=1)
            base_perm_list = np.fromstring(SoccerCPD.most_common(perms_str)[1:-1], dtype='float32', sep=' ')
            base_perm_dict = dict(zip(perms.columns, base_perm_list))
            self.fgp_df.loc[fgp_df.index, LABEL_BASE_ROLE] = fgp_df[LABEL_PLAYER_ID].apply(lambda x: base_perm_dict[x])
        self.fgp_df = self.fgp_df.groupby(LABEL_DATETIME).apply(SoccerCPD.recompute_switch_rate)

    def generate_role_records(self):
        grouped = self.fgp_df.groupby([LABEL_PLAYER_ID, LABEL_ROLE_PERIOD], as_index=False)
        role_records = grouped[[LABEL_PLAYER_PERIOD, LABEL_BASE_ROLE]].first()
        role_records = pd.merge(role_records, self.role_periods[HEADER_ROLE_PERIODS[:-1]])

        role_records = pd.merge(role_records, self.form_periods[[LABEL_FORM_PERIOD, LABEL_COORDS]])
        role_records[LABEL_X] = role_records.apply(lambda x: x[LABEL_COORDS][x[LABEL_BASE_ROLE]-1, 0], axis=1)
        role_records[LABEL_Y] = role_records.apply(lambda x: x[LABEL_COORDS][x[LABEL_BASE_ROLE]-1, 1], axis=1)

        role_records = pd.merge(role_records, self.match.roster[HEADER_ROSTER])
        return role_records[HEADER_ROLE_RECORDS].astype({LABEL_PLAYER_PERIOD: int})

    def run(self, use_precomputed_fgp=True, freq='1S'):
        fgp_path = f'data/{self.formcpd_type}/fgp/{self.activity_id}.csv'

        # If self.use_precomputed_fgp == True, load and initialize the precomputed FGP data
        if use_precomputed_fgp and os.path.exists(fgp_path):
            self.fgp_df = pd.read_csv(fgp_path, header=0, encoding='utf-8-sig')
            self.fgp_df[LABEL_DATETIME] = self.fgp_df[LABEL_DATETIME].apply(
                lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            )
            self.reset_precomputed_fgp()

        # Initialize formation and role period labels by the session labels
        self.ugp_df[LABEL_FORM_PERIOD] = self.ugp_df[LABEL_SESSION]
        self.ugp_df[LABEL_ROLE_PERIOD] = self.ugp_df[LABEL_SESSION]

        fgp_list = []
        perm_list = []

        for session in self.ugp_df[LABEL_SESSION].unique():
            print(f"\n{'-' * 33} Session {session} {'-' * 34}")
            player_periods = self.player_periods[self.player_periods[LABEL_SESSION] == session]
            session_start_dt = pd.to_datetime(player_periods[LABEL_START_DT].iloc[0])
            session_end_dt = pd.to_datetime(player_periods[LABEL_END_DT].iloc[-1])
            ugp_df = self.ugp_df[self.ugp_df[LABEL_SESSION] == session]

            if ugp_df[ugp_df[LABEL_X].notna()].groupby(LABEL_UNIXTIME)[LABEL_PLAYER_ID].apply(len).max() < 10:
                # If less than 10 players have been measured during the session, skip the process
                print('Not enough players to estimate a formation.')
                continue
            else:
                print(player_periods[HEADER_PLAYER_PERIODS[2:7]])

            if use_precomputed_fgp and not self.fgp_df.empty:
                print("\n* Step 1: Load the pre-computed role assignment result")
                fgp_df = self.fgp_df[self.fgp_df[LABEL_SESSION] == session]
                print(f"Session FGP data loaded and filtered from '{fgp_path}'.")
            else:
                print("\n* Step 1: Frame-by-frame role assignment using RoleRep")
                rolerep = RoleRep(ugp_df)
                fgp_df = rolerep.run(freq=freq)
            
            # Exclude situations such as set-pieces that are irrelevant to the team formation
            valid_fgp_df = fgp_df[fgp_df[LABEL_SWITCH_RATE] <= self.max_sr]

            # Check whether all the 10 outfield players are measured for some periods
            role_x = valid_fgp_df.pivot_table(LABEL_X_NORM, LABEL_DATETIME, LABEL_ROLE, aggfunc='first')
            role_y = valid_fgp_df.pivot_table(LABEL_Y_NORM, LABEL_DATETIME, LABEL_ROLE, aggfunc='first')
            role_coords = np.dstack([role_x.dropna().values, role_y.dropna().values])
            if role_coords.shape[1] < 10:
                print('Not enough players to estimate a formation.')
                continue
            else:
                fgp_list.append(fgp_df)

            # Generate the sequence of role-adjacency matrices
            edge_mats = []
            for coords in role_coords:
                edge_mats.append(SoccerCPD.delaunay_edge_mat(coords).reshape(-1))
            edge_mats = pd.DataFrame(np.stack(edge_mats, axis=0), index=role_x.dropna().index)

            if self.apply_cpd:
                print("\n* Step 2: FormCPD based on role-adjacency matrices")
                # Recursive change-point detection for the matrix sequence
                sub_dts = pd.to_datetime(
                    player_periods.loc[player_periods[LABEL_TYPE].isin(['SUB', 'RED']), LABEL_START_DT].values
                )
                form_chg_dts = self.detect_change_times(edge_mats, sub_dts, mode='form')

                print("Detected formation change-points (rounded off to the nearest 10 second mark):")
                form_chg_dts_rounded = []
                for dt in form_chg_dts:
                    form_chg_dts_rounded.append(dt - timedelta(seconds=dt.second % 10))
                pprint(form_chg_dts_rounded)
            
                print("\n* Step 3: RoleCPD per formation period based on role permutations")
                form_chg_dts = [session_start_dt] + form_chg_dts_rounded + [session_end_dt]

            else:
                print("\n* Step 2: Compute the formation graph of the session")
                # Assume there are no formation change throughout the session
                form_chg_dts = [session_start_dt, session_end_dt]

                print("\n* Step 3: Find the most frequent role permutation per 5-minute segment")

            # Generate the sequence of role permutations
            perms = valid_fgp_df.pivot_table(LABEL_BASE_ROLE, LABEL_DATETIME, LABEL_ROLE, aggfunc='first')
            role_set = set(perms.dropna().iloc[0])
            perms = perms.apply(SoccerCPD.complete_perm, axis=1, args=(role_set,)).astype(int)
            perms_str = perms.apply(lambda perm: np.array2string(perm.values), axis=1)
            perm_list.append(perms_str.rename(LABEL_PERM).to_frame())
            
            for form_chg_idx in range(1, len(form_chg_dts)):
                # form_period = len(self.form_periods) + 1
                form_period = session

                form_start_dt = form_chg_dts[form_chg_idx - 1]
                form_end_dt = form_chg_dts[form_chg_idx]

                mean_x = role_x[form_start_dt:form_end_dt].dropna().mean(axis=0).round().values
                mean_y = role_y[form_start_dt:form_end_dt].dropna().mean(axis=0).round().values
                mean_coords = np.stack([mean_x, mean_y]).T
                mean_edge_mat = edge_mats[form_start_dt:form_end_dt].mean(axis=0).round(3).values

                # Recording the details of the formation period
                self.form_periods = self.form_periods.append({
                    LABEL_ACTIVITY_ID: self.activity_id,
                    LABEL_SESSION: session,
                    LABEL_FORM_PERIOD: form_period,
                    LABEL_START_DT: form_start_dt,
                    LABEL_END_DT: form_end_dt,
                    LABEL_DURATION: (form_end_dt - form_start_dt).total_seconds(),
                    LABEL_COORDS: mean_coords,
                    LABEL_EDGE_MAT: mean_edge_mat.reshape(10, 10)
                }, ignore_index=True)

                if self.apply_cpd:
                    # Recursive change-point detection for the permutation sequence
                    print(f"\nRoleCPD for the formation period {form_period}:")
                    input_perms = perms[form_start_dt:form_end_dt]
                    input_sub_dts = np.array([dt for dt in sub_dts if (dt >= form_start_dt) and (dt < form_end_dt)])
                    role_chg_dts = self.detect_change_times(input_perms, input_sub_dts, mode='role')

                    print("Detected role change-points (rounded off to the nearset 10 second mark):")
                    role_chg_dts_rounded = []
                    for dt in role_chg_dts:
                        role_chg_dts_rounded.append(dt - timedelta(seconds=dt.second % 10))
                    pprint(role_chg_dts_rounded)

                    role_chg_dts = [form_start_dt, form_end_dt] + input_sub_dts.tolist()
                    role_chg_dts = list(set(role_chg_dts) | set(pd.to_datetime(role_chg_dts_rounded)))
                    role_chg_dts.sort()

                    for role_chg_idx in range(1, len(role_chg_dts)):
                        role_period = len(self.role_periods) + 1
                        role_start_dt = role_chg_dts[role_chg_idx - 1]
                        role_end_dt = role_chg_dts[role_chg_idx]

                        # Set the instructed roles per player by the most frequent permutation in the role period
                        counter = Counter(perms_str[role_start_dt:role_end_dt])
                        base_perm_list = np.fromstring(counter.most_common(1)[0][0][1:-1], dtype=int, sep=' ')
                        base_perm_dict = dict(zip(perms.columns, base_perm_list))

                        # Recording the details of the role period
                        self.role_periods = self.role_periods.append({
                            LABEL_ACTIVITY_ID: self.activity_id,
                            LABEL_SESSION: session,
                            LABEL_FORM_PERIOD: form_period,
                            LABEL_ROLE_PERIOD: role_period,
                            LABEL_START_DT: role_start_dt,
                            LABEL_END_DT: role_end_dt,
                            LABEL_DURATION: (role_end_dt - role_start_dt).total_seconds(),
                            LABEL_BASE_PERM: base_perm_dict
                        }, ignore_index=True)

        if fgp_list:
            self.fgp_df = pd.concat(fgp_list, ignore_index=True)
        else:
            return

        if not self.apply_cpd:
            # Finding the most frequent role permutation per 5-minute segment

            perms_str = pd.concat(perm_list)
            bins = self.player_periods[LABEL_START_DT].tolist()[1:] + [self.player_periods[LABEL_END_DT].iloc[0]]
            perms_str[LABEL_PLAYER_PERIOD] = pd.cut(perms_str.index, bins, labels=self.player_periods.index[1:])
            
            base_perm_list = []
            for i in self.player_periods.index[1:]:
                period_perms_str = perms_str[perms_str[LABEL_PLAYER_PERIOD] == i]
                if period_perms_str.empty:
                    continue

                session = self.player_periods.at[i, LABEL_SESSION]
                period_perms_str[LABEL_SESSION] = session
                period_perms_str[LABEL_FORM_PERIOD] = session

                period_start_dt = self.player_periods.at[i, LABEL_START_DT]
                offset = f'{period_start_dt.minute * SCALAR_TIME + period_start_dt.second}S'
                resampler = period_perms_str.resample('5T', closed='right', offset=offset)

                base_perms = resampler.apply(SoccerCPD.most_common).reset_index()
                base_perms[LABEL_END_DT] = base_perms[LABEL_DATETIME].shift(-1)
                base_perms.iat[-1, -1] = self.player_periods.at[i, LABEL_END_DT]
                base_perm_list.append(base_perms)
            
            role_periods = pd.concat(base_perm_list, ignore_index=True)
            role_periods.rename(columns={LABEL_DATETIME: LABEL_START_DT}, inplace=True)

            perms_list = role_periods[LABEL_PERM].apply(lambda x: np.fromstring(x[1:-1], dtype=int, sep=' '))
            role_periods[LABEL_BASE_PERM] = perms_list.apply(lambda perm: dict(zip(np.arange(10) + 1, perm)))

            role_periods[LABEL_ACTIVITY_ID] = self.activity_id
            role_periods[LABEL_ROLE_PERIOD] = role_periods.index + 1
            role_periods[LABEL_DURATION] = (
                role_periods[LABEL_END_DT] - role_periods[LABEL_START_DT]
            ).apply(lambda td: td.total_seconds())
            self.role_periods = role_periods[HEADER_ROLE_PERIODS]

        self.form_periods.set_index(LABEL_FORM_PERIOD, inplace=True)
        self.role_periods.set_index(LABEL_ROLE_PERIOD, inplace=True)

        # Label formation and role periods to the timestamps in fgp_df
        match_end_dt = self.player_periods[LABEL_END_DT].iloc[-1]
        form_bins = self.form_periods[LABEL_START_DT].tolist() + [match_end_dt]
        role_bins = self.role_periods[LABEL_START_DT].tolist() + [match_end_dt]
        self.fgp_df[LABEL_FORM_PERIOD] = pd.cut(
            self.fgp_df[LABEL_DATETIME], bins=form_bins, labels=self.form_periods.index
        )
        self.fgp_df[LABEL_ROLE_PERIOD] = pd.cut(
            self.fgp_df[LABEL_DATETIME], bins=role_bins, labels=self.role_periods.index
        )

        # Reflect the instructed roles and recompute switch rates in fgp_df
        self.fgp_df = self.fgp_df.apply(self.reassign_base_role, axis=1)
        self.fgp_df = self.fgp_df.groupby(LABEL_DATETIME).apply(SoccerCPD.recompute_switch_rate)
        self.fgp_df, self.form_periods = SoccerCPD.align_formations(self.fgp_df, self.form_periods)
        self.fgp_df = pd.merge(
            self.fgp_df, self.match.roster[HEADER_ROSTER]
        ).sort_values(by=[LABEL_PLAYER_ID, LABEL_DATETIME], ignore_index=True)

        self.form_periods = self.form_periods.reset_index()[HEADER_FORM_PERIODS]
        self.role_periods = self.role_periods.reset_index()[HEADER_ROLE_PERIODS]
        self.role_records = self.generate_role_records()
        print()
        print('-' * 78)
        print('Formation Periods:')
        print(self.form_periods[HEADER_FORM_PERIODS[1:-2]])
        print()
        print('Role Periods:')
        print(self.role_periods[HEADER_ROLE_PERIODS[1:-1]])
        print()

    def visualize(self):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
        import math
        sns.set(font='Arial', rc={'axes.unicode_minus': False}, font_scale=1.5)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        gs = gridspec.GridSpec(2, 4, left=0.05, right=0.95, wspace=0.3, hspace=0.1)

        xlim = 3500
        ylim = 4000

        for idx, form_period in enumerate(self.form_periods[LABEL_FORM_PERIOD][:4]):
            fgp_df = self.fgp_df[(self.fgp_df[LABEL_FORM_PERIOD] == form_period) & (self.fgp_df[LABEL_ROLE].notna())]
            role_coords = np.dot(self.form_periods.at[idx, LABEL_COORDS], [[0, 1], [-1, 0]])
            edge_mat = self.form_periods.at[idx, LABEL_EDGE_MAT]

            plt.subplot(gs[0, idx])
            role_period_records = self.role_periods[self.role_periods[LABEL_FORM_PERIOD] == form_period]
            role_period_from = role_period_records[LABEL_ROLE_PERIOD].iloc[0]
            if len(role_period_records) == 1:
                plt.title(f'Role Period {role_period_from}', fontsize=20)
            else:
                role_period_to = role_period_records[LABEL_ROLE_PERIOD].iloc[-1]
                plt.title(f'Role Periods {role_period_from}-{role_period_to}', fontsize=20)

            plt.scatter(-fgp_df[LABEL_Y_NORM], fgp_df[LABEL_X_NORM],
                        c=fgp_df[LABEL_ROLE], vmin=0.5, vmax=10.5, cmap='tab10', alpha=0.4, zorder=0)
            plt.scatter(role_coords[:, 0], role_coords[:, 1], s=500, c='w', edgecolors='k', zorder=2)

            for r in np.arange(10):
                plt.annotate(r + 1, xy=role_coords[r], ha='center', va='center', fontsize=20, zorder=3)
                for s in np.arange(10):
                    plt.plot(role_coords[[r, s], 0], role_coords[[r, s], 1],
                             linewidth=edge_mat[r, s] ** 2 * 4, c='k', zorder=1)

            plt.xlim(-xlim, xlim)
            plt.ylim(-ylim, ylim)
            plt.vlines([-xlim, xlim], ymin=-ylim, ymax=ylim, color='k')
            plt.hlines([-ylim, 0, ylim], xmin=-xlim, xmax=xlim, color='k', zorder=1)
            plt.axis('off')

        ax = fig.add_subplot(gs[1, :])
        box = ax.get_position()
        ax.set_position([box.x0 + box.width * 0.1, box.y0, box.width * 0.85, box.height * 0.9])
        plt.title('Timeline of Instructed Roles', fontsize=20)

        role_assigns = self.fgp_df.groupby(LABEL_PLAYER_ID).apply(
            lambda df: df.set_index(LABEL_DATETIME).resample('1T', closed='right', label='left')[
                [LABEL_SESSION, LABEL_GAMETIME, LABEL_BASE_ROLE]].first()
        ).reset_index().dropna()
        role_assigns = pd.merge(self.match.roster, role_assigns)
        role_assigns[LABEL_SESSION] = role_assigns[LABEL_SESSION].astype(int)
        role_assigns[LABEL_GAMETIME] = role_assigns.apply(
            lambda df: f'S{df[LABEL_SESSION]}-{df[LABEL_GAMETIME][:2]}T', axis=1
        )
        role_assigns[LABEL_PLAYER_NAME] = role_assigns.apply(
            lambda df: 'Player No.{:02d}'.format(df[LABEL_SQUAD_NUM]), axis=1
        )
        role_assigns.sort_values(by=LABEL_SQUAD_NUM, inplace=True)
        role_assigns_2d = role_assigns.pivot_table(
            values=LABEL_BASE_ROLE, index=LABEL_PLAYER_NAME, columns=LABEL_GAMETIME, aggfunc='first'
        )
        sns.heatmap(role_assigns_2d, vmin=0.5, vmax=10.5, cmap='tab10', cbar=False)

        duration = 0
        vline_idxs = []
        for idx in self.role_periods.index[:-1]:
            duration += self.role_periods.at[idx, LABEL_DURATION]
            vline_idxs.append(math.ceil(duration / SCALAR_TIME))
        plt.vlines(vline_idxs, ymin=0, ymax=len(role_assigns_2d), colors='k', linestyles='--')

        vline_idxs.append(0)
        plt.xticks(ticks=vline_idxs, labels=role_assigns_2d.columns[vline_idxs].tolist(), rotation=45)
        plt.xlabel('session-time')
        plt.ylabel('player')

        report_dir = f'{self.target_dir}/report'
        report_path = f'{report_dir}/{self.match.record[LABEL_ACTIVITY_ID]}.png'
        if not os.path.exists(f'{self.target_dir}'):
            os.mkdir(f'{self.target_dir}')
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)

        plt.savefig(report_path)
        plt.close(fig)
        print(f"'{report_path}' saving done.")

    def save_stats(self, fgp=True, form=True, role=True):
        if not os.path.exists(f'{self.target_dir}'):
            os.mkdir(f'{self.target_dir}')

        # Save fgp_df
        if fgp:
            fgp_dir = f'{self.target_dir}/fgp'
            if not os.path.exists(fgp_dir):
                os.mkdir(fgp_dir)
            fgp_path = f'{fgp_dir}/{self.activity_id}.csv'
            self.fgp_df.to_csv(fgp_path, index=False, encoding='utf-8-sig')
            print(f"'{fgp_path}' saving done.")

        # Save form_periods
        if form:
            form_dir = f'{self.target_dir}/form'
            if not os.path.exists(form_dir):
                os.mkdir(form_dir)
            form_path = f'{form_dir}/{self.activity_id}.pkl'
            self.form_periods.to_pickle(form_path)
            print(f"'{form_path}' saving done.")

        # save role_records
        if role:
            role_dir = f'{self.target_dir}/role'
            if not os.path.exists(role_dir):
                os.mkdir(role_dir)
            role_path = f'{role_dir}/{self.activity_id}.csv'
            self.role_records.to_csv(role_path, index=False, encoding='utf-8-sig')
            print(f"'{role_path}' saving done.")
