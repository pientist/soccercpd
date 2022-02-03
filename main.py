import os
import pandas as pd
from datetime import datetime

# Uncomment this code if it raises an error
# os.environ['R_HOME'] = "C:\Program Files\R\R-3.6.0"  # or whereever your R is installed"

import rpy2.robjects.packages as rpackages
from src.myconstants import *
from src.record_manager import RecordManager
from src.match import Match
from src.soccercpd import SoccerCPD

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)


if __name__ == '__main__':

    # Install and import the R package 'gSeg' to be used in SoccerCPD
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    if not rpackages.isinstalled('gSeg'):
        utils.install_packages('gSeg')
    rpackages.importr('gSeg')

    # Select the records of target matches (having data files but not analyzed yet)
    rm = RecordManager()
    activity_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(DIR_UGP_DATA) if f.endswith('.ugp')]
    activity_records = rm.activity_records[(rm.activity_records[LABEL_DATA_SAVED] == 1) &
                                           (rm.activity_records[LABEL_STATS_SAVED] == 0)]
    print()
    print('Activity Records:')
    print(activity_records)

    # Perform SoccerCPD per match
    outliers = pd.read_csv('data/outliers.csv', header=0) if os.path.exists('data/outliers.csv') else None
    for i in activity_records.index[:10]:
        tic = datetime.now()
        activity_id = activity_records.at[i, LABEL_ACTIVITY_ID]
        date = activity_records.at[i, LABEL_DATE]
        team_name = activity_records.at[i, LABEL_TEAM_NAME]
        print()
        print('=' * 78)
        print(f'[{i}] activity_id: {activity_id}, date: {date}, team_name: {team_name}')

        activity_args = rm.load_activity_data(activity_id)
        activity_outliers = None
        if outliers is not None:
            activity_outliers = outliers[outliers[LABEL_ACTIVITY_ID] == activity_id][LABEL_PLAYER_ID].tolist()
        match = Match(*activity_args, outliers=activity_outliers)

        if match.player_periods[LABEL_PLAYER_IDS].iloc[1:].apply(lambda x: len(x)).max() >= 10:
            # Filter in-play data from the measured data using the start, end, and substitution records
            match.construct_inplay_df()

            # Rotate the pitch for one of the sessions so that the team always attacks from left to right
            match.rotate_pitch()

            # Apply SoccerCPD on the preprocessed match data
            cpd = SoccerCPD(match, formcpd_type='gseg_avg')
            cpd.run()
            if not cpd.fgp_df.empty:
                cpd.visualize()
                cpd.save_stats()
                toc = datetime.now()
                print("The total process takes {:.3f} sec.".format((toc - tic).total_seconds()))

        else:
            # If at least one player has not been measured during the match,
            # skip the process for the match since the data is incomplete
            print("Not enough players to estimate a formation.")

        # Set 'stats_saved' value for the match to 1 to avoid redundant executions
        # rm.activity_records.at[i, LABEL_STATS_SAVED] = 1
        # rm.save_records(VARNAME_ACTIVITY_RECORDS)
