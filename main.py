import os
import pandas as pd
from datetime import datetime
import rpy2.robjects.packages as rpackages
from src.myconstants import *
from src.recordmanager import RecordManager
from src.match import Match
from src.footballcpd import FootballCPD

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)


if __name__ == '__main__':

    # Install and import the R package 'gSeg' to be used in FootballCPD
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    if not rpackages.isinstalled('gSeg'):
        utils.install_packages('gSeg')
    rpackages.importr('gSeg')

    # Select the records of target activities with data files but not analyzed yet
    rm = RecordManager()
    activity_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(DIR_UGP_DATA) if f.endswith('.ugp')]
    activity_records = rm.activity_records[(rm.activity_records[LABEL_DATA_SAVED] == 1) &
                                           (rm.activity_records[LABEL_STATS_SAVED] == 0)]
    print()
    print('Activity Records:')
    print(activity_records)

    for i in activity_records.index:
        tic = datetime.now()
        activity_id = activity_records.at[i, LABEL_ACTIVITY_ID]
        date = activity_records.at[i, LABEL_DATE]
        team_name = activity_records.at[i, LABEL_TEAM_NAME]
        print()
        print('=' * 68)
        print(f'[{i}] activity_id: {activity_id}, date: {date}, team_name: {team_name}')

        activity_args = rm.load_activity_data(activity_id)
        match = Match(*activity_args)
        if match.player_periods[LABEL_PLAYER_IDS].iloc[1:].apply(lambda x: len(x)).max() >= 10:
            match.construct_inplay_df()
            match.rotate_pitch()
            cpd = FootballCPD(match, gseg_type='avg')
            cpd.run()
            if not cpd.fgp_df.empty:
                cpd.visualize()
                cpd.save_stats()
                toc = datetime.now()
                print("The total process takes {:.3f} sec.".format((toc - tic).total_seconds()))
        else:
            print("Not enough players to estimate a formation.")

        # pp.activity_records.at[i, LABEL_STATS_SAVED] = 1
        # pp.save_records(VARNAME_ACTIVITY_RECORDS)
