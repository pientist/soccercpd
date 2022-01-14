import os
import pandas as pd
from src.myconstants import *


# For convenient data loading and saving
class RecordManager:
    def __init__(self, root_dir=DIR_DATA):
        self.root_dir = root_dir
        self.activity_records = None
        self.player_records = None
        self.player_periods = None
        self.metadata = pd.DataFrame([
            [VARNAME_ACTIVITY_RECORDS, HEADER_ACTIVITY_RECORDS, '.csv'],
            [VARNAME_PLAYER_RECORDS, HEADER_PLAYER_RECORDS, '.csv'],
            [VARNAME_PLAYER_PERIODS, HEADER_PLAYER_PERIODS, '.pkl'],
        ], columns=[LABEL_VARNAME, LABEL_HEADER, LABEL_EXTENSION])
        self._load_records()

    def _load_records(self):
        for i in self.metadata.index:
            metadata = self.metadata.loc[i]
            path = f'{self.root_dir}/{metadata[LABEL_VARNAME]}{metadata[LABEL_EXTENSION]}'
            if not os.path.exists(path):
                if metadata[LABEL_HEADER] is None:
                    records = dict()
                else:
                    records = pd.DataFrame(columns=metadata[LABEL_HEADER])
            elif metadata[LABEL_EXTENSION] == '.csv':
                try:
                    records = pd.read_csv(path, header=0, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    records = pd.read_csv(path, header=0, encoding='cp949')
            else:
                records = pd.read_pickle(path)
            setattr(self, metadata[LABEL_VARNAME], records[metadata[LABEL_HEADER]])

    def save_records(self, attr=VARNAME_ACTIVITY_RECORDS):
        metadata = self.metadata[self.metadata[LABEL_VARNAME] == attr].iloc[0]
        records = getattr(self, attr)
        path = f'{self.root_dir}/{metadata[LABEL_VARNAME]}{metadata[LABEL_EXTENSION]}'
        if metadata[LABEL_EXTENSION] == '.csv':
            records.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"'{path}' saving done.")
        elif metadata[LABEL_EXTENSION] == '.pkl':
            records.to_pickle(path)

    def load_activity_data(self, activity_id):
        activity_record = self.activity_records[self.activity_records[LABEL_ACTIVITY_ID] == activity_id].iloc[0]
        player_periods = self.player_periods[self.player_periods[LABEL_ACTIVITY_ID] == activity_id]
        player_periods = player_periods.set_index(LABEL_PLAYER_PERIOD)[HEADER_PLAYER_PERIODS[2:]]
        player_records = self.player_records[self.player_records[LABEL_ACTIVITY_ID] == activity_id]
        roster = player_records[HEADER_ROSTER].drop_duplicates().set_index(LABEL_PLAYER_ID)
        ugp_df = pd.read_pickle(f'{DIR_UGP_DATA}/{activity_id}.ugp')
        return activity_record, player_periods, roster, ugp_df
