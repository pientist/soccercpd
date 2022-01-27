# File paths and variable names
DIR_DATA = './data'
DIR_UGP_DATA = f'{DIR_DATA}/ugp'
DIR_TEMP_DATA = f'{DIR_DATA}/temp'

VARNAME_ACTIVITY_RECORDS = 'activity_records'
VARNAME_PLAYER_RECORDS = 'player_records'
VARNAME_PLAYER_PERIODS = 'player_periods'

# Column names and headers
LABEL_ID = 'id'
LABEL_NAME = 'name'
LABEL_VARNAME = 'varname'
LABEL_RECORDS = 'records'
LABEL_HEADER = 'header'
LABEL_DTYPES = 'dtypes'
LABEL_PATH = 'path'
LABEL_FILE = 'file'
LABEL_EXTENSION = 'extension'

LABEL_ACTIVITY_ID = 'activity_id'
LABEL_TEAM_ID = 'team_id'
LABEL_TYPE = 'type'
LABEL_DATE = 'date'
LABEL_TEAM_NAME = 'team_name'
LABEL_HOME_AWAY = 'home_away'
LABEL_ROTATED_SESSION = 'rotated_session'
LABEL_DATA_SAVED = 'data_saved'
LABEL_STATS_SAVED = 'stats_saved'
HEADER_ACTIVITY_RECORDS = [
    LABEL_ACTIVITY_ID, LABEL_TEAM_ID, LABEL_TYPE, LABEL_DATE, LABEL_TEAM_NAME,
    LABEL_HOME_AWAY, LABEL_ROTATED_SESSION, LABEL_DATA_SAVED, LABEL_STATS_SAVED
]

LABEL_PLAYER_ID = 'player_id'
LABEL_SQUAD_NUM = 'squad_num'
LABEL_PLAYER_NAME = 'player_name'
HEADER_ROSTER = [LABEL_PLAYER_ID, LABEL_SQUAD_NUM, LABEL_PLAYER_NAME]
HEADER_PLAYER_RECORDS = [LABEL_ACTIVITY_ID, LABEL_DATE, LABEL_TEAM_NAME] + HEADER_ROSTER

LABEL_PLAYER_PERIOD = 'player_period'
LABEL_SESSION = 'session'
LABEL_GAMETIME = 'gametime'
LABEL_START_DT = 'start_dt'
LABEL_END_DT = 'end_dt'
LABEL_DURATION = 'duration'
LABEL_PLAYER_IDS = 'player_ids'
HEADER_PLAYER_PERIODS = [
    LABEL_ACTIVITY_ID, LABEL_PLAYER_PERIOD, LABEL_TYPE, LABEL_SESSION, LABEL_GAMETIME,
    LABEL_START_DT, LABEL_END_DT, LABEL_DURATION, LABEL_PLAYER_IDS
]

LABEL_DATETIME = 'datetime'
LABEL_UNIXTIME = 'unixtime'
LABEL_X = 'x'
LABEL_Y = 'y'
LABEL_SPEED = 'speed'
HEADER_UGP = [
    LABEL_PLAYER_ID, LABEL_SESSION, LABEL_GAMETIME, LABEL_UNIXTIME,
    LABEL_PLAYER_PERIOD, LABEL_DURATION, LABEL_X, LABEL_Y, LABEL_SPEED
]

LABEL_INDEX = 'index'
LABEL_FORM_PERIOD = 'form_period'
LABEL_ROLE_PERIOD = 'role_period'
LABEL_X_NORM = 'x_norm'
LABEL_Y_NORM = 'y_norm'
LABEL_ROLE = 'role'
LABEL_BASE_ROLE = 'base_role'
LABEL_ALIGNED_ROLE = 'aligned_role'
LABEL_SWITCH_RATE = 'switch_rate'
HEADER_FGP = [
    LABEL_PLAYER_ID, LABEL_SESSION, LABEL_GAMETIME,
    LABEL_PLAYER_PERIOD, LABEL_FORM_PERIOD, LABEL_ROLE_PERIOD,
    LABEL_X, LABEL_Y, LABEL_X_NORM, LABEL_Y_NORM,
    LABEL_ROLE, LABEL_BASE_ROLE, LABEL_SWITCH_RATE
]

LABEL_COORDS = 'coords'
LABEL_EDGE_MAT = 'edge_mat'
LABEL_BASE_PERM = 'base_perm'
LABEL_DISTN = 'distn'
LABEL_CLUSTER = 'cluster'
LABEL_FORMATION = 'formation'
HEADER_FORM_PERIODS = [
    LABEL_SESSION, LABEL_FORM_PERIOD, LABEL_START_DT, LABEL_END_DT,
    LABEL_DURATION, LABEL_COORDS, LABEL_EDGE_MAT
]
HEADER_ROLE_PERIODS = [
    LABEL_SESSION, LABEL_FORM_PERIOD, LABEL_ROLE_PERIOD,
    LABEL_START_DT, LABEL_END_DT, LABEL_DURATION, LABEL_BASE_PERM
]
HEADER_ROLE_ALIGNS = [
    LABEL_ACTIVITY_ID, LABEL_FORM_PERIOD, LABEL_BASE_ROLE, LABEL_ALIGNED_ROLE, LABEL_X, LABEL_Y
]
HEADER_ROLE_RECORDS = [
    LABEL_ACTIVITY_ID, LABEL_SESSION, LABEL_PLAYER_PERIOD, LABEL_FORM_PERIOD, LABEL_ROLE_PERIOD,
    LABEL_START_DT, LABEL_END_DT, LABEL_DURATION,
    LABEL_PLAYER_ID, LABEL_SQUAD_NUM, LABEL_PLAYER_NAME, LABEL_BASE_ROLE, LABEL_X, LABEL_Y
]

# Numeric constants
SCALAR_CENTI = 100
SCALAR_MILLI = 1000
SCALAR_MICRO = 1000000
SCALAR_TIME = 60

# Hyperparameters for SoccerCPD
MAX_SWITCH_RATE = 0.6
MAX_PVAL = 0.01
MIN_PERIOD_DUR = 300
MIN_FORM_DIST = 7
