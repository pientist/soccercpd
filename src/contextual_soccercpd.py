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


class ContextualSoccerCPD(SoccerCPD):
    def __init__(
        self, match, apply_cpd=True, formcpd_type='gseg_avg', rolecpd_type='gseg_avg',
        max_sr=MAX_SWITCH_RATE, max_pval=MAX_PVAL, min_pdur=MIN_PERIOD_DUR, min_fdist=MIN_FORM_DIST
    ):
        super().__init__(match, apply_cpd, formcpd_type, rolecpd_type, max_sr, max_pval, min_pdur, min_fdist)
