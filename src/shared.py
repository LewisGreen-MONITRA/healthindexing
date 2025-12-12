from pathlib import Path

import math 
from scipy.stats import *
import pandas as pd 
import numpy as np


app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / "data.csv")
# reading == value rounded to 3 sf
# tidy the column names
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.lower()

sensor = df['sensor_name'].unique()

features = ['value', 'time', 'sensor_name']
filtered = df[features]
filtered = filtered.set_index('time')

def modified_zscore(x):

    x_med = x.median()

    mad = np.abs(x - x_med).median()

    mz  = 0.6745 * (x - x_med) / mad

    return mz
    