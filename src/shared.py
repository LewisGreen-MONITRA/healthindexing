from pathlib import Path

import math 
from scipy.stats import *
import pandas as pd 
import numpy as np

"""
Seaborn config 
"""
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

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

def plotResults(df, outliers):
        """
        Helper function to plot time series health indexing data

        :param df: dataset 
        :param outliers: outliers detected through z_test 
        """  
        # plots trend 
        sns.lineplot(
            x=df.index,
            y='value',
            data=df,
            label='Normal Data',
            color='blue',
            linewidth=1.5
        )          
        # plot outliers
        sns.scatterplot(
            x=outliers.index,
            y='value',
            data=outliers,
            label='Outlier',
            color='red',
            s=100,
            marker='o'
        )
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
