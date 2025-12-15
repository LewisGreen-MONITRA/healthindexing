"""
Helper functions 

"""
from pathlib import Path

import math 
from scipy.stats import *
from scipy.signal import *
from statsmodels.tsa.seasonal import STL
import itertools
import pandas as pd 
import numpy as np

"""
Seaborn config 
"""
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")


df = pd.read_csv("C:/Users/lewis.green/Documents/healthindexing/src/data.csv")
# reading == value rounded to 3 sf
# tidy the column names
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.lower()

sensor = df['sensor_name'].unique()

features = ['value', 'time', 'sensor_name', 'units']
filtered = df[features]
filtered = filtered.set_index('time')

# TODO create dictionary of all units of measure 
# filter plots based on the unit, create new plot something like that 

def modified_zscore(x):
    """
    Implementation of zscore for where n < 30 
    
    :param x: sample, array or list of values
    """
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

def plotSTL(arr):
     stl = STL(arr['value'], period= 12, robust= True)
     result = stl.fit()
     
     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7,4))
    
     ax1.plot(arr.index, result.trend, label='Trend', color='red')
     ax1.set_title('Trend Component')
     ax2.plot(arr.index, result.seasonal, label='Seasonal', color='blue')
     ax2.set_title('Seasonal Component')
     ax3.plot(arr.index, result.resid, label='Residual')
     ax3.set_title('Residual Component')
     plt.tight_layout()
     plt.show()
     
     
     return 

def esd_values(n, j, alpha):
     df = n - j - 1
     if df <= 0:
          raise ValueError("Degrees of freedom must be positive!")
     
     t_quantile = t.ppf(1 - alpha / (2 * (n - j)), df)
     numer = (n - j) * t_quantile
     denom = np.sqrt((df + t_quantile**2) * (n - j + 1))
     return numer / denom


def genESD(x, alpha, max, return_stats: bool):
    x = np.asarray(x,  dtype=float).ravel()
    n = len(x)

    if n < 3:
        raise ValueError("Need at least 3 observations!")
    
    if max is None:
         max = n // 2
    max = min(max, n - 2)

    remaining = np.arange(n)
    outlier_indicies = []
    r_vals = []
    lambda_vals = []
    indicies = []

    for j in range(1, max + 1):
         subset = x[remaining]
         mu = np.mean(subset)
         simga = np.std(subset, ddof=1)
         if simga == 0:
              break
         resid = np.abs(subset - mu) / simga
         max_local = np.argmax(resid)
         r_j = resid[max_local]
         lambda_j = esd_values(n, j ,alpha)

         r_vals.append(r_j)
         lambda_vals.append(lambda_j)
         indicies.append(remaining[max_local])

         if r_j > lambda_j:
              outlier_indicies.append(remaining[max_local])
              remaining = remaining[remaining != remaining[max_local]]
              if len(remaining) < 2:
                   break
              else:
                   break
    outlier_indicies = np.array(outlier_indicies, dtype=int)
    outlier_values = x[outlier_indicies]

    if return_stats: 
         return outlier_indicies, outlier_values, {
              "R": np.array(r_vals),
              "lambda": np.array(lambda_vals),
              "candidate": np.array(indicies)         
        }
    else: 
         return outlier_indicies, outlier_values
    
def lightESD(arr):
    """
    Implementation of lightESD as outlined by Das & Luo 20223
    https://arxiv.org/pdf/2305.12266
    
    :param arr: time series data 
    """
    outlier_index = []

    def periodDetection(arr):
         max_power = []
         for i in range(1, 100):         
            arr_i = list(itertools.permutations(arr))
            freq, pow = welch(arr_i)
            Pmax = max(pow)
            max_power.append(Pmax)
         max_power = max_power.sort()
         index = 0.99 * len(max_power)
         thresh = max_power.index
         freq, pow = welch(arr)
         prd = -1 
         temp_psd = -1
         for j in len(pow) - 1:
              if pow[j] > thresh & pow[j] > pow[j-1] & pow[j] > pow[j+1]:
                   if pow[j] > temp_psd:
                        prd +=(1/freq[j])
                        temp_psd = pow[j]
         if prd == -1: 
              prd = 1
         return prd
    
    period = periodDetection(arr)
    if period == 1:
          x += 1
    else: 
        x =+ 1  
    a_max = 0.1 * len(arr)
    outliers = genESD(arr, alpha=0.05, max= a_max, return_stats=True)
    outlier_index = outliers.index 
    if outliers[1] == True & outliers[2] == False:
         outliers[1] = False
         outlier_index.drop[1]

    anomalies = outlier_index
    return anomalies

