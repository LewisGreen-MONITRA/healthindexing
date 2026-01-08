"""
Helper functions 

"""
from pathlib import Path

import math 
from scipy.stats import *
from scipy.signal import *
from statsmodels.tsa.seasonal import STL
from sklearn.covariance import EllipticEnvelope

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

unit_Dict = df['units'].to_dict()
sensor_Dict = df['sensor_name'].to_dict()

def modifiedZscore(x):
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
     """
     Plot signal decomposition

     :param arr: sample of values
     """
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

def robustTestStat(y):
     std = np.std(y)
     mu = np.mean(y)
     abs_val = abs(y - mu)
     max_dev = abs(abs_val)
     max_ind = np.argmax(abs_val)
     cal = max_dev / std
     return cal, max_ind

def esd_values(n, j, alpha):
     """
     critical values for esd test
     
     :param n: number of samples
     :param j: iterator
     :param alpha: significance level 
     """
     df = n - j - 1
     if df <= 0:
          raise ValueError("Degrees of freedom must be positive!")
     
     t_quantile = t.ppf(1 - alpha / (2 * (n - j)), df)
     numer = (n - j) * t_quantile
     denom = np.sqrt((df + t_quantile**2) * (n - j + 1))
     return numer / denom


def genESD(x, alpha, max, return_stats: bool):
    """
    general ESD test for outlier detection
    
    :param x: sample of data array
    :param alpha: confidence interval, 0.005
    :param max: max number of outliers
    :param return_stats: return debugging stats 
    :type return_stats: bool
    """
    x = np.asarray(x,  dtype=float).ravel()
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 observations!")
    if max is None:
         max = n // 2
    max = math.floor(min(max, n - 2))

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

def covariance(df):
     """
     Calculate matrix covariance for given sensors of time series data. 
     given a matrix m, each vector within the matrix contains time series data for each 
     sensor, say m1 m2....mi with data taken from each value measured say x y... 
     each of the matricies are created by the data collected at each timestamp 
     
     m =    x  y 
         m1 1  2
         m2 2  1
         m3 1  2
     
     calculate the mean of a given matrix, the average values of each vector within the
     initial matrix m 

     can then calculate the std between means at each vector space 

     need to create the matrix. isolate each value and assign them to a vector/column 
     in the matrix. 

     :param df: Sample 
     """
     



     
def elipticalOutlier(df):
     """
     Implementation of eliptical outlier detection method for each sensor and 
     unit of measure found in the data set     
     
     :param df: Dataframe
     """
     for s in df['sensor_name'].unique():
          for u in df['units'].unique():
               sample = df['value'].to_numpy().reshape(-1, 1)
               clf = EllipticEnvelope(random_state = 42).fit(sample)
               prediction = clf.predict(sample).tolist()
     return prediction 



def spearmansTest(df):
     spearman_corr = df.corr(method='spearman')
     return spearman_corr

     












