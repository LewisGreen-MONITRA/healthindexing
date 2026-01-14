"""
Helper functions 

"""
from pathlib import Path

import math 
from scipy.stats import *
from scipy.signal import *
from statsmodels.tsa.seasonal import STL
from sklearn.covariance import EllipticEnvelope, LedoitWolf

import pandas as pd 
import numpy as np

"""
Seaborn config 
"""
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

def getData():
     # change path as needed 
     df = pd.read_csv("C:/Users/lewis.green/Documents/healthindexing/src/data.csv")

     # reading == value rounded to 3 sf
     # tidy the column names
     df.columns = df.columns.str.replace(" ", "_")
     df.columns = df.columns.str.lower()

     sensor = df['sensor_name'].unique()

     features = ['value', 'time', 'sensor_name', 'units']
     filtered = df[features]
     filtered = filtered.set_index('time')

     return filtered


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
    
def empriicalCovar(X, unbiased = True):
    X = np.asarray(X)
    if X.ndim != 2: 
        raise ValueError("Input must be 2d: (time and value recorded for a given sensor)")
    n = X.shape[0]
    mean = np.mean(X, axis= 0)
    Xc = X - mean
    denominator = n - 1 if unbiased else n
    cov = (Xc.T @ Xc) / denominator
    return cov 

def rollingCovar(X , window):
   X = np.asarray(X)
   n, d = X.shape
   covs = np.zeros((n - window + 1 , d, d))
   for i in range(n - window + 1):
       covs[i] = empriicalCovar(X[i:i+window])
   return covs 

def shrinkageCovar(x):
     """
     sklearn has a robust approach to covariance shrinkage 
     
     :param x: values 
     """
     lw = LedoitWolf()
     lw.fit(x)
     cov_shrunk = lw.covariance_ 
     return cov_shrunk 

def elipticalOutlier(df):
     """
     Implementation of eliptical outlier detection method for each sensor and 
     unit of measure found in the data set     
     
     :param df: Dataframe
     """
     for s in df['sensor_name'].unique():
          for u in df['units'].unique():
               sample = df['value'].to_numpy().reshape(-1, 1)
               clf = EllipticEnvelope(random_state = 42, support_fraction = 0.5).fit(sample)
               prediction = clf.predict(sample).tolist()
     return prediction 

def spearmansTest(df):
     spearman_corr = df.corr(method='spearman')
     return spearman_corr

def kalmanZscoreHybrid(df, sensor_name, units="V", z_threshold=2.0, nis_threshold=6.33, 
                       window_size=15, process_noise=1.0, measurement_noise=0.01):
    """
    Hybrid anomaly detection combining Kalman filtering with z-score analysis.
    
    :param df: full dataframe
    :param sensor_name: sensor to filter by
    :param units: units to filter by
    :param z_threshold: z-score threshold
    :param nis_threshold: NIS threshold
    :param window_size: window for rolling z-score
    :param process_noise: Kalman process noise
    :param measurement_noise: Kalman measurement noise
    :return: dataframe with anomaly indicators and metrics
    """
    from kalman import detectAnomaliesKalman
    
    # Filter data
    sensor_df = df[df['sensor_name'] == sensor_name].copy()
    sensor_df = sensor_df[sensor_df['units'] == units].copy()
    
    if len(sensor_df) == 0:
        return None
    
    # Run detection
    results = detectAnomaliesKalman(
        sensor_df['value'].values,
        z_score_threshold=z_threshold,
        nis_threshold=nis_threshold,
        window_size=window_size,
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )
    
    # Add results to dataframe
    sensor_df['kalman_residual'] = results['residuals']
    sensor_df['nis'] = results['nis_values']
    sensor_df['kalman_zscore'] = results['z_scores']
    sensor_df['is_anomaly'] = results['anomaly_flags']
    
    return sensor_df



