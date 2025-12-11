"""
Simple anomaly detection method in python.
Identifies points along time series data that deviates from the mean value 
to a highly significant degree. 
"""

import numpy as np
import pandas as pd 
import math 
from scipy.stats import zscore
"""
Seaborn config 
"""
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

def main():
    df = pd.read_csv("./data.csv")
    # reading == value rounded to 3 sf
    # tidy the column names
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.lower()

    # filter by sensor
    sensor = df['sensor_name'].unique()
    selected_sensor = "GT1 W"
    features = ['value', 'time', 'sensor_name']
    filtered = df[features]
    filtered = filtered.set_index('time')
    sensor_filtered_df = filtered[filtered['sensor_name'] == selected_sensor]

    sensor_filtered_df['z_score'] = zscore(sensor_filtered_df['value'])
    threshold = 3 # standard for cpaturing highly significant outliers, p<0.001 
    outliers = sensor_filtered_df[(sensor_filtered_df['z_score'].abs() > threshold)]
    # plots trend 
    sns.lineplot(
        x=sensor_filtered_df.index,
        y='value',
        data=sensor_filtered_df,
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
    plt.title(f'Value over time with Z-score Outliers (threshold > {threshold}) for {selected_sensor}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return print(sensor_filtered_df.head(1))

if __name__ == "__main__":
    main() 