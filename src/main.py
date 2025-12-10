import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from scipy.stats import zscore
from math import sqrt

"""
Seaborn config 
"""
import seaborn as sns 
sns.set_theme(style="whitegrid")

def main():
    
    df = pd.read_csv("C:/Users/ldgre/Desktop/wfh/healthindexing/data/data.csv")
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
    threshold = 2 # standard for cpaturing highly significant outliers
    outliers = sensor_filtered_df[(sensor_filtered_df['z_score'].abs() > threshold)]
    
    sns.lineplot(
        x=sensor_filtered_df.index,
        y='value',
        data=sensor_filtered_df,
        label='Normal Data',
        color='blue',
        linewidth=1.5
    )          
  
    sns.scatterplot(
        x=outliers.index,
        y='value',
        data=outliers,
        label='Outlier',
        color='red',
        s=100, # Adjust size for visibility
        marker='X' # Use a distinct marker
    )
    # displaying one continuos line plot, multiple values assignged to the same index?
    # doing this on matplotlib, test with sns 
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